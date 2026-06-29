"""
Lumen — Backend proxy sécurisé vers l'API Groq.

Rôle : le frontend (PWA publique) n'a JAMAIS la clé Groq. Il envoie l'image
ici, ce serveur appelle Groq, et renvoie uniquement le JSON de la fiche.

Sécurité intégrée :
  - La clé Groq vit uniquement dans la variable d'environnement GROQ_API_KEY.
  - CORS restreint aux origines déclarées (ALLOWED_ORIGINS).
  - Validation stricte du type et de la taille de l'image.
  - Rate limiting par IP (fenêtre glissante en mémoire).
  - Erreurs upstream « aplaties » : aucune fuite de la clé ni du prompt.
  - Timeouts sur les appels Groq.

Lancer en local :
    pip install -r requirements.txt
    export GROQ_API_KEY=gsk_...
    uvicorn main:app --reload --port 8000
"""

import os
import json
import time
import base64
import logging
from collections import defaultdict, deque

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from groq import Groq

# --------------------------------------------------------------------------- #
# Configuration (tout vient de l'environnement, rien n'est en dur)
# --------------------------------------------------------------------------- #

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("lumen")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    # On échoue tôt et clairement plutôt que de renvoyer des 500 obscurs.
    raise RuntimeError("GROQ_API_KEY manquante. Définis-la avant de démarrer.")

# Origines autorisées : ta page locale + ton domaine GitHub Pages.
# Exemple : ALLOWED_ORIGINS="http://localhost:5500,https://lechat45.github.io"
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:5500,http://127.0.0.1:5500").split(",")
    if o.strip()
]

MAX_IMAGE_MB = float(os.environ.get("MAX_IMAGE_MB", "4"))          # limite base64 Groq = 4 Mo
MAX_IMAGE_BYTES = int(MAX_IMAGE_MB * 1024 * 1024)
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "20"))

VISION_MODEL = os.environ.get("VISION_MODEL", "qwen/qwen3.6-27b")   # multimodal, JSON mode
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "groq/compound")    # recherche web intégrée

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}

client = Groq(api_key=GROQ_API_KEY, timeout=90.0)

app = FastAPI(title="Lumen API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# --------------------------------------------------------------------------- #
# Rate limiting simple par IP (mémoire process).
# Pour un déploiement multi-instances, remplace par Redis.
# --------------------------------------------------------------------------- #

_hits: dict[str, deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    # Render/derrière proxy : on lit X-Forwarded-For en priorité.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(request: Request) -> None:
    ip = _client_ip(request)
    now = time.time()
    window = _hits[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="Trop de requêtes. Réessaie dans un instant.")
    window.append(now)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

VISION_SYSTEM = (
    "Tu es un module d'identification visuelle. On te donne une photo prise "
    "dans le monde réel (logo d'entreprise, monument, plaque, texte ou image "
    "d'un événement historique). Tu réponds UNIQUEMENT avec un objet JSON "
    "valide, sans texte ni markdown autour.\n"
    "Schéma exact :\n"
    "{\"type\":\"entreprise|histoire|autre\",\"nom_probable\":\"\","
    "\"candidats\":[\"max 3 si tu hésites\"],\"texte_ocr\":\"texte lisible sur l'image\","
    "\"indices_visuels\":\"couleurs, symboles, style\",\"confiance\":0}\n"
    "confiance est un entier 0-100. Si tu ne reconnais pas, mets une confiance "
    "basse et propose des candidats. N'invente jamais un nom avec une fausse certitude."
)

SUMMARY_SYSTEM = (
    "Tu es un rédacteur encyclopédique pour une app mobile de scan intelligent. "
    "À partir d'un SUJET identifié (entreprise ou événement historique), tu "
    "produis une fiche captivante, factuelle et concise, en {langue}.\n\n"
    "RÈGLES ABSOLUES\n"
    "- Tu réponds UNIQUEMENT avec un objet JSON valide. Aucun texte avant ou "
    "après, aucun bloc markdown, aucun commentaire.\n"
    "- Tu utilises la recherche web pour VÉRIFIER les faits et chiffres. Tu ne "
    "cites QUE des sources réellement consultées, avec leur URL exacte. Tu "
    "n'inventes JAMAIS d'URL ni de statistique.\n"
    "- Si une donnée est incertaine ou introuvable, tu l'omets plutôt que de la "
    "deviner, et tu le signales dans \"avertissement\".\n"
    "- Ton captivant mais sobre : phrases courtes, zéro remplissage.\n\n"
    "SCHÉMA DE SORTIE (respecte exactement les clés)\n"
    "{\n"
    '  "sujet": "nom exact identifié",\n'
    '  "categorie": "entreprise | histoire | inconnu",\n'
    '  "confiance": 0,\n'
    '  "accroche": "une phrase qui donne envie de lire",\n'
    '  "resume": "2 à 4 paragraphes structurés",\n'
    '  "chiffres_cles": [{"label": "", "valeur": "", "annee": "ou null"}],\n'
    '  "chronologie": [{"date": "", "evenement": ""}],\n'
    '  "impact": "pourquoi c\'est important aujourd\'hui",\n'
    '  "le_saviez_vous": ["fait surprenant et vérifié"],\n'
    '  "sources": [{"titre": "", "url": "", "fiabilite": "officielle | encyclopedique | presse | autre"}],\n'
    '  "pour_approfondir": ["piste de lecture ou mot-clé"],\n'
    '  "avertissement": null\n'
    "}\n\n"
    "CONTRAINTES PAR CATÉGORIE\n"
    "- entreprise : fondation, fondateurs, secteur, chiffres clés (CA, effectif, "
    "valorisation) AVEC année, faits marquants.\n"
    "- histoire : contexte, dates clés dans \"chronologie\", causes, conséquences, postérité.\n"
    "- inconnu : champs vides, confiance basse, explication dans \"avertissement\"."
)


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #

def _extract_json(text: str) -> dict:
    """Parse robuste : tolère un éventuel bloc ```json ou du texte parasite."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


# --------------------------------------------------------------------------- #
# Modèles de requête
# --------------------------------------------------------------------------- #

class SummarizeBody(BaseModel):
    sujet: str = Field(min_length=1, max_length=200)
    categorie: str = "autre"
    langue: str = "français"


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/")
def health():
    return {"status": "ok", "service": "lumen", "models": {"vision": VISION_MODEL, "summary": SUMMARY_MODEL}}


@app.post("/api/identify")
async def identify(request: Request, image: UploadFile = File(...), mode: str = Form("auto")):
    """Étage 1 — identifie le sujet sur la photo. Rapide et focalisé."""
    _check_rate_limit(request)

    if image.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=415, detail="Format non supporté (JPEG, PNG ou WebP).")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Image vide.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"Image trop lourde (max {MAX_IMAGE_MB:.0f} Mo).")

    b64 = base64.b64encode(data).decode("utf-8")
    data_uri = f"data:{image.content_type};base64,{b64}"

    indication = ""
    if mode == "entreprise":
        indication = " L'utilisateur indique qu'il s'agit probablement d'une entreprise/logo."
    elif mode == "histoire":
        indication = " L'utilisateur indique qu'il s'agit probablement d'un sujet historique."

    try:
        completion = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": VISION_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Identifie le sujet de cette image." + indication},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
            temperature=0.2,
            max_completion_tokens=600,
            response_format={"type": "json_object"},
        )
        result = _extract_json(completion.choices[0].message.content)
    except Exception as exc:  # noqa: BLE001
        log.error("Erreur identify: %s", exc)
        raise HTTPException(status_code=502, detail="L'identification a échoué. Réessaie.")

    return JSONResponse(result)


@app.post("/api/summarize")
def summarize(request: Request, body: SummarizeBody):
    """Étage 2 — rédige la fiche sourcée (recherche web réelle)."""
    _check_rate_limit(request)

    system = SUMMARY_SYSTEM.replace("{langue}", body.langue)
    user = (
        f"Sujet : {body.sujet}\n"
        f"Catégorie présumée : {body.categorie}\n"
        f"Rédige la fiche en {body.langue}, en respectant le schéma JSON imposé."
    )

    try:
        completion = client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.4,
            max_completion_tokens=4000,
        )
        fiche = _extract_json(completion.choices[0].message.content)
    except json.JSONDecodeError:
        log.error("JSON invalide depuis le modèle de résumé")
        raise HTTPException(status_code=502, detail="La fiche n'a pas pu être structurée. Réessaie.")
    except Exception as exc:  # noqa: BLE001
        log.error("Erreur summarize: %s", exc)
        raise HTTPException(status_code=502, detail="La génération de la fiche a échoué. Réessaie.")

    return JSONResponse(fiche)
