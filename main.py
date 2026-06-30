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
import httpx
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

VISION_MODEL = os.environ.get("VISION_MODEL", "qwen/qwen3.6-27b")   # vision Groq (repli)
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "groq/compound")    # Groq : recherche web réelle (gratuit)

# --- Gemini : identification (logos/marques/monuments) ------------------- #
# Si GEMINI_API_KEY est défini, l'identification passe par Gemini (bien meilleur).
# La rédaction, elle, utilise Groq compound (vraies sources web, gratuit),
# avec repli sur Gemini si Groq échoue — pour toujours renvoyer une fiche.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

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
    "RÈGLES\n"
    "- Reste strictement factuel : n'invente aucun chiffre ni aucune source.\n"
    "- Si une donnée est incertaine ou introuvable, laisse le champ vide et "
    "explique-le brièvement dans le champ avertissement.\n"
    "- Style captivant mais sobre : phrases courtes, zéro remplissage.\n\n"
    "CONTENU ATTENDU DE CHAQUE CHAMP\n"
    "- sujet : le nom exact identifié (texte simple, ex. « Lego »).\n"
    "- categorie : « entreprise », « histoire » ou « inconnu ».\n"
    "- confiance : un nombre de 0 à 100 selon ta certitude.\n"
    "- accroche : une seule phrase qui donne envie de lire.\n"
    "- resume : 2 à 4 courts paragraphes structurés.\n"
    "- chiffres_cles : faits chiffrés (label, valeur, et année si pertinent).\n"
    "- chronologie : dates clés (date + événement), surtout pour l'histoire.\n"
    "- impact : pourquoi le sujet compte aujourd'hui.\n"
    "- le_saviez_vous : faits surprenants et vérifiés.\n"
    "- sources : titres et URL réelles (laisse vide si tu n'es pas sûr).\n"
    "- pour_approfondir : pistes de lecture ou mots-clés.\n"
    "- avertissement : tout doute ou limite (sinon laisse vide).\n\n"
    "CONTRAINTES PAR CATÉGORIE\n"
    "- entreprise : fondation, fondateurs, secteur, chiffres clés (CA, effectif, "
    "valorisation) AVEC année, faits marquants.\n"
    "- histoire : contexte, dates clés dans la chronologie, causes, conséquences, postérité.\n"
    "- inconnu : champs vides, confiance basse, explication dans avertissement."
)


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #

def _extract_json(text: str) -> dict:
    """Parse robuste : tolère un bloc ```json, du texte parasite ou un bloc <think>."""
    text = (text or "").strip()
    # Certains modèles (mode "thinking") préfixent un bloc <think>...</think>
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
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

def _empty_identify() -> dict:
    """Résultat neutre quand l'identification n'aboutit pas (au lieu d'une 502)."""
    return {
        "type": "autre",
        "nom_probable": "",
        "candidats": [],
        "texte_ocr": "",
        "indices_visuels": "",
        "confiance": 0,
    }


def gemini_identify(b64: str, mime: str, indication: str) -> dict:
    """Identification via Gemini — bien meilleure pour les logos, marques, monuments."""
    url = GEMINI_URL.format(model=GEMINI_MODEL)
    payload = {
        "systemInstruction": {"parts": [{"text": VISION_SYSTEM}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "Identifie le sujet de cette image." + indication},
                    {"inlineData": {"mimeType": mime, "data": b64}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 4096,   # marge pour le raisonnement + le JSON
            "responseMimeType": "application/json",
        },
    }
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}
    with httpx.Client(timeout=45) as http:
        resp = http.post(url, json=payload, headers=headers)
    if resp.status_code != 200:
        # On laisse le message Gemini remonter dans les logs (modèle introuvable, quota…).
        raise RuntimeError(f"Gemini {resp.status_code}: {resp.text[:300]}")
    candidates = resp.json().get("candidates") or []
    if not candidates:
        return _empty_identify()
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        return _empty_identify()
    try:
        return _extract_json(text)
    except json.JSONDecodeError:
        return _empty_identify()


# Schéma imposé à Gemini : il DOIT renvoyer un JSON conforme -> plus d'erreur de parsing.
FICHE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "sujet": {"type": "STRING"},
        "categorie": {"type": "STRING"},
        "confiance": {"type": "INTEGER"},
        "accroche": {"type": "STRING"},
        "resume": {"type": "STRING"},
        "chiffres_cles": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "label": {"type": "STRING"},
                    "valeur": {"type": "STRING"},
                    "annee": {"type": "STRING"},
                },
            },
        },
        "chronologie": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "date": {"type": "STRING"},
                    "evenement": {"type": "STRING"},
                },
            },
        },
        "impact": {"type": "STRING"},
        "le_saviez_vous": {"type": "ARRAY", "items": {"type": "STRING"}},
        "sources": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "titre": {"type": "STRING"},
                    "url": {"type": "STRING"},
                    "fiabilite": {"type": "STRING"},
                },
            },
        },
        "pour_approfondir": {"type": "ARRAY", "items": {"type": "STRING"}},
        "avertissement": {"type": "STRING"},
    },
}


WIKI_CHARS = 5000  # longueur d'extrait Wikipédia envoyée à Gemini


def _wiki_lang(langue: str) -> str:
    """Déduit le code Wikipédia (fr, en, …) à partir de la langue demandée."""
    l = (langue or "").lower()
    if l.startswith("en") or "angl" in l or "english" in l:
        return "en"
    if l.startswith("es") or "espagn" in l or "spanish" in l:
        return "es"
    if l.startswith("de") or "allem" in l or "german" in l:
        return "de"
    if l.startswith("it") or "ital" in l:
        return "it"
    return "fr"


def wikipedia_context(sujet: str, langue: str) -> dict | None:
    """Cherche le sujet sur Wikipédia et renvoie {titre, url, extrait} ou None.
    API publique, sans clé, sans facturation."""
    lang = _wiki_lang(langue)
    api = f"https://{lang}.wikipedia.org/w/api.php"
    try:
        with httpx.Client(
            timeout=15,
            headers={
                # Wikimedia exige un User-Agent identifiable AVEC un contact (URL/email).
                "User-Agent": "LumenApp/1.0 (https://lechat45.github.io; projet educatif)",
                "Accept": "application/json",
            },
        ) as http:
            # 1) Trouver la meilleure page correspondant au sujet.
            search = http.get(api, params={
                "action": "query", "format": "json",
                "list": "search", "srsearch": sujet, "srlimit": 1,
            })
            hits = search.json().get("query", {}).get("search", [])
            if not hits:
                return None
            titre = hits[0]["title"]
            # 2) Récupérer l'extrait (intro complète) en texte brut + l'URL canonique.
            page = http.get(api, params={
                "action": "query", "format": "json", "redirects": 1,
                "prop": "extracts|info", "explaintext": 1, "exintro": 1,
                "inprop": "url", "titles": titre,
            })
            pages = page.json().get("query", {}).get("pages", {})
            for p in pages.values():
                extrait = (p.get("extract") or "").strip()
                if extrait:
                    return {
                        "titre": p.get("title", titre),
                        "url": p.get("fullurl", f"https://{lang}.wikipedia.org/wiki/{titre.replace(' ', '_')}"),
                        "extrait": extrait[:WIKI_CHARS],  # borne la taille envoyée à Gemini
                    }
    except Exception as exc:  # noqa: BLE001
        log.warning("Wikipédia indisponible (%s)", exc)
    return None


def _recover_double_encoded(fiche: dict) -> dict:
    """Filet de sécurité : si le modèle a double-encodé (toute la fiche coincée dans
    le champ 'sujet'), on tente de reconstruire l'objet correct."""
    s = fiche.get("sujet")
    if not (isinstance(s, str) and ('","resume":' in s or '","categorie":' in s)):
        return fiche
    try:
        repaired = json.loads('{"sujet":"' + s + "}")
        if isinstance(repaired, dict) and "resume" in repaired:
            return repaired
    except Exception:  # noqa: BLE001
        pass
    # Au minimum, on nettoie le titre (garde le nom avant le JSON parasite).
    fiche["sujet"] = s.split('","', 1)[0].strip().strip('"')
    return fiche


def gemini_summarize(system: str, user: str, wiki: dict | None = None) -> dict:
    """Rédige la fiche via Gemini avec un schéma JSON imposé (sortie toujours valide).
    Si `wiki` est fourni, Gemini se base UNIQUEMENT sur l'extrait Wikipédia réel
    (faits vérifiés, vraie source). Sinon, il s'appuie sur ses connaissances."""
    if wiki:
        system_final = system + (
            "\n\nTu disposes ci-dessous d'un EXTRAIT WIKIPÉDIA réel sur le sujet. "
            "Rédige la fiche EN TE BASANT UNIQUEMENT sur cet extrait : n'ajoute aucun "
            "fait qui n'y figure pas, n'invente aucun chiffre. La seule source est la "
            "page Wikipédia fournie ; mets-la dans \"sources\". Si l'extrait ne suffit "
            "pas pour un champ, laisse-le vide."
        )
        user_final = (
            f"{user}\n\n--- EXTRAIT WIKIPÉDIA ({wiki['titre']}) ---\n{wiki['extrait']}\n"
            f"--- URL SOURCE : {wiki['url']} ---"
        )
    else:
        system_final = system + (
            "\n\nIMPORTANT : tu n'as PAS accès à la recherche web. Ne cite donc dans "
            "\"sources\" QUE des URL stables et certaines (site officiel, Wikipédia). "
            "N'invente JAMAIS d'URL d'article précis. Signale tout doute dans \"avertissement\"."
        )
        user_final = user

    # Gabarit JSON propre (mêmes clés que la fiche). On NE met PAS de responseSchema :
    # le décodage contraint de Gemini 3 déraille sur cette structure imbriquée.
    schema_hint = (
        "\n\nRéponds en JSON valide, avec EXACTEMENT ces clés et cette structure, "
        "sans aucun texte ni balise autour :\n"
        '{"sujet":"nom court","categorie":"entreprise|histoire|inconnu","confiance":0,'
        '"accroche":"","resume":"","chiffres_cles":[{"label":"","valeur":"","annee":""}],'
        '"chronologie":[{"date":"","evenement":""}],"impact":"","le_saviez_vous":[""],'
        '"sources":[{"titre":"","url":"","fiabilite":""}],"pour_approfondir":[""],'
        '"avertissement":""}'
    )
    sys_instr = {"parts": [{"text": system_final + schema_hint}]}
    contents = [{"role": "user", "parts": [{"text": user_final}]}]
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}
    url = GEMINI_URL.format(model=GEMINI_MODEL)

    gen = {
        "temperature": 0.4,
        "maxOutputTokens": 8192,
        "responseMimeType": "application/json",
        # Réduit le raisonnement de Gemini 3 pour laisser sortir le JSON complet.
        "thinkingConfig": {"thinkingLevel": "low"},
    }
    payload = {"systemInstruction": sys_instr, "contents": contents, "generationConfig": gen}

    # Un SEUL appel, mais on retente brièvement en cas de surcharge passagère (503).
    # On ne retente PAS sur 429 (quota) : ce serait inutile et gaspilleur.
    text = ""
    last = ""
    with httpx.Client(timeout=90) as http:
        for wait in (0, 2, 5):
            if wait:
                time.sleep(wait)
            resp = http.post(url, json=payload, headers=headers)
            if resp.status_code == 503:
                last = "Gemini surchargé (503)"
                continue
            if resp.status_code == 429:
                raise RuntimeError("Quota Gemini dépassé (429)")
            if resp.status_code != 200:
                raise RuntimeError(f"Gemini summarize {resp.status_code}: {resp.text[:200]}")
            cand = (resp.json().get("candidates") or [{}])[0]
            parts = cand.get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts).strip()
            break

    if not text:
        raise RuntimeError(last or "Gemini summarize : réponse vide.")

    fiche = _extract_json(text)
    fiche = _recover_double_encoded(fiche)  # filet de sécurité anti double-encodage

    # Avec Wikipédia, on garantit la source exacte (on n'autorise pas le modèle à dériver).
    if wiki:
        fiche["sources"] = [{
            "titre": f"Wikipédia — {wiki['titre']}",
            "url": wiki["url"],
            "fiabilite": "encyclopedique",
        }]
    return fiche


@app.get("/")
def health():
    if GEMINI_API_KEY:
        models = {"provider": "gemini", "vision": GEMINI_MODEL, "summary": GEMINI_MODEL}
    else:
        models = {"provider": "groq", "vision": VISION_MODEL, "summary": SUMMARY_MODEL}
    return {"status": "ok", "service": "lumen", "models": models}


@app.get("/api/models")
def list_gemini_models():
    """Debug : liste les modèles Gemini utilisables avec ta clé (ceux qui gèrent generateContent)."""
    if not GEMINI_API_KEY:
        return {"erreur": "GEMINI_API_KEY non définie."}
    try:
        with httpx.Client(timeout=20) as http:
            resp = http.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": GEMINI_API_KEY},
            )
        if resp.status_code != 200:
            return {"erreur": f"Gemini {resp.status_code}", "detail": resp.text[:300]}
        noms = [
            m.get("name", "").replace("models/", "")
            for m in resp.json().get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]
        return {"modele_actuel": GEMINI_MODEL, "disponibles": noms}
    except Exception as exc:  # noqa: BLE001
        log.error("Erreur list_models: %s", exc)
        return {"erreur": "Impossible de lister les modèles."}


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
        if GEMINI_API_KEY:
            # Voie recommandée : Gemini (meilleure reconnaissance visuelle).
            result = gemini_identify(b64, image.content_type, indication)
        else:
            # Repli : vision Groq.
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
                max_tokens=600,
            )
            raw = completion.choices[0].message.content or ""
            try:
                result = _extract_json(raw)
            except json.JSONDecodeError:
                result = _empty_identify()
    except Exception as exc:  # noqa: BLE001
        log.error("Erreur identify: %s", exc)
        raise HTTPException(status_code=502, detail="L'identification a échoué. Réessaie.")

    return JSONResponse(result)


@app.post("/api/summarize")
def summarize(request: Request, body: SummarizeBody):
    """Étage 2 — rédige la fiche (Gemini, JSON imposé, sources canoniques)."""
    _check_rate_limit(request)

    system = SUMMARY_SYSTEM.replace("{langue}", body.langue)
    user = (
        f"Sujet : {body.sujet}\n"
        f"Catégorie présumée : {body.categorie}\n"
        f"Rédige la fiche en {body.langue}."
    )

    try:
        if GEMINI_API_KEY:
            # On récupère le vrai texte Wikipédia du sujet (gratuit, sans clé)…
            wiki = wikipedia_context(body.sujet, body.langue)
            # …puis Gemini rédige à partir de cet extrait réel (schéma JSON imposé).
            fiche = gemini_summarize(system, user, wiki=wiki)
        else:
            # Repli si aucune clé Gemini : Groq (sortie courte pour limiter le risque 413).
            completion = client.chat.completions.create(
                model=SUMMARY_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.4,
                max_tokens=1024,
            )
            fiche = _extract_json(completion.choices[0].message.content)
    except json.JSONDecodeError:
        log.error("JSON invalide depuis le modèle de résumé")
        raise HTTPException(status_code=502, detail="La fiche n'a pas pu être structurée. Réessaie.")
    except Exception as exc:  # noqa: BLE001
        log.error("Erreur summarize: %s", exc)
        raise HTTPException(status_code=502, detail="La génération de la fiche a échoué. Réessaie.")

    # Garde-fou : si le titre est aberrant (modèle qui « collapse » tout dedans),
    # on le remplace par le sujet réellement identifié.
    sujet = fiche.get("sujet")
    if not isinstance(sujet, str) or len(sujet) > 120 or "_" in sujet and len(sujet) > 60:
        fiche["sujet"] = body.sujet

    return JSONResponse(fiche)
