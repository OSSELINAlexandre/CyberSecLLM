"""
Serveur web de l'interface graphique (FastAPI + uvicorn).

Une seule commande lance tout (API + front-end statique) :

    python -m app.server                       # Qwen/Qwen3-4B, cuda, float16
    python -m app.server --modele Qwen/Qwen3-1.7B --port 8001

puis ouvrir http://localhost:8000

Le modèle est chargé UNE SEULE FOIS au démarrage (via `src.model.charger_modele`).
Les options peuvent aussi venir de variables d'environnement :
LLMVIZ_MODELE, LLMVIZ_DEVICE, LLMVIZ_HOTE, LLMVIZ_PORT, LLMVIZ_MAX_TOKENS.

Points d'entrée :
  - GET  /api/architecture : résumé de la configuration du modèle ;
  - POST /api/activations  : {"texte": "...", "top_k": 10} -> activations par couche ;
  - GET  /                 : front-end (fichiers de app/static/).
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.analyse import ErreurMemoireGPU, ErreurTexte, analyser_texte, architecture, infos_gpu

MODELE_PAR_DEFAUT = "Qwen/Qwen3-4B"
DOSSIER_STATIQUE = Path(__file__).resolve().parent / "static"


# ---------------------------------------------------------------------------
# État global : le modèle chargé une seule fois, et un verrou pour que les
# requêtes passent l'une après l'autre sur le GPU (évite de doubler la VRAM).
# ---------------------------------------------------------------------------

class EtatServeur:
    """Contient le modèle chargé et les réglages du serveur."""

    def __init__(self) -> None:
        self.modele = None
        self.max_tokens: int = 128
        self.architecture: Optional[dict] = None
        self.verrou = threading.Lock()

    def installer(self, modele, max_tokens: int) -> None:
        """Enregistre le modèle chargé et précalcule le résumé d'architecture."""
        self.modele = modele
        self.max_tokens = max_tokens
        self.architecture = architecture(modele, max_tokens)


ETAT = EtatServeur()


class RequeteActivations(BaseModel):
    """Corps de la requête POST /api/activations."""

    texte: str = Field("", description="Phrase à analyser.")
    top_k: int = Field(10, ge=1, le=100, description="Nombre de neurones MLP renvoyés par couche.")


def _erreur(code: int, message: str) -> JSONResponse:
    """Réponse d'erreur uniforme : {"erreur": "..."}."""
    return JSONResponse(status_code=code, content={"erreur": message})


app = FastAPI(title="llm-visualizer — interface graphique", docs_url="/api/docs", redoc_url=None)


@app.exception_handler(RequestValidationError)
async def _erreur_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Traduit les erreurs de validation de la requête en français."""
    erreurs = exc.errors()
    if any(e.get("type") == "json_invalid" for e in erreurs):
        return _erreur(422, 'Corps de requête invalide : JSON attendu, ex. {"texte": "une phrase", "top_k": 10}.')
    champs = ", ".join(str(e.get("loc", ["?"])[-1]) for e in erreurs)
    return _erreur(
        422,
        f"Requête invalide (champ(s) : {champs}). Attendu : "
        '{"texte": "une phrase", "top_k": entier entre 1 et 100}.',
    )


@app.get("/api/architecture")
def obtenir_architecture() -> JSONResponse:
    """Résumé de la configuration du modèle chargé."""
    if ETAT.modele is None:
        return _erreur(503, "Le modèle n'est pas encore chargé.")
    infos = dict(ETAT.architecture)
    infos["gpu"] = infos_gpu(infos.get("peripherique", ""))  # VRAM libre à jour
    return JSONResponse(infos)


@app.post("/api/activations")
def calculer_activations(requete: RequeteActivations) -> JSONResponse:
    """
    Fait passer le texte dans le modèle et renvoie les activations par couche.
    Fonction synchrone : FastAPI l'exécute dans un fil séparé, le verrou
    garantit une seule inférence à la fois.
    """
    if ETAT.modele is None:
        return _erreur(503, "Le modèle n'est pas encore chargé.")
    with ETAT.verrou:
        try:
            resultat = analyser_texte(ETAT.modele, requete.texte, requete.top_k, ETAT.max_tokens)
        except ErreurTexte as erreur:
            return _erreur(400, str(erreur))
        except ErreurMemoireGPU as erreur:
            return _erreur(507, str(erreur))
        except Exception as erreur:  # noqa: BLE001 - on renvoie toujours un message lisible
            print(f"[serveur] Erreur inattendue : {erreur!r}", file=sys.stderr)
            return _erreur(500, f"Erreur interne pendant l'inférence : {erreur}")
    return JSONResponse(resultat)


# Le front-end statique est servi à la racine (monté APRÈS les routes /api).
app.mount("/", StaticFiles(directory=DOSSIER_STATIQUE, html=True), name="statique")


# ---------------------------------------------------------------------------
# Démarrage
# ---------------------------------------------------------------------------

def analyser_arguments() -> argparse.Namespace:
    """Options de la ligne de commande (avec repli sur les variables d'environnement)."""
    env = os.environ.get
    parseur = argparse.ArgumentParser(description="Interface graphique web de llm-visualizer.")
    parseur.add_argument(
        "--modele", "--model", dest="modele", default=env("LLMVIZ_MODELE", MODELE_PAR_DEFAUT),
        help=f"Identifiant Hugging Face du modèle (défaut : {MODELE_PAR_DEFAUT}).",
    )
    parseur.add_argument(
        "--device", default=env("LLMVIZ_DEVICE", "cuda"),
        help="Périphérique : cuda (défaut), cuda:1... ou cpu (très lent pour 4B).",
    )
    parseur.add_argument(
        "--hote", "--host", dest="hote", default=env("LLMVIZ_HOTE", "127.0.0.1"),
        help="Adresse d'écoute (défaut : 127.0.0.1 ; 0.0.0.0 pour le réseau local).",
    )
    parseur.add_argument(
        "--port", type=int, default=int(env("LLMVIZ_PORT", "8000")), help="Port HTTP (défaut : 8000).",
    )
    parseur.add_argument(
        "--max-tokens", dest="max_tokens", type=int, default=int(env("LLMVIZ_MAX_TOKENS", "128")),
        help="Nombre maximal de tokens par texte (défaut : 128).",
    )
    return parseur.parse_args()


def afficher_etat_gpu(device: str, moment: str) -> None:
    """Affiche le nom du GPU et la VRAM libre (vérification au démarrage)."""
    infos = infos_gpu(device)
    if infos is None:
        print(f"[serveur] {moment} : pas de GPU CUDA utilisé (périphérique « {device} »).")
        return
    print(
        f"[serveur] {moment} : GPU « {infos['nom']} » — VRAM libre "
        f"{infos['vram_libre_go']:.2f} Go / {infos['vram_totale_go']:.2f} Go"
    )


def lancer(hote: str, port: int) -> None:
    """Lance uvicorn (le modèle doit déjà être installé dans ETAT)."""
    import uvicorn

    print(f"[serveur] Interface disponible sur http://localhost:{port}  (Ctrl+C pour arrêter)")
    uvicorn.run(app, host=hote, port=port, log_level="info")


def main() -> int:
    """Charge le modèle une seule fois puis démarre le serveur."""
    args = analyser_arguments()

    from src.model import ErreurChargementModele, charger_modele, verifier_cuda

    try:
        verifier_cuda(args.device)
    except ErreurChargementModele as erreur:
        print(f"\n[ERREUR] {erreur}", file=sys.stderr)
        return 1
    afficher_etat_gpu(args.device, "Avant chargement")

    # float16 sur GPU, float32 sur CPU (comme dans main.py).
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    debut = time.perf_counter()
    try:
        modele = charger_modele(args.modele, device=args.device, dtype=dtype)
    except ErreurChargementModele as erreur:
        print(f"\n[ERREUR] {erreur}", file=sys.stderr)
        return 1
    except torch.cuda.OutOfMemoryError:
        print(
            "\n[ERREUR] Mémoire GPU insuffisante pour charger le modèle "
            f"« {args.modele} ». Qwen3-4B demande environ 8 à 9 Go de VRAM libre en float16 ; "
            "fermez les autres programmes GPU ou essayez --modele Qwen/Qwen3-1.7B.",
            file=sys.stderr,
        )
        return 1
    print(f"[serveur] Modèle « {args.modele} » chargé en {time.perf_counter() - debut:.1f} s.")
    afficher_etat_gpu(args.device, "Après chargement")

    ETAT.installer(modele, args.max_tokens)
    lancer(args.hote, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
