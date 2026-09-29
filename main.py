"""
Point d'entrée du projet llm-visualizer.

Enchaînement :
  1. chargement du modèle (TransformerLens, float16, GPU) ;
  2. affichage de l'architecture (V1) ;
  3. capture des activations pour un concept d'exemple ;
  4. affichage des neurones MLP les plus activés et des plus « similaires » au concept ;
  5. (optionnel) export JSON pour la future V2.

Exemples :
    python main.py
    python main.py --concept "Jordan Peterson" --couche 20 --top-k 15
    python main.py --modele Qwen/Qwen3-0.6B-Base --export exports/test.json
"""

from __future__ import annotations

import argparse
import sys

import torch


def analyser_arguments() -> argparse.Namespace:
    """Définit et lit les options de la ligne de commande."""
    from src.model import MODEL_NAME

    parseur = argparse.ArgumentParser(
        description="Visualiser l'architecture et les activations d'un LLM (TransformerLens)."
    )
    parseur.add_argument("--concept", default="Bible", help="Concept à analyser (défaut : « Bible »).")
    parseur.add_argument(
        "--couche", "--layer", dest="couche", type=int, default=None,
        help="Indice de la couche à analyser (défaut : couche du milieu).",
    )
    parseur.add_argument("--top-k", dest="top_k", type=int, default=10, help="Nombre de neurones à afficher.")
    parseur.add_argument(
        "--modele", "--model", dest="modele", default=MODEL_NAME,
        help=f"Identifiant Hugging Face du modèle (défaut : {MODEL_NAME}).",
    )
    parseur.add_argument("--device", default="cuda", help="Périphérique : cuda (défaut) ou cpu.")
    parseur.add_argument(
        "--direction", choices=["W_in", "W_out", "W_gate"], default="W_out",
        help="Directions de neurones utilisées pour la similarité cosinus.",
    )
    parseur.add_argument(
        "--export", default=None,
        help="Chemin d'un fichier JSON où exporter les activations (V2, optionnel).",
    )
    return parseur.parse_args()


def main() -> int:
    """Orchestre le chargement, la capture et l'affichage."""
    args = analyser_arguments()

    from src.activations import capturer_activations, top_k_neurones, top_k_neurones_similaires
    from src.concepts import plongement_concept
    from src.model import ErreurChargementModele, charger_modele
    from src.visualize import (
        afficher_architecture,
        afficher_top_neurones,
        exporter_activations_json,
        resume_architecture,
    )

    # float16 sur GPU ; float32 sur CPU (le float16 y est lent / mal pris en charge).
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32

    # 1. Chargement du modèle
    try:
        modele = charger_modele(args.modele, device=args.device, dtype=dtype)
    except ErreurChargementModele as erreur:
        print(f"\n[ERREUR] {erreur}", file=sys.stderr)
        return 1

    # 2. Architecture (V1)
    afficher_architecture(modele)

    couche = args.couche if args.couche is not None else modele.cfg.n_layers // 2
    print(f"\nConcept analysé : « {args.concept} » — couche {couche}")
    print(f"Tokens : {modele.to_str_tokens(' ' + args.concept, prepend_bos=False)}")

    # 3. Capture des activations
    activations = capturer_activations(modele, args.concept, couche)
    for nom, tenseur in activations.items():
        print(f"  - {nom:<10} forme = {tuple(tenseur.shape)}")

    # 4a. Neurones MLP les plus activés
    top_actives = top_k_neurones(activations["mlp_post"], args.top_k)
    afficher_top_neurones(f"Top {args.top_k} neurones MLP les plus activés (moyenne sur les tokens) :", top_actives, couche)

    # 4b. Neurones dont la direction est la plus proche du plongement du concept
    vecteur = plongement_concept(modele, args.concept)
    top_similaires = top_k_neurones_similaires(modele, vecteur, couche, args.top_k, args.direction)
    afficher_top_neurones(
        f"Top {args.top_k} neurones les plus similaires (cosinus, directions {args.direction}) :",
        top_similaires, couche,
    )

    # 5. Export JSON optionnel (V2)
    if args.export:
        chemin = exporter_activations_json(
            args.export, args.concept, couche, activations,
            top_neurones=top_actives, top_similaires=top_similaires,
            architecture=resume_architecture(modele),
            tokens=modele.to_str_tokens(args.concept),
        )
        print(f"\nActivations exportées dans : {chemin}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
