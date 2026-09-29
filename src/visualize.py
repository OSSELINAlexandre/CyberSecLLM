"""
Visualisation.

  - V1 (active)  : affichage console de l'architecture du modèle.
  - V2 (préparée): export JSON des activations (actif) + vue Streamlit
                   interactive (COMMENTÉE, à activer plus tard).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


# ===========================================================================
# V1 : affichage console de l'architecture
# ===========================================================================

def compter_parametres(modele) -> int:
    """Nombre total de paramètres du modèle."""
    return sum(p.numel() for p in modele.parameters())


def resume_architecture(modele) -> Dict[str, object]:
    """Rassemble les caractéristiques principales de l'architecture dans un dictionnaire."""
    cfg = modele.cfg
    premier_bloc = modele.blocks[0]
    return {
        "nom_modele": cfg.model_name,
        "n_couches": cfg.n_layers,
        "d_model": cfg.d_model,
        "d_mlp": cfg.d_mlp,
        "n_tetes": cfg.n_heads,
        "n_tetes_cle_valeur": getattr(cfg, "n_key_value_heads", None) or cfg.n_heads,
        "d_tete": cfg.d_head,
        "d_vocab": cfg.d_vocab,
        "n_ctx": cfg.n_ctx,
        "fonction_activation": cfg.act_fn,
        "type_normalisation": cfg.normalization_type,
        "type_positionnel": cfg.positional_embedding_type,
        "mlp_a_porte": bool(getattr(cfg, "gated_mlp", False)),
        "type_bloc": type(premier_bloc).__name__,
        "type_attention": type(premier_bloc.attn).__name__,
        "type_mlp": type(premier_bloc.mlp).__name__,
        "n_parametres": compter_parametres(modele),
        "dtype": str(cfg.dtype),
        "peripherique": str(cfg.device),
    }


def afficher_architecture(modele) -> None:
    """V1 : affiche l'architecture du modèle dans la console."""
    infos = resume_architecture(modele)
    libelles = {
        "nom_modele": "Modèle",
        "n_couches": "Nombre de couches (n_layers)",
        "d_model": "Dimension du flux résiduel (d_model)",
        "d_mlp": "Neurones MLP par couche (d_mlp)",
        "n_tetes": "Têtes d'attention (n_heads)",
        "n_tetes_cle_valeur": "Têtes clé/valeur (GQA)",
        "d_tete": "Dimension par tête (d_head)",
        "d_vocab": "Taille du vocabulaire (d_vocab)",
        "n_ctx": "Contexte maximal (n_ctx)",
        "fonction_activation": "Fonction d'activation",
        "type_normalisation": "Normalisation",
        "type_positionnel": "Encodage positionnel",
        "mlp_a_porte": "MLP à porte (SwiGLU)",
        "type_bloc": "Type de bloc",
        "type_attention": "Type d'attention",
        "type_mlp": "Type de MLP",
        "n_parametres": "Nombre de paramètres",
        "dtype": "Type des poids",
        "peripherique": "Périphérique",
    }
    largeur = max(len(v) for v in libelles.values())
    print("=" * (largeur + 30))
    print(" ARCHITECTURE DU MODÈLE")
    print("=" * (largeur + 30))
    for cle, libelle in libelles.items():
        valeur = infos[cle]
        if cle == "n_parametres":
            valeur = f"{valeur:,}".replace(",", " ") + f"  (~{infos[cle] / 1e9:.2f} milliards)"
        print(f"  {libelle:<{largeur}} : {valeur}")
    print("-" * (largeur + 30))
    print("  Structure d'un bloc : entrée -> Norme -> Attention -> (+) -> Norme -> MLP -> (+) -> sortie")
    print(f"  Neurones MLP au total : {infos['n_couches'] * infos['d_mlp']:,}".replace(",", " "))
    print("=" * (largeur + 30))


def afficher_top_neurones(titre: str, neurones: List[Tuple[int, float]], couche: int) -> None:
    """Affiche joliment une liste de (indice_neurone, valeur)."""
    print(f"\n{titre}")
    for rang, (indice, valeur) in enumerate(neurones, start=1):
        print(f"  {rang:>3}. L{couche}N{indice:<6} valeur = {valeur:+.4f}")


# ===========================================================================
# V2 : export JSON (actif) pour un futur front-end
# ===========================================================================

def _vers_liste(valeur):
    """Convertit récursivement tenseurs / tuples en types sérialisables JSON."""
    if isinstance(valeur, torch.Tensor):
        return valeur.detach().float().cpu().tolist()
    if isinstance(valeur, dict):
        return {cle: _vers_liste(v) for cle, v in valeur.items()}
    if isinstance(valeur, (list, tuple)):
        return [_vers_liste(v) for v in valeur]
    return valeur


def exporter_activations_json(
    chemin: str | Path,
    concept: str,
    couche: int,
    activations: Dict[str, torch.Tensor],
    top_neurones: Optional[List[Tuple[int, float]]] = None,
    top_similaires: Optional[List[Tuple[int, float]]] = None,
    architecture: Optional[Dict[str, object]] = None,
    tokens: Optional[List[str]] = None,
) -> Path:
    """
    Exporte les activations capturées dans un fichier JSON (pour un front-end V2/V3).

    Returns:
        Le chemin du fichier écrit.
    """
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    donnees = {
        "concept": concept,
        "couche": couche,
        "tokens": tokens,
        "architecture": architecture,
        "top_neurones_actives": [{"neurone": i, "activation": v} for i, v in (top_neurones or [])],
        "top_neurones_similaires": [{"neurone": i, "cosinus": v} for i, v in (top_similaires or [])],
        "activations": _vers_liste(activations),
    }
    chemin.write_text(json.dumps(donnees, ensure_ascii=False, indent=2), encoding="utf-8")
    return chemin


# Alias anglais (nom demandé dans la spécification initiale).
export_activations_json = exporter_activations_json


# ===========================================================================
# V2 : vue Streamlit interactive (COMMENTÉE — à activer plus tard)
# ---------------------------------------------------------------------------
# Pour l'activer :
#   1. pip install streamlit
#   2. Décommenter le bloc ci-dessous et le placer dans un fichier `app.py`
#      à la racine du projet.
#   3. Lancer : streamlit run app.py
# ===========================================================================
#
# import streamlit as st
# import numpy as np
#
# from src.model import charger_modele, MODEL_NAME
# from src.concepts import CONCEPTS_TEST
# from src.activations import capturer_mlp, top_k_neurones
#
#
# @st.cache_resource
# def _modele_en_cache(nom_modele: str):
#     """Charge le modèle une seule fois (mis en cache par Streamlit)."""
#     return charger_modele(nom_modele)
#
#
# def lancer_vue_streamlit():
#     """Vue interactive : activations par neurone pour un concept et une couche."""
#     st.title("Visualiseur d'activations de LLM")
#     modele = _modele_en_cache(MODEL_NAME)
#     concept = st.selectbox("Concept", CONCEPTS_TEST)
#     texte_libre = st.text_input("…ou texte libre", "")
#     couche = st.slider("Couche", 0, modele.cfg.n_layers - 1, modele.cfg.n_layers // 2)
#     k = st.slider("Nombre de neurones (top-k)", 5, 100, 20)
#
#     texte = texte_libre or concept
#     acts = capturer_mlp(modele, texte, couche)          # [n_positions, d_mlp]
#     moyenne = acts.float().mean(dim=0).cpu().numpy()    # [d_mlp]
#
#     st.subheader("Activation moyenne de chaque neurone MLP")
#     st.line_chart(moyenne)
#
#     st.subheader(f"Top {k} neurones")
#     top = top_k_neurones(acts, k)
#     st.table({"neurone": [i for i, _ in top], "activation": [v for _, v in top]})
#
#     st.subheader("Carte de chaleur positions × top neurones")
#     indices = [i for i, _ in top]
#     st.dataframe(acts[:, indices].float().cpu().numpy())
#
#
# if __name__ == "__main__":
#     lancer_vue_streamlit()
