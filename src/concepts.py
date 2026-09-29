"""
Concepts de test et calcul de leur « plongement » (embedding) dans l'espace d_model.
"""

from __future__ import annotations

import torch

# Concepts « cibles » que l'on souhaite étudier.
CONCEPTS_CIBLES = ["Bible", "chrétien", "Jordan Peterson"]

# Concepts « neutres » servant de points de comparaison (témoins).
CONCEPTS_NEUTRES = ["table", "voiture", "ordinateur", "montagne", "météo"]

# Liste complète des concepts de test (modifiez-la pour ajouter vos propres concepts).
CONCEPTS_TEST = CONCEPTS_CIBLES + CONCEPTS_NEUTRES


def tokeniser_concept(modele, concept: str, ajouter_espace: bool = True) -> torch.Tensor:
    """
    Transforme un mot/concept en identifiants de tokens (sans token BOS).

    Un espace initial est ajouté par défaut : dans un texte, un mot est
    généralement précédé d'un espace, ce qui change sa tokenisation.

    Returns:
        Tenseur 1D d'identifiants de tokens.
    """
    texte = (" " + concept) if ajouter_espace else concept
    tokens = modele.to_tokens(texte, prepend_bos=False)  # forme [1, n_tokens]
    return tokens[0]


def plongement_concept(modele, concept: str, ajouter_espace: bool = True) -> torch.Tensor:
    """
    Plongement « statique » d'un concept : moyenne des lignes de W_E
    (matrice d'embedding) correspondant à ses tokens.

    Returns:
        Vecteur de taille [d_model] (en float32 pour la précision des calculs).
    """
    tokens = tokeniser_concept(modele, concept, ajouter_espace)
    lignes = modele.W_E[tokens]  # [n_tokens, d_model]
    return lignes.float().mean(dim=0)


def plongement_residuel_concept(
    modele,
    concept: str,
    couche: int,
    position: str = "resid_post",
    ajouter_espace: bool = True,
) -> torch.Tensor:
    """
    Plongement « contextuel » d'un concept : moyenne du flux résiduel
    (resid_pre ou resid_post) à la couche `couche`, sur les tokens du concept.

    Le token BOS éventuel est exclu de la moyenne.

    Returns:
        Vecteur de taille [d_model] (float32).
    """
    if position not in ("resid_pre", "resid_post"):
        raise ValueError("position doit valoir 'resid_pre' ou 'resid_post'.")
    nom_hook = f"blocks.{couche}.hook_{position}"
    texte = (" " + concept) if ajouter_espace else concept
    tokens = modele.to_tokens(texte)  # avec BOS si le modèle en utilise un
    _, cache = modele.run_with_cache(tokens, names_filter=lambda nom: nom == nom_hook)
    residuel = cache[nom_hook][0]  # [n_positions, d_model]
    n_tokens_concept = tokeniser_concept(modele, concept, ajouter_espace).shape[0]
    return residuel[-n_tokens_concept:].float().mean(dim=0)
