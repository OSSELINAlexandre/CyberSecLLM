"""
Capture des activations internes du modèle et comparaison concepts / neurones.

Rappel des noms de « hooks » TransformerLens utilisés (l = indice de couche) :
  - blocks.{l}.hook_resid_pre     : flux résiduel à l'ENTRÉE du bloc l
  - blocks.{l}.hook_resid_post    : flux résiduel à la SORTIE du bloc l
  - blocks.{l}.hook_attn_out      : sortie du bloc d'attention
  - blocks.{l}.attn.hook_pattern  : motifs d'attention [tête, requête, clé]
  - blocks.{l}.mlp.hook_post      : activations des neurones du MLP [d_mlp]
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

# Types de points de capture disponibles -> gabarit du nom de hook.
POINTS_DE_CAPTURE = {
    "resid_pre": "blocks.{l}.hook_resid_pre",
    "resid_post": "blocks.{l}.hook_resid_post",
    "attn_out": "blocks.{l}.hook_attn_out",
    "pattern": "blocks.{l}.attn.hook_pattern",
    "mlp_post": "blocks.{l}.mlp.hook_post",
}


def nom_hook(type_capture: str, couche: int) -> str:
    """Construit le nom complet d'un hook à partir de son type et de la couche."""
    if type_capture not in POINTS_DE_CAPTURE:
        raise ValueError(
            f"Type de capture inconnu : {type_capture!r}. "
            f"Valeurs possibles : {sorted(POINTS_DE_CAPTURE)}"
        )
    return POINTS_DE_CAPTURE[type_capture].format(l=couche)


def verifier_couche(modele, couche: int) -> None:
    """Vérifie que l'indice de couche est valide pour ce modèle."""
    n_couches = modele.cfg.n_layers
    if not 0 <= couche < n_couches:
        raise ValueError(f"Couche {couche} invalide : le modèle a {n_couches} couches (0 à {n_couches - 1}).")


def capturer_activations(
    modele,
    texte: str,
    couche: int,
    types_capture: Tuple[str, ...] = ("resid_pre", "resid_post", "attn_out", "pattern", "mlp_post"),
) -> Dict[str, torch.Tensor]:
    """
    Exécute le modèle sur `texte` et capture uniquement les activations demandées
    (grâce à `names_filter`, ce qui économise beaucoup de mémoire).

    Returns:
        Dictionnaire {type_capture: tenseur} (dimension batch retirée), par ex. :
          - resid_pre / resid_post / attn_out : [n_positions, d_model]
          - pattern                           : [n_heads, n_positions, n_positions]
          - mlp_post                          : [n_positions, d_mlp]
    """
    verifier_couche(modele, couche)
    noms = {nom_hook(t, couche): t for t in types_capture}
    tokens = modele.to_tokens(texte)
    _, cache = modele.run_with_cache(tokens, names_filter=lambda nom: nom in noms)
    return {type_: cache[nom][0].detach() for nom, type_ in noms.items()}


def capturer_residuel(modele, texte: str, couche: int, position: str = "resid_post") -> torch.Tensor:
    """Capture le flux résiduel (resid_pre ou resid_post) d'une couche."""
    return capturer_activations(modele, texte, couche, (position,))[position]


def capturer_attention(modele, texte: str, couche: int) -> Dict[str, torch.Tensor]:
    """Capture la sortie du bloc d'attention et les motifs d'attention d'une couche."""
    return capturer_activations(modele, texte, couche, ("attn_out", "pattern"))


def capturer_mlp(modele, texte: str, couche: int) -> torch.Tensor:
    """Capture les activations des neurones du MLP (blocks.{l}.mlp.hook_post)."""
    return capturer_activations(modele, texte, couche, ("mlp_post",))["mlp_post"]


def top_k_neurones(activations_mlp: torch.Tensor, k: int = 10, agregation: str = "moyenne") -> List[Tuple[int, float]]:
    """
    Renvoie les k neurones MLP les plus activés.

    Args:
        activations_mlp: tenseur [n_positions, d_mlp] (ou [d_mlp]).
        k: nombre de neurones à renvoyer.
        agregation: "moyenne" (moyenne sur les positions), "max" ou "dernier"
            (dernier token uniquement).

    Returns:
        Liste de couples (indice_neurone, activation), triée par activation décroissante.
    """
    acts = activations_mlp.float()
    if acts.dim() == 2:
        if agregation == "moyenne":
            acts = acts.mean(dim=0)
        elif agregation == "max":
            acts = acts.max(dim=0).values
        elif agregation == "dernier":
            acts = acts[-1]
        else:
            raise ValueError("agregation doit valoir 'moyenne', 'max' ou 'dernier'.")
    k = min(k, acts.shape[-1])
    valeurs, indices = torch.topk(acts, k)
    return [(int(i), float(v)) for i, v in zip(indices.tolist(), valeurs.tolist())]


def directions_neurones(modele, couche: int, type_direction: str = "W_out") -> torch.Tensor:
    """
    Directions des neurones MLP d'une couche, exprimées dans l'espace d_model.

    - "W_in"  : colonnes de W_in  (ce que le neurone « lit » dans le flux résiduel)
    - "W_out" : lignes de W_out   (ce que le neurone « écrit » dans le flux résiduel)
    - "W_gate": colonnes de W_gate (MLP à porte, comme dans Qwen3 / SwiGLU)

    Returns:
        Tenseur [d_mlp, d_model] (float32).
    """
    verifier_couche(modele, couche)
    mlp = modele.blocks[couche].mlp
    if type_direction == "W_in":
        return mlp.W_in.detach().float().T  # W_in : [d_model, d_mlp] -> [d_mlp, d_model]
    if type_direction == "W_out":
        return mlp.W_out.detach().float()  # W_out : [d_mlp, d_model]
    if type_direction == "W_gate":
        if not hasattr(mlp, "W_gate"):
            raise ValueError("Ce modèle n'a pas de MLP à porte (W_gate absent).")
        return mlp.W_gate.detach().float().T
    raise ValueError("type_direction doit valoir 'W_in', 'W_out' ou 'W_gate'.")


def similarite_cosinus_neurones(
    modele,
    vecteur_concept: torch.Tensor,
    couche: int,
    type_direction: str = "W_out",
) -> torch.Tensor:
    """
    Similarité cosinus entre le vecteur d'un concept (taille d_model) et la
    direction de chaque neurone MLP de la couche.

    Returns:
        Tenseur [d_mlp] de similarités dans [-1, 1].
    """
    directions = directions_neurones(modele, couche, type_direction)
    vecteur = vecteur_concept.to(directions.device).float().unsqueeze(0)  # [1, d_model]
    return torch.nn.functional.cosine_similarity(directions, vecteur, dim=-1)


def top_k_neurones_similaires(
    modele,
    vecteur_concept: torch.Tensor,
    couche: int,
    k: int = 10,
    type_direction: str = "W_out",
) -> List[Tuple[int, float]]:
    """Renvoie les k neurones dont la direction est la plus proche (cosinus) du concept."""
    similarites = similarite_cosinus_neurones(modele, vecteur_concept, couche, type_direction)
    return top_k_neurones(similarites, k)


# Alias anglais (nom demandé dans la spécification initiale).
top_k_neurons = top_k_neurones
