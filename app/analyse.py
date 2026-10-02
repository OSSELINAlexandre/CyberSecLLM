"""
Analyse « toutes couches » d'un texte, partagée par le serveur web et par
`scripts/verifier_gpu.py` (même chemin de code).

On réutilise les briques existantes du projet :
  - `src.activations.nom_hook`       : noms des hooks TransformerLens ;
  - `src.activations.top_k_neurones` : top-k des neurones MLP ;
  - `src.visualize.resume_architecture` : résumé de la configuration du modèle.

Précautions mémoire (GPU) :
  - un SEUL passage avant (`run_with_cache`) avec `names_filter`, qui ne garde
    que les hooks nécessaires ;
  - `torch.inference_mode()` (pas de graphe de calcul) ;
  - chaque tenseur est réduit sur le GPU, puis passé immédiatement sur CPU en float32 ;
  - le cache est supprimé et `torch.cuda.empty_cache()` est appelé après chaque requête.
"""

from __future__ import annotations

import gc
import math
import time
from typing import Dict, List, Optional

import torch

from src.activations import nom_hook, top_k_neurones
from src.visualize import resume_architecture

# Types de hooks capturés pour chaque couche (voir src/activations.py).
TYPES_CAPTURES = ("resid_post", "attn_out", "mlp_post", "pattern")

# Au-delà de ce nombre de tokens, on ne renvoie plus le motif d'attention
# complet (moyenné sur les têtes) afin de garder une réponse JSON légère.
SEUIL_MOTIF_COMPLET = 32

# Nombre de décimales conservées dans le JSON.
DECIMALES = 4


class ErreurTexte(ValueError):
    """Texte refusé (vide, trop long...) : message destiné à l'utilisateur."""


class ErreurMemoireGPU(RuntimeError):
    """Mémoire GPU insuffisante pendant l'inférence."""


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _arrondir(valeurs, decimales: int = DECIMALES):
    """Arrondit un tenseur (ou une liste imbriquée) pour alléger le JSON."""
    if isinstance(valeurs, torch.Tensor):
        valeurs = valeurs.tolist()
    if isinstance(valeurs, list):
        return [_arrondir(v, decimales) for v in valeurs]
    if isinstance(valeurs, float):
        if math.isnan(valeurs) or math.isinf(valeurs):
            return None  # JSON ne connaît pas NaN / Infini
        return round(valeurs, decimales)
    return valeurs


def liberer_memoire() -> None:
    """Libère la mémoire inutilisée (Python, puis cache CUDA si présent)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def infos_gpu(device: str) -> Optional[Dict[str, object]]:
    """Nom du GPU et VRAM libre / totale (en Go), ou None si l'on n'est pas sur CUDA."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    index = torch.device(device).index
    index = torch.cuda.current_device() if index is None else index
    libre, total = torch.cuda.mem_get_info(index)
    return {
        "nom": torch.cuda.get_device_name(index),
        "vram_libre_go": round(libre / 1024**3, 2),
        "vram_totale_go": round(total / 1024**3, 2),
    }


def architecture(modele, max_tokens: int) -> Dict[str, object]:
    """
    Résumé de l'architecture (réutilise `resume_architecture`) complété par
    quelques champs utiles au front-end.
    """
    infos = dict(resume_architecture(modele))
    cfg = modele.cfg
    infos.update(
        {
            "taille_rotary": getattr(cfg, "rotary_dim", None),
            "norme_qk": bool(getattr(cfg, "use_qk_norm", False)),
            "max_tokens_serveur": max_tokens,
            "gpu": infos_gpu(str(cfg.device)),
        }
    )
    return infos


# ---------------------------------------------------------------------------
# Analyse principale
# ---------------------------------------------------------------------------

def tokeniser(modele, texte: str, max_tokens: int) -> torch.Tensor:
    """Tokenise le texte (avec BOS) après vérification de sa validité."""
    if texte is None or not texte.strip():
        raise ErreurTexte("Le texte est vide : saisissez une phrase à analyser.")
    tokens = modele.to_tokens(texte)  # [1, n_positions]
    n_tokens = tokens.shape[-1]
    if n_tokens > max_tokens:
        raise ErreurTexte(
            f"Texte trop long : {n_tokens} tokens, alors que la limite est de {max_tokens}. "
            "Raccourcissez la phrase (ou relancez le serveur avec --max-tokens)."
        )
    return tokens


def _resume_tetes(motif: torch.Tensor) -> List[Dict[str, float]]:
    """
    Résumé de chaque tête d'attention à partir de son motif [têtes, requête, clé] :
      - entropie moyenne (attention diffuse = élevée, focalisée = faible) ;
      - part moyenne d'attention envoyée au premier token (« puits d'attention ») ;
      - part moyenne d'attention portée sur le token lui-même (diagonale).
    """
    p = motif.clamp_min(1e-12)
    entropie = -(motif * p.log()).sum(-1).mean(-1)  # [têtes]
    vers_premier = motif[:, :, 0].mean(-1)  # [têtes]
    diagonale = torch.diagonal(motif, dim1=-2, dim2=-1).mean(-1)  # [têtes]
    return [
        {"entropie": e, "vers_premier": b, "diagonale": d}
        for e, b, d in zip(
            _arrondir(entropie), _arrondir(vers_premier), _arrondir(diagonale)
        )
    ]


def analyser_texte(modele, texte: str, top_k: int = 10, max_tokens: int = 128) -> Dict[str, object]:
    """
    Fait passer `texte` dans le modèle et renvoie, pour chaque couche, des
    résumés compacts des activations (prêts à être sérialisés en JSON).

    Returns:
        {
          "tokens": [...],
          "n_tokens": int,
          "couches": [
            {
              "couche": l,
              "norme_residuel": [n_tokens],   # ‖resid_post‖ par token
              "norme_attn": [n_tokens],       # ‖attn_out‖ par token
              "magnitude_mlp": [n_tokens],    # moyenne de |mlp_post| par token
              "top_neurones": [{"indice", "valeur"}, ...],
              "tetes": [{"entropie", "vers_premier", "diagonale"}, ...],
              "attention_dernier_token": [n_têtes][n_tokens],
              "motif_moyen": [n_tokens][n_tokens] ou None (si texte long),
            }, ...
          ],
          "durees_ms": {...},
        }
    """
    top_k = max(1, min(int(top_k), modele.cfg.d_mlp))
    t0 = time.perf_counter()
    tokens = tokeniser(modele, texte, max_tokens)
    n_tokens = tokens.shape[-1]
    noms = {
        nom_hook(type_, couche)
        for couche in range(modele.cfg.n_layers)
        for type_ in TYPES_CAPTURES
    }

    cache = None
    try:
        t1 = time.perf_counter()
        with torch.inference_mode():
            _, cache = modele.run_with_cache(tokens, names_filter=lambda nom: nom in noms)
            if torch.cuda.is_available() and str(tokens.device).startswith("cuda"):
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            couches = []
            for couche in range(modele.cfg.n_layers):
                # Réductions effectuées sur le périphérique du modèle, puis passage
                # immédiat sur CPU en float32 (les gros tenseurs ne quittent pas le GPU).
                resid = cache[nom_hook("resid_post", couche)][0].float()  # [pos, d_model]
                attn = cache[nom_hook("attn_out", couche)][0].float()  # [pos, d_model]
                mlp = cache[nom_hook("mlp_post", couche)][0]  # [pos, d_mlp]
                motif = cache[nom_hook("pattern", couche)][0].float()  # [têtes, req, clé]

                norme_residuel = resid.norm(dim=-1).cpu()
                norme_attn = attn.norm(dim=-1).cpu()
                magnitude_mlp = mlp.float().abs().mean(dim=-1).cpu()
                top = top_k_neurones(mlp, top_k)  # réutilise src/activations.py
                motif_cpu = motif.cpu()

                couches.append(
                    {
                        "couche": couche,
                        "norme_residuel": _arrondir(norme_residuel),
                        "norme_attn": _arrondir(norme_attn),
                        "magnitude_mlp": _arrondir(magnitude_mlp),
                        "top_neurones": [
                            {"indice": i, "valeur": _arrondir(v)} for i, v in top
                        ],
                        "tetes": _resume_tetes(motif_cpu),
                        "attention_dernier_token": _arrondir(motif_cpu[:, -1, :], 3),
                        "motif_moyen": (
                            _arrondir(motif_cpu.mean(0), 3)
                            if n_tokens <= SEUIL_MOTIF_COMPLET
                            else None
                        ),
                    }
                )
                del resid, attn, mlp, motif, motif_cpu
        t3 = time.perf_counter()
    except torch.cuda.OutOfMemoryError as erreur:
        raise ErreurMemoireGPU(
            "Mémoire GPU insuffisante (CUDA out of memory) pendant l'inférence. "
            "Essayez un texte plus court, fermez les autres programmes qui utilisent le GPU, "
            "ou utilisez un modèle plus petit (ex. --modele Qwen/Qwen3-1.7B)."
        ) from erreur
    finally:
        # Le cache peut peser plusieurs centaines de Mo : on le libère tout de suite.
        del cache
        liberer_memoire()

    return {
        "texte": texte,
        "tokens": modele.to_str_tokens(tokens[0]),
        "n_tokens": n_tokens,
        "top_k": top_k,
        "seuil_motif_complet": SEUIL_MOTIF_COMPLET,
        "couches": couches,
        "durees_ms": {
            "tokenisation": round((t1 - t0) * 1000, 1),
            "inference": round((t2 - t1) * 1000, 1),
            "reduction": round((t3 - t2) * 1000, 1),
            "total": round((t3 - t0) * 1000, 1),
        },
    }
