"""
Chargement du modèle de langage via TransformerLens (HookedTransformer).

Aucun poids n'est inclus dans ce projet : ils sont téléchargés depuis le Hub
Hugging Face (ou lus depuis le cache local) au premier appel de `charger_modele`.

Note sur la compatibilité TransformerLens / Qwen3 :
  - Qwen3 est pris en charge par HookedTransformer à partir de TransformerLens 2.16.0.
  - La liste officielle de HookedTransformer ne contient que les variantes
    « instruct » (`Qwen/Qwen3-0.6B`, `-1.7B`, `-4B`, `-8B`, `-14B`), PAS les
    variantes `-Base`. Pour une variante `-Base`, on charge donc les poids HF
    nous-mêmes puis on les injecte (`hf_model=`) en utilisant le nom officiel
    de même architecture (ex. `Qwen/Qwen3-8B`) comme « gabarit » de configuration.
  - TransformerLens 4.0 a SUPPRIMÉ HookedTransformer (remplacé par
    TransformerBridge) : d'où la contrainte `<4.0` dans requirements.txt.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Constante de configuration : identifiant Hugging Face du modèle cible.
# (L'utilisateur a mentionné « Qwen 3.8B », qui n'existe pas : on vise Qwen3-8B Base.)
# Alternatives possibles (moins gourmandes en VRAM) :
#   - "Qwen/Qwen3-4B-Base"   (~8 Go de VRAM en float16)
#   - "Qwen/Qwen3-1.7B-Base" (~4 Go)
#   - "Qwen/Qwen3-0.6B-Base" (~1,5 Go, idéal pour tester le code)
#   - "Qwen/Qwen3-8B"        (version instruct, présente dans la liste officielle)
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-8B-Base"

# Correspondance « variante Base » -> « nom officiel TransformerLens de même architecture ».
# Les variantes Base et instruct partagent exactement la même configuration
# (nombre de couches, dimensions, etc.) ; seuls les poids diffèrent.
GABARITS_TRANSFORMERLENS = {
    "Qwen/Qwen3-0.6B-Base": "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B-Base": "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B-Base": "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B-Base": "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B-Base": "Qwen/Qwen3-14B",
}


class ErreurChargementModele(RuntimeError):
    """Erreur levée lorsque le modèle ne peut pas être chargé."""


def verifier_cuda(device: str) -> None:
    """Vérifie que CUDA est disponible si l'on demande un périphérique GPU."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ErreurChargementModele(
            "CUDA n'est pas disponible : impossible de charger le modèle sur GPU.\n"
            "  - Vérifiez que vous avez installé une version de PyTorch compilée avec CUDA\n"
            "    (voir https://pytorch.org/get-started/locally/).\n"
            "  - Vérifiez les pilotes NVIDIA avec la commande `nvidia-smi`.\n"
            "  - Pour tester sans GPU, utilisez un petit modèle (ex. Qwen/Qwen3-0.6B-Base)\n"
            "    avec l'option --device cpu (lent, et float32 recommandé sur CPU)."
        )


def _importer_hooked_transformer():
    """Importe HookedTransformer avec un message clair en cas d'échec."""
    try:
        from transformer_lens import HookedTransformer  # noqa: WPS433
    except ImportError as erreur:
        raise ErreurChargementModele(
            "Impossible d'importer HookedTransformer. Installez une version compatible :\n"
            "    pip install 'transformer_lens>=2.16.0,<4.0'\n"
            "(TransformerLens 4.0 a supprimé HookedTransformer au profit de TransformerBridge.)"
        ) from erreur
    return HookedTransformer


def _est_nom_officiel(nom: str) -> bool:
    """Indique si `nom` figure dans la liste officielle (ou les alias) de TransformerLens."""
    try:
        from transformer_lens.loading_from_pretrained import get_official_model_name

        get_official_model_name(nom)
        return True
    except Exception:  # noqa: BLE001 - toute erreur signifie « non officiel »
        return False


def _charger_poids_hf(nom_hf: str, dtype: torch.dtype):
    """Charge le modèle et le tokenizer Hugging Face (sur CPU, pour conversion)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(nom_hf)
    modele_hf = AutoModelForCausalLM.from_pretrained(
        nom_hf,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    return modele_hf, tokenizer


def charger_modele(
    model_name: str = MODEL_NAME,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """
    Charge un modèle sous forme de `HookedTransformer` (TransformerLens).

    Stratégie :
      1. Si `model_name` est un nom officiel TransformerLens : chargement direct.
      2. Sinon, si c'est une variante connue (ex. Qwen3-*-Base) : on charge les
         poids HF puis on les injecte dans le gabarit officiel de même architecture.
      3. En dernier recours : `from_pretrained_no_processing` (sans repliement
         des normes / centrage des poids), plus tolérant.

    Args:
        model_name: identifiant Hugging Face (ex. "Qwen/Qwen3-8B-Base").
        device: "cuda" (par défaut), "cuda:0", "cpu"...
        dtype: type des poids (float16 par défaut, ~16 Go de VRAM pour 8B).

    Returns:
        Une instance de `HookedTransformer` en mode évaluation.
    """
    verifier_cuda(device)
    HookedTransformer = _importer_hooked_transformer()

    # Options communes : on désactive le calcul du gradient (inférence uniquement).
    torch.set_grad_enabled(False)
    options = {"device": device, "dtype": dtype}

    erreurs: list[str] = []

    # --- 1. Nom officiel : chargement direct ---------------------------------
    if _est_nom_officiel(model_name):
        try:
            print(f"[modèle] Chargement direct de « {model_name} » via HookedTransformer...")
            modele = HookedTransformer.from_pretrained(model_name, **options)
            return modele.eval()
        except Exception as erreur:  # noqa: BLE001
            erreurs.append(f"from_pretrained direct : {erreur}")

    # --- 2. Variante Base : injection des poids HF dans le gabarit officiel ---
    gabarit = GABARITS_TRANSFORMERLENS.get(model_name)
    if gabarit is not None:
        try:
            print(
                f"[modèle] « {model_name} » absent de la liste officielle TransformerLens : "
                f"chargement des poids HF puis injection dans le gabarit « {gabarit} »..."
            )
            modele_hf, tokenizer = _charger_poids_hf(model_name, dtype)
            try:
                modele = HookedTransformer.from_pretrained(
                    gabarit, hf_model=modele_hf, tokenizer=tokenizer, **options
                )
            except Exception as erreur:  # noqa: BLE001
                erreurs.append(f"from_pretrained (gabarit) : {erreur}")
                print("[modèle] Échec, nouvel essai avec from_pretrained_no_processing...")
                modele = HookedTransformer.from_pretrained_no_processing(
                    gabarit, hf_model=modele_hf, tokenizer=tokenizer, **options
                )
            del modele_hf  # libère la mémoire CPU occupée par la copie HF
            return modele.eval()
        except Exception as erreur:  # noqa: BLE001
            erreurs.append(f"injection HF : {erreur}")

    # --- 3. Dernier recours : chargement sans post-traitement ----------------
    try:
        print(f"[modèle] Dernier recours : from_pretrained_no_processing(« {model_name} »)...")
        modele = HookedTransformer.from_pretrained_no_processing(model_name, **options)
        return modele.eval()
    except Exception as erreur:  # noqa: BLE001
        erreurs.append(f"from_pretrained_no_processing : {erreur}")

    details = "\n  - ".join(erreurs)
    raise ErreurChargementModele(
        f"Impossible de charger « {model_name} ». Détails des tentatives :\n  - {details}\n"
        "Pistes : vérifier la version de transformer_lens (>=2.16,<4.0), la connexion au "
        "Hub (`huggingface-cli login`), la VRAM disponible, ou essayer un modèle plus petit."
    )


# Alias anglais (nom demandé dans la spécification initiale).
load_model = charger_modele
