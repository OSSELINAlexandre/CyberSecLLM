"""
Vérification sur GPU du chemin de code de l'interface graphique.

Charge Qwen/Qwen3-4B (par défaut) sur CUDA en float16, fait passer UNE phrase
par exactement la même fonction que l'API (`app.analyse.analyser_texte`), puis
affiche un résumé par couche, les durées et le pic de VRAM.

Code de sortie : 0 en cas de succès, 1 en cas d'échec.

Exemples (depuis la racine du projet) :
    python scripts/verifier_gpu.py
    python scripts/verifier_gpu.py --modele Qwen/Qwen3-1.7B --texte "Bonjour à tous"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Permet de lancer le script depuis n'importe où : on ajoute la racine du projet.
RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

import torch  # noqa: E402

from app.analyse import ErreurMemoireGPU, analyser_texte, architecture, infos_gpu  # noqa: E402
from src.model import ErreurChargementModele, charger_modele, verifier_cuda  # noqa: E402


def analyser_arguments() -> argparse.Namespace:
    """Options de la ligne de commande."""
    parseur = argparse.ArgumentParser(description="Vérifie l'inférence de l'interface graphique sur GPU.")
    parseur.add_argument("--modele", default="Qwen/Qwen3-4B", help="Modèle à charger (défaut : Qwen/Qwen3-4B).")
    parseur.add_argument("--device", default="cuda", help="Périphérique CUDA (défaut : cuda).")
    parseur.add_argument(
        "--texte", default="Au commencement était le Verbe, et le Verbe était avec Dieu.",
        help="Phrase de test.",
    )
    parseur.add_argument("--top-k", dest="top_k", type=int, default=5)
    parseur.add_argument("--max-tokens", dest="max_tokens", type=int, default=128)
    return parseur.parse_args()


def main() -> int:
    """Charge le modèle, lance une analyse et affiche le bilan."""
    args = analyser_arguments()
    print("=" * 72)
    print(" VÉRIFICATION GPU — llm-visualizer (interface graphique)")
    print("=" * 72)
    print(f"PyTorch {torch.__version__} — CUDA disponible : {torch.cuda.is_available()}")

    try:
        verifier_cuda(args.device)
    except ErreurChargementModele as erreur:
        print(f"[ÉCHEC] {erreur}", file=sys.stderr)
        return 1
    gpu = infos_gpu(args.device)
    if gpu:
        print(f"GPU : {gpu['nom']} — VRAM libre {gpu['vram_libre_go']} Go / {gpu['vram_totale_go']} Go")
        torch.cuda.reset_peak_memory_stats()

    # 1. Chargement (même fonction que le serveur et que main.py)
    debut = time.perf_counter()
    try:
        modele = charger_modele(args.modele, device=args.device, dtype=torch.float16)
    except ErreurChargementModele as erreur:
        print(f"[ÉCHEC] Chargement impossible : {erreur}", file=sys.stderr)
        return 1
    duree_chargement = time.perf_counter() - debut
    vram_modele = torch.cuda.memory_allocated() / 1024**3 if gpu else 0.0
    print(f"Modèle chargé en {duree_chargement:.1f} s — VRAM occupée par les poids : {vram_modele:.2f} Go")

    infos = architecture(modele, args.max_tokens)
    print(
        f"Architecture : {infos['n_couches']} couches, d_model={infos['d_model']}, "
        f"d_mlp={infos['d_mlp']}, têtes={infos['n_tetes']} (KV {infos['n_tetes_cle_valeur']}), "
        f"normalisation={infos['type_normalisation']}"
    )

    # 2. Analyse (chemin de code identique à POST /api/activations)
    try:
        resultat = analyser_texte(modele, args.texte, args.top_k, args.max_tokens)
        # Deuxième passage : mesure « à chaud » (sans le coût du premier appel CUDA).
        resultat = analyser_texte(modele, args.texte, args.top_k, args.max_tokens)
    except ErreurMemoireGPU as erreur:
        print(f"[ÉCHEC] {erreur}", file=sys.stderr)
        return 1
    except Exception as erreur:  # noqa: BLE001
        print(f"[ÉCHEC] Erreur pendant l'analyse : {erreur!r}", file=sys.stderr)
        return 1

    print(f"\nTexte : « {args.texte} »")
    print(f"Tokens ({resultat['n_tokens']}) : {resultat['tokens']}")
    print("\n Couche | ‖résiduel‖ moy. | ‖attn‖ moy. | |MLP| moy. | neurone n°1 (valeur)")
    print(" -------+-----------------+-------------+------------+---------------------")
    for c in resultat["couches"]:
        moy = lambda valeurs: sum(v or 0.0 for v in valeurs) / len(valeurs)  # noqa: E731
        n1 = c["top_neurones"][0]
        print(
            f"  {c['couche']:>5} | {moy(c['norme_residuel']):>15.2f} | {moy(c['norme_attn']):>11.2f} "
            f"| {moy(c['magnitude_mlp']):>10.4f} | N{n1['indice']} ({n1['valeur']:+.3f})"
        )

    taille_json = len(json.dumps(resultat, ensure_ascii=False)) / 1024
    print("\nDurées (ms, 2e passage) :", resultat["durees_ms"])
    print(f"Taille de la réponse JSON : {taille_json:.0f} Ko")
    if gpu:
        pic = torch.cuda.max_memory_allocated() / 1024**3
        apres = infos_gpu(args.device)
        print(f"Pic de VRAM allouée par PyTorch : {pic:.2f} Go")
        print(f"VRAM libre après nettoyage : {apres['vram_libre_go']} Go")

    print("\n[SUCCÈS] Le chemin de code de l'interface fonctionne sur ce GPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
