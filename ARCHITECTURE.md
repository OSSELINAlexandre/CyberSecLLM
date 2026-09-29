# Architecture du code

Ce document décrit brièvement le rôle de chaque fichier du projet.

| Fichier | Rôle |
|---|---|
| `main.py` | Point d'entrée : lit les options de la ligne de commande, puis enchaîne chargement du modèle → affichage de l'architecture → capture des activations → affichage des neurones les plus activés et les plus similaires au concept. |
| `src/__init__.py` | Fait de `src/` un paquet Python et documente son contenu. |
| `src/model.py` | Charge le modèle (par défaut `Qwen/Qwen3-8B-Base`, constante `MODEL_NAME`) en float16 sur GPU sous forme de `HookedTransformer`. Gère le cas des variantes « Base » absentes de la liste officielle de TransformerLens et affiche une erreur claire si CUDA manque. |
| `src/concepts.py` | Contient la liste des concepts de test (« Bible », « chrétien », « Jordan Peterson » et des concepts neutres) et calcule leur vecteur dans l'espace `d_model` (moyenne des lignes de `W_E`, ou moyenne du flux résiduel à une couche). |
| `src/activations.py` | Capture les activations d'une couche (flux résiduel, attention, MLP) avec `run_with_cache` et `names_filter`, trouve les neurones les plus activés (`top_k_neurones`) et mesure la similarité cosinus entre un concept et les directions des neurones (`W_in`, `W_out`, `W_gate`). |
| `src/visualize.py` | V1 : affiche l'architecture dans la console. V2 : exporte les activations en JSON (actif) et contient une vue Streamlit interactive prête mais commentée. |
| `requirements.txt` | Liste des dépendances Python avec les versions minimales nécessaires (TransformerLens ≥ 2.16 et < 4.0). |
| `README.md` | Guide d'installation et d'utilisation pour débutants, feuille de route. |
| `.gitignore` | Exclut caches, environnements virtuels, poids de modèles et exports. |

## Flux de données

1. **Chargement du modèle** (`model.py`) : les poids sont téléchargés depuis le Hub Hugging Face au premier lancement, puis convertis en `HookedTransformer`.
2. **Capture des activations** (`concepts.py` + `activations.py`) : le concept est tokenisé, passé dans le modèle, et les activations de la couche choisie sont enregistrées.
3. **Visualisation** (`visualize.py`) : affichage console (V1), export JSON (V2), puis interfaces interactives (V2/V3).
