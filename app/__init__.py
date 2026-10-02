"""
Paquet `app` : interface graphique web du projet llm-visualizer.

  - `analyse.py` : calcul des activations de TOUTES les couches pour un texte
    (réutilise `src.activations` et `src.visualize`) et mise en forme JSON compacte ;
  - `server.py`  : serveur FastAPI (API + fichiers statiques du front-end) ;
  - `static/`    : front-end HTML/CSS/JS sans étape de compilation.
"""
