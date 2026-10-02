/* ==========================================================================
   llm-visualizer — logique du front-end (JavaScript natif, sans compilation).
   1. récupère l'architecture (GET /api/architecture) ;
   2. envoie la phrase (POST /api/activations) ;
   3. dessine : tokens, carte d'activation couches × tokens, pile des couches.
   ========================================================================== */
"use strict";

// Teintes (HSL) associées à chaque type de tenseur — cohérentes avec style.css.
const TEINTES = { norme_residuel: 155, norme_attn: 28, magnitude_mlp: 255 };
const LIBELLES_MESURES = {
  norme_residuel: "‖résiduel‖",
  norme_attn: "‖attention‖",
  magnitude_mlp: "|MLP| moyen",
};

const $ = (id) => document.getElementById(id);
let dernierResultat = null; // dernière réponse de /api/activations
let architecture = null;    // réponse de /api/architecture

/* ---------- Utilitaires ---------- */

/** Crée un élément HTML avec classe et texte optionnels. */
function el(balise, classe, texte) {
  const e = document.createElement(balise);
  if (classe) e.className = classe;
  if (texte !== undefined) e.textContent = texte;
  return e;
}

/** Couleur pastel -> soutenue selon l'intensité t ∈ [0, 1]. */
function couleur(teinte, t) {
  t = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0));
  const lum = 97 - 45 * t;
  const sat = 55 + 25 * t;
  return `hsl(${teinte} ${sat}% ${lum}%)`;
}

/** Formatage court des nombres (séparateurs français). */
function formater(n, decimales = 2) {
  if (n === null || n === undefined) return "—";
  if (typeof n !== "number") return String(n);
  if (Math.abs(n) >= 1e9) return (n / 1e9).toLocaleString("fr-FR", { maximumFractionDigits: 2 }) + " Md";
  if (Number.isInteger(n)) return n.toLocaleString("fr-FR");
  return n.toLocaleString("fr-FR", { maximumFractionDigits: decimales });
}

/** Affichage lisible d'un token (espaces et retours à la ligne visibles). */
function afficherToken(t) {
  return t.replace(/\n/g, "↵").replace(/\t/g, "⇥");
}
const estSpecial = (t) => /^<\|.*\|>$/.test(t) || t === "<s>" || t === "</s>";

/** Indices de tokens pris en compte pour les échelles de couleur. */
function indicesEchelle(n) {
  const exclure = $("exclure-premier").checked && n > 1;
  return [...Array(n).keys()].filter((i) => !(exclure && i === 0));
}

/**
 * Échelle min-max : renvoie une fonction v -> [0.08, 1] (0.08 pour garder une
 * teinte visible même pour la valeur minimale).
 */
function echelle(valeurs) {
  const v = valeurs.filter((x) => x !== null && x !== undefined && Number.isFinite(x));
  const min = Math.min(...v), max = Math.max(...v);
  const etendue = max - min;
  return (x) => (x === null || x === undefined || !(etendue > 0)) ? 0.5 : 0.08 + 0.92 * (x - min) / etendue;
}

const moyenne = (vals, idx) => idx.reduce((s, i) => s + (vals[i] ?? 0), 0) / Math.max(1, idx.length);

/* ---------- Architecture ---------- */

async function chargerArchitecture() {
  try {
    const rep = await fetch("/api/architecture");
    const donnees = await rep.json();
    if (!rep.ok) throw new Error(donnees.erreur || `Erreur ${rep.status}`);
    architecture = donnees;
    afficherArchitecture(donnees);
    const etat = $("etat-serveur");
    etat.className = "pastille pastille--ok";
    etat.textContent = `${donnees.nom_modele} prêt · ${donnees.peripherique}`;
  } catch (e) {
    const etat = $("etat-serveur");
    etat.className = "pastille pastille--erreur";
    etat.textContent = "Serveur injoignable";
    $("cartes-architecture").innerHTML = "";
    $("cartes-architecture").append(el("p", "message message--erreur",
      `Impossible de lire l'architecture : ${e.message}`));
  }
}

function afficherArchitecture(a) {
  const gpu = a.gpu ? `${a.gpu.nom}` : null;
  const cartes = [
    ["Modèle", a.nom_modele, null, ""],
    ["Couches", a.n_couches, "blocs empilés", "resid"],
    ["d_model", a.d_model, "largeur du flux résiduel", "resid"],
    ["Têtes d'attention", `${a.n_tetes} / ${a.n_tetes_cle_valeur}`, "requêtes / clé-valeur (GQA)", "attn"],
    ["d_tête", a.d_tete, "dimension par tête", "attn"],
    ["d_mlp", a.d_mlp, a.mlp_a_porte ? `MLP à porte (${a.fonction_activation})` : a.fonction_activation, "mlp"],
    ["Vocabulaire", a.d_vocab, "tokens", "emb"],
    ["Contexte max.", a.n_ctx, `limite serveur : ${a.max_tokens_serveur} tokens`, "emb"],
    ["Normalisation", a.type_normalisation, a.norme_qk ? "+ norme QK" : null, ""],
    ["Positions", a.type_positionnel, null, ""],
    ["Paramètres", a.n_parametres, null, ""],
    ["Exécution", `${a.peripherique} · ${String(a.dtype).replace("torch.", "")}`, gpu, ""],
  ];
  const conteneur = $("cartes-architecture");
  conteneur.innerHTML = "";
  cartes.forEach(([libelle, valeur, note, type], i) => {
    const c = el("div", "carte-spec" + (type ? ` carte-spec--${type}` : ""));
    c.style.animationDelay = `${i * 35}ms`;
    c.append(el("div", "carte-spec__libelle", libelle), el("div", "carte-spec__valeur", formater(valeur)));
    if (note) c.append(el("div", "carte-spec__note", note));
    conteneur.append(c);
  });
}

/* ---------- Envoi de la phrase ---------- */

async function lancerAnalyse(evenement) {
  evenement.preventDefault();
  const texte = $("texte").value;
  const topK = parseInt($("top-k").value, 10) || 10;
  const message = $("message");
  const bouton = $("bouton-lancer");

  if (!texte.trim()) {
    message.className = "message message--erreur";
    message.textContent = "Saisissez une phrase avant de lancer l'analyse.";
    return;
  }
  bouton.disabled = true;
  bouton.classList.add("en-cours");
  message.className = "message";
  message.textContent = "Inférence en cours… le texte traverse les couches du modèle.";

  try {
    const debut = performance.now();
    const rep = await fetch("/api/activations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texte, top_k: topK }),
    });
    let donnees;
    try { donnees = await rep.json(); } catch { donnees = {}; }
    if (!rep.ok) throw new Error(donnees.erreur || `Erreur du serveur (${rep.status}).`);
    dernierResultat = donnees;
    const duree = performance.now() - debut;
    message.textContent = "";
    $("durees").textContent =
      `${donnees.n_tokens} tokens · ${donnees.couches.length} couches · inférence ` +
      `${formater(donnees.durees_ms.inference, 0)} ms · aller-retour ${formater(duree, 0)} ms`;
    afficherResultats(donnees);
  } catch (e) {
    message.className = "message message--erreur";
    message.textContent = e.message === "Failed to fetch"
      ? "Le serveur ne répond pas. Est-il toujours lancé ?"
      : e.message;
  } finally {
    bouton.disabled = false;
    bouton.classList.remove("en-cours");
  }
}

/* ---------- Résultats ---------- */

function afficherResultats(r) {
  $("resultats").classList.remove("cache");
  afficherTokens(r.tokens);
  dessinerCarte(r);
  dessinerPile(r);
}

function afficherTokens(tokens) {
  const conteneur = $("tokens");
  conteneur.innerHTML = "";
  tokens.forEach((t, i) => {
    const j = el("span", "jeton" + (estSpecial(t) ? " jeton--special" : ""), afficherToken(t));
    j.title = `Token ${i} : ${JSON.stringify(t)}`;
    j.style.animationDelay = `${i * 20}ms`;
    conteneur.append(j);
  });
}

/** Carte d'activation globale : une ligne par couche, une colonne par token. */
function dessinerCarte(r) {
  const mesure = $("mesure").value;
  const teinte = TEINTES[mesure];
  const n = r.n_tokens;
  const idx = indicesEchelle(n);
  let min = Infinity, max = -Infinity;
  r.couches.forEach((c) => idx.forEach((i) => {
    const v = c[mesure][i];
    if (v !== null) { min = Math.min(min, v); max = Math.max(max, v); }
  }));
  const etendue = max - min || 1;
  const parCouche = $("echelle-carte").value === "couche";

  const grille = el("div", "carte__grille");
  grille.style.gridTemplateColumns = `48px repeat(${n}, minmax(26px, 1fr))`;
  grille.append(el("div"));
  r.tokens.forEach((t) => {
    const e = el("div", "carte__entete-col", afficherToken(t));
    e.title = t;
    grille.append(e);
  });
  r.couches.forEach((c) => {
    grille.append(el("div", "carte__entete-ligne", `C${c.couche}`));
    const echLigne = echelle(idx.map((i) => c[mesure][i]));
    c[mesure].forEach((v, i) => {
      const caseCarte = el("div", "carte__case");
      caseCarte.style.background = couleur(teinte, parCouche ? echLigne(v) : (v - min) / etendue);
      caseCarte.title = `Couche ${c.couche} · token ${i} « ${r.tokens[i]} » · ${LIBELLES_MESURES[mesure]} = ${formater(v, 3)}`;
      grille.append(caseCarte);
    });
  });

  const legende = el("div", "legende");
  const degrade = el("span", "legende__degrade");
  degrade.style.background = `linear-gradient(90deg, ${couleur(teinte, 0)}, ${couleur(teinte, 1)})`;
  legende.append(el("span", null, parCouche ? "min. de la couche" : formater(min, 2)), degrade,
    el("span", null, parCouche ? "max. de la couche" : formater(max, 2)),
    el("span", null, `· ${LIBELLES_MESURES[mesure]}`));

  const conteneur = $("carte");
  conteneur.innerHTML = "";
  conteneur.append(grille, legende);
}

/** Pile verticale des couches, du plongement jusqu'à la sortie. */
function dessinerPile(r) {
  const pile = $("pile");
  pile.innerHTML = "";
  const idx = indicesEchelle(r.n_tokens);

  // Moyennes par couche, pour normaliser les couleurs sur toute la pile.
  const moyAttn = r.couches.map((c) => moyenne(c.norme_attn, idx));
  const moyMlp = r.couches.map((c) => moyenne(c.magnitude_mlp, idx));
  const echAttn = echelle(moyAttn);
  const echMlp = echelle(moyMlp);
  const echResid = echelle(r.couches.flatMap((c) => idx.map((i) => c.norme_residuel[i])));

  const entree = el("div", "bloc-extremite", "Plongement des tokens");
  entree.append(el("small", null, `${r.n_tokens} tokens → vecteurs de taille ${architecture ? architecture.d_model : "d_model"}`));
  pile.append(entree);

  r.couches.forEach((c, l) => {
    pile.append(el("div", "fleche"));
    const bloc = el("div", "couche");
    bloc.style.animationDelay = `${l * 45}ms`; // les couches « s'allument » l'une après l'autre
    bloc.tabIndex = 0;
    bloc.setAttribute("role", "button");
    bloc.setAttribute("aria-expanded", "false");

    const ligne = el("div", "couche__ligne");
    const nom = el("div", "couche__nom", `Couche ${c.couche}`);
    nom.append(el("small", null, `N${c.top_neurones[0].indice} en tête`));

    const sousAttn = el("div", "sous-bloc");
    const tAttn = echAttn(moyAttn[l]);
    sousAttn.style.background = couleur(TEINTES.norme_attn, tAttn);
    sousAttn.style.color = tAttn > .62 ? "#fff" : "#7a3f0b";
    sousAttn.append(el("span", null, "Attention"), el("span", null, formater(moyAttn[l], 1)));
    sousAttn.title = "Norme moyenne de la sortie d'attention (hook_attn_out)";

    const sousMlp = el("div", "sous-bloc");
    const tMlp = echMlp(moyMlp[l]);
    sousMlp.style.background = couleur(TEINTES.magnitude_mlp, tMlp);
    sousMlp.style.color = tMlp > .62 ? "#fff" : "#3b2a86";
    sousMlp.append(el("span", null, "MLP"), el("span", null, formater(moyMlp[l], 3)));
    sousMlp.title = "Magnitude moyenne des neurones MLP (mlp.hook_post)";

    // Vecteur « tenseur » : une case par token, couleur = norme du flux résiduel.
    const tenseur = el("div", "tenseur");
    tenseur.title = "Flux résiduel en sortie de couche (une case par token)";
    c.norme_residuel.forEach((v, i) => {
      const caseT = el("span", "tenseur__case");
      caseT.style.background = couleur(TEINTES.norme_residuel, echResid(v));
      caseT.title = `« ${r.tokens[i]} » · ‖résiduel‖ = ${formater(v, 2)}`;
      tenseur.append(caseT);
    });

    ligne.append(nom, sousAttn, sousMlp, tenseur);
    bloc.append(ligne);
    const basculer = () => basculerDetails(bloc, c, r);
    bloc.addEventListener("click", (e) => { if (!e.target.closest(".details")) basculer(); });
    bloc.addEventListener("keydown", (e) => {
      if ((e.key === "Enter" || e.key === " ") && e.target === bloc) { e.preventDefault(); basculer(); }
    });
    pile.append(bloc);
  });

  pile.append(el("div", "fleche"));
  const sortie = el("div", "bloc-extremite", "Normalisation finale → logits");
  sortie.append(el("small", null, "prédiction du token suivant"));
  pile.append(sortie);
}

/* ---------- Détails d'une couche ---------- */

function basculerDetails(bloc, c, r) {
  const existant = bloc.querySelector(".details");
  if (existant) {
    existant.remove();
    bloc.classList.remove("ouverte");
    bloc.setAttribute("aria-expanded", "false");
    return;
  }
  bloc.classList.add("ouverte");
  bloc.setAttribute("aria-expanded", "true");
  const details = el("div", "details");

  // 1. Top-k neurones : barres proportionnelles à la valeur.
  const colNeurones = el("div");
  colNeurones.append(el("h3", null, `Top ${c.top_neurones.length} neurones MLP (moyenne sur les tokens)`));
  const maxAbs = Math.max(...c.top_neurones.map((n) => Math.abs(n.valeur))) || 1;
  const barres = [];
  c.top_neurones.forEach((n) => {
    const barre = el("div", "barre" + (n.valeur < 0 ? " barre--negative" : ""));
    const fond = el("div", "barre__fond");
    const rempl = el("div", "barre__remplissage");
    fond.append(rempl);
    barre.append(el("span", "barre__nom", `N${n.indice}`), fond, el("span", "barre__valeur", formater(n.valeur, 3)));
    barre.title = `L${c.couche}N${n.indice} : ${n.valeur}`;
    colNeurones.append(barre);
    barres.push([rempl, Math.abs(n.valeur) / maxAbs]);
  });

  // 2. Têtes d'attention : attention du dernier token vers chaque token.
  const colTetes = el("div", "heatmap-tetes");
  colTetes.append(el("h3", null, `Têtes d'attention (${c.attention_dernier_token.length})`));
  const info = el("div", "info-survol");
  colTetes.append(
    heatmap(c.attention_dernier_token, TEINTES.norme_attn, (h, k, v) => {
      const t = c.tetes[h];
      info.textContent = `Tête ${h} → « ${r.tokens[k]} » : ${formater(v, 3)} · entropie ${formater(t.entropie, 2)}`;
    }),
    el("p", "discret", "Lignes : têtes · colonnes : tokens regardés par le DERNIER token."),
  );
  if (c.motif_moyen) {
    colTetes.append(
      heatmap(c.motif_moyen, TEINTES.magnitude_mlp, (q, k, v) => {
        info.textContent = `« ${r.tokens[q]} » regarde « ${r.tokens[k]} » : ${formater(v, 3)} (moyenne des têtes)`;
      }, true),
      el("p", "discret", "Motif moyen des têtes : lignes = token qui regarde, colonnes = token regardé."),
    );
  }
  colTetes.append(info);

  details.append(colNeurones, colTetes);
  bloc.append(details);
  requestAnimationFrame(() => requestAnimationFrame(() => {
    barres.forEach(([rempl, t]) => { rempl.style.width = `${Math.max(2, t * 100)}%`; });
  }));
}

/** Petite carte de chaleur sur <canvas> (valeurs dans [0, 1]) ; `carre` = cases carrées. */
function heatmap(matrice, teinte, survol, carre = false) {
  const lignes = matrice.length;
  const colonnes = matrice[0] ? matrice[0].length : 0;
  const canvas = document.createElement("canvas");
  canvas.width = colonnes;
  canvas.height = lignes;
  // Cases de taille fixe (entre 5 et 16 px), largeur totale limitée à ~420 px.
  const tailleCase = Math.max(5, Math.min(16, Math.floor(420 / Math.max(colonnes, 1))));
  canvas.style.width = `${colonnes * tailleCase}px`;
  canvas.style.height = `${lignes * (carre ? tailleCase : Math.min(tailleCase, 9))}px`;
  const ctx = canvas.getContext("2d");
  let max = 0;
  matrice.forEach((l) => l.forEach((v) => { max = Math.max(max, v ?? 0); }));
  matrice.forEach((l, i) => l.forEach((v, j) => {
    ctx.fillStyle = couleur(teinte, (v ?? 0) / (max || 1));
    ctx.fillRect(j, i, 1, 1);
  }));
  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const j = Math.floor(((e.clientX - rect.left) / rect.width) * colonnes);
    const i = Math.floor(((e.clientY - rect.top) / rect.height) * lignes);
    if (i >= 0 && i < lignes && j >= 0 && j < colonnes) survol(i, j, matrice[i][j]);
  });
  return canvas;
}

/* ---------- Initialisation ---------- */

document.addEventListener("DOMContentLoaded", () => {
  $("formulaire").addEventListener("submit", lancerAnalyse);
  $("texte").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) $("formulaire").requestSubmit();
  });
  const redessiner = () => { if (dernierResultat) { dessinerCarte(dernierResultat); dessinerPile(dernierResultat); } };
  $("mesure").addEventListener("change", () => dernierResultat && dessinerCarte(dernierResultat));
  $("echelle-carte").addEventListener("change", () => dernierResultat && dessinerCarte(dernierResultat));
  $("exclure-premier").addEventListener("change", redessiner);
  chargerArchitecture();
});
