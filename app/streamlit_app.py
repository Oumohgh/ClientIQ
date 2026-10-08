"""Application Streamlit de segmentation des clients (projet ClientIQ).

Lancement depuis la racine du projet :
    streamlit run app/streamlit_app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Chemins construits à partir de l'emplacement du fichier : l'application
# fonctionne donc quel que soit le dossier courant d'exécution.
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR.parent / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Colonnes attendues par les pipelines, dans l'ordre d'entraînement.
FEATURES = ["Recence", "Frequence", "Montant"]

# Modèle affiché -> (fichier joblib, clé dans metrics.json)
MODELES_DISPONIBLES = {
    "Random Forest (forêt aléatoire)": ("random_forest.joblib", "random_forest"),
    "Régression logistique": ("logistic_regression.joblib", "logistic_regression"),
    "K plus proches voisins (KNN)": ("knn.joblib", "knn"),
}

# Noms des segments (vérifiés sur le profil réel des clusters du notebook).
NOMS_SEGMENTS = {
    0: "Clients intermédiaires",
    1: "Clients à forte valeur",
    2: "Clients inactifs / occasionnels",
}


# ---------------------------------------------------------------------------
# Chargement des modèles et des métriques (mis en cache par Streamlit).
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def charger_modele(nom_fichier: str):
    """Charge un modèle joblib depuis le dossier models/."""
    return joblib.load(MODELS_DIR / nom_fichier)


@st.cache_resource(show_spinner=False)
def charger_metriques():
    """Charge models/metrics.json (accuracy / F1 de chaque modèle)."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def predire(modele, client: pd.DataFrame):
    """Retourne (cluster, segment, confiance) pour un client RFM."""
    probabilites = modele.predict_proba(client)[0]
    classes = list(modele.classes_)
    index = int(np.argmax(probabilites))          # classe la plus probable
    cluster = int(classes[index])
    confiance = float(probabilites[index])
    segment = NOMS_SEGMENTS.get(cluster, "Segment inconnu")
    return cluster, segment, confiance


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Segmentation clients", page_icon="🧭", layout="centered")

st.title("🧭 Segmentation des clients")
st.write(
    "Saisissez les informations RFM d'un client pour découvrir à quel **segment** "
    "il appartient. Le calcul est effectué par le modèle d'apprentissage choisi."
)

# Choix du modèle (Random Forest par défaut).
nom_modele = st.selectbox(
    "Modèle de prédiction",
    options=list(MODELES_DISPONIBLES.keys()),
    index=0,  # 0 = Random Forest, choix par défaut
    help="Le Random Forest est le modèle principal recommandé.",
)
fichier_modele, cle_metrique = MODELES_DISPONIBLES[nom_modele]

st.subheader("Informations du client")
col1, col2, col3 = st.columns(3)
with col1:
    recence = st.number_input(
        "Récence (jours)",
        min_value=0,
        max_value=1000,
        value=30,
        step=1,
        help=(
            "Nombre de jours écoulés depuis le dernier achat du client. "
            "Dans ce projet, il est compté par rapport à la date de référence "
            "du jeu de données (10/12/2011, lendemain de la dernière facture). "
            "Une valeur faible = client récent."
        ),
    )
with col2:
    frequence = st.number_input(
        "Fréquence (commandes)",
        min_value=1,
        max_value=1000,
        value=5,
        step=1,
        help="Nombre de commandes distinctes passées par le client. Une valeur élevée = client actif.",
    )
with col3:
    montant = st.number_input(
        "Montant (€)",
        min_value=0.0,
        max_value=1_000_000.0,
        value=1000.0,
        step=10.0,
        help="Somme totale dépensée par le client, en euros.",
    )

if st.button("Prédire", type="primary"):
    # Valeurs saisies regroupées dans un tableau (format attendu par le modèle).
    client = pd.DataFrame(
        {"Recence": [recence], "Frequence": [frequence], "Montant": [montant]}
    )

    try:
        modele = charger_modele(fichier_modele)
    except FileNotFoundError:
        st.error(
            f"Le fichier du modèle « {fichier_modele} » est introuvable dans le dossier models/. "
            "Exécutez d'abord la Partie 11 du notebook pour entraîner et sauvegarder les modèles."
        )
        st.stop()
    except Exception as erreur:  # noqa: BLE001 - message convivial pour tout autre problème
        st.error(f"Impossible de charger le modèle « {nom_modele} » : {erreur}")
        st.stop()

    # Validation des entrées (sécurité supplémentaire).
    if montant < 0 or frequence < 1 or recence < 0:
        st.warning("Veuillez saisir des valeurs valides (positives).")
        st.stop()

    try:
        cluster, segment, confiance = predire(modele, client)
    except Exception as erreur:  # noqa: BLE001
        st.error(f"Erreur pendant la prédiction : {erreur}")
        st.stop()

    # --- Résultat principal ---
    st.success(f"**Cluster prédit : {cluster}** — {segment}")

    st.write("**Valeurs RFM saisies :**")
    st.dataframe(client, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Confiance du modèle", f"{confiance * 100:.1f} %")
    with col_b:
        metriques = charger_metriques()
        accuracy = metriques.get(cle_metrique, {}).get("accuracy")
        if accuracy is not None:
            st.metric("Accuracy du modèle (test)", f"{accuracy * 100:.2f} %")
        else:
            st.metric("Accuracy du modèle (test)", "—")

    # --- Bonus : comparaison de tous les modèles ---
    st.subheader("Comparaison de tous les modèles")
    lignes = []
    for label, (fichier, _) in MODELES_DISPONIBLES.items():
        try:
            autre_modele = charger_modele(fichier)
            c, s, conf = predire(autre_modele, client)
            lignes.append(
                {
                    "Modèle": label,
                    "Cluster": c,
                    "Segment": s,
                    "Confiance": f"{conf * 100:.1f} %",
                }
            )
        except FileNotFoundError:
            lignes.append(
                {"Modèle": label, "Cluster": "—", "Segment": "modèle non sauvegardé", "Confiance": "—"}
            )
        except Exception:  # noqa: BLE001
            lignes.append(
                {"Modèle": label, "Cluster": "—", "Segment": "indisponible", "Confiance": "—"}
            )
    st.dataframe(pd.DataFrame(lignes), hide_index=True)

st.caption(
    "Projet ClientIQ — segmentation RFM (K-Means k=3) et classification supervisée."
)
