## Objectif du projet

Ce projet realise une segmentation de clients à partir de données transactionnelles (jeu de données Online Retail). L'objectif est d'identifier des segments de clients grâce à une analyse RFM (Récence, Fréquence, Montant), à un clustering K-Means (k=3), puis à des classifieurs supervisés, et enfin de les exposer dans une application Streamlit.

## Structure du projet

- `data/data.csv` : données transactionnelles brutes
- `Notebook.ipynb` : analyse complète (EDA, RFM, K-Means, classification, Parties 9 à 12)
- `models/` : modèles entraînés sauvegardés au format `joblib` + `metrics.json`
- `app/streamlit_app.py` : application interactive de prédiction
- `requirements.txt` : dépendances Python

## Installation

Depuis la racine du projet :

```bash
pip install -r requirements.txt
```

## Lancer l'application Streamlit

Toujours depuis la racine du projet :

```bash
streamlit run app/streamlit_app.py
```

L'application s'ouvre dans le navigateur. L'utilisateur choisit un modèle (Random Forest par défaut), saisit les valeurs RFM brutes d'un client (jours, nombre de commandes, euros) puis clique sur **Prédire** pour obtenir le cluster, le nom du segment, la confiance et l'accuracy du modèle.

> Remarque : les modèles doivent d'abord être entraînés et sauvegardés en exécutant la Partie 11 du notebook (`Notebook.ipynb`).
