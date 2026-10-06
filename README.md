# Segmentation Client (RFM + K-Means + Random Forest + Streamlit)

## Objectif du projet

Ce projet réalise une segmentation de clients à partir de données transactionnelles. Il combine :

- Analyse RFM (Récence, Fréquence, Montant)
- Clustering K-Means pour identifier des groupes de clients
- Modélisation avec Random Forest
- Application interactive avec Streamlit

## Structure du projet

```text
ClientIQ/
├── data/
│   └── data.csv
├── models/
├── app/
│   └── streamlit_app.py
├── enonce-6ab96c62165f6088665164.ipynb
├── Notebook.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Comment lancer le notebook

1. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

2. Installer les dépendances

```bash
pip install -r requirements.txt
```

3. Lancer Jupyter Notebook

```bash
jupyter notebook
```

4. Ouvrir `Notebook.ipynb` depuis l'interface Jupyter.