# Prédiction du churn client — Telco Customer Churn

Projet de Machine Learning (TP3) qui prédit si un client d'un opérateur télécom va résilier son abonnement.

**Dataset :** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7 043 clients, 21 colonnes)

## Chaîne de traitement

```
Data → EDA → Prétraitement → Transformation → ML → Évaluation → Déploiement
```

| Étape | Contenu |
|---|---|
| Data | Chargement, structure et types des colonnes |
| EDA | Distribution de la cible, analyse des variables numériques et catégorielles, corrélations |
| Prétraitement | Conversion de `TotalCharges`, valeurs manquantes, doublons, valeurs aberrantes, harmonisation des modalités |
| Transformation | Feature engineering (4 nouvelles variables), StandardScaler + OneHotEncoder dans un pipeline, split stratifié |
| ML | Régression logistique, Random Forest, Gradient Boosting, XGBoost — validation croisée 5 plis + GridSearchCV |
| Évaluation | Accuracy, Precision, Recall, F1, ROC-AUC, matrices de confusion, courbes ROC, choix du seuil, importance des variables |
| Déploiement | Pipeline sauvegardé avec joblib + application web Streamlit |

## Résultats

Modèle retenu : **XGBoost optimisé**

| Métrique (jeu de test) | Valeur |
|---|---|
| ROC-AUC | 0.847 |
| Recall (churn) | 0.77 |
| F1 (churn) | 0.63 |
| Accuracy | 0.76 |

Principaux facteurs de churn : contrat mensuel, faible ancienneté, fibre optique, paiement par chèque électronique, absence de support technique.

## Structure du projet

```
├── telco_churn_pipeline.ipynb            # notebook complet
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # données
├── model/churn_model.joblib              # modèle entraîné
├── app.py                                # application Streamlit
└── requirements.txt
```

## Exécution

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt jupyter matplotlib seaborn

jupyter notebook          # exécuter telco_churn_pipeline.ipynb
streamlit run app.py      # lancer l'application de prédiction
```
