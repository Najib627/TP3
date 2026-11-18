# TP3 — Chaîne complète de Machine Learning sur des données Kaggle

**Auteur :** Najib Mounsif

## Objectif du TP

Choisir des exercices sur Kaggle et les traiter avec une chaîne de Machine Learning complète, de la donnée brute jusqu'au déploiement :

```
Data  →  EDA  →  Prétraitement  →  Transformation  →  ML  →  Évaluation  →  Déploiement
```

Le TP est divisé en deux parties, selon la taille des données :

| Partie | Dossier | Dataset | Taille | Problème |
|---|---|---|---|---|
| **Small** | [`small/`](small/) | [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) | 7 043 lignes × 21 colonnes | Prédire si un client va résilier son abonnement |
| **Big** | [`Big/`](Big/) | [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) | ~590 000 lignes × 434 colonnes | Prédire si une transaction bancaire est frauduleuse |

## Organisation du dépôt

```
TP3/
├── README.md                          ← ce fichier
├── .gitignore
├── small/                             ← Telco Customer Churn
│   ├── telco_churn_pipeline.ipynb
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── model/churn_model.joblib
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
└── Big/                               ← IEEE-CIS Fraud Detection
    ├── ieee_fraud_pipeline.ipynb
    ├── fraud_features.py
    ├── model/fraud_model.joblib
    ├── app.py
    ├── exemple_transactions.csv
    ├── requirements.txt
    └── README.md
```

---

## Étape 0 — Préparation de l'environnement

### Environnement Python

Sous Ubuntu, `pip install` est bloqué en dehors d'un environnement virtuel (erreur `externally-managed-environment`). J'ai donc créé un environnement virtuel dans le dossier du TP :

```bash
sudo apt install python3-venv python3-full
cd ~/Bureau/Big_data/TP3
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm joblib jupyter streamlit
```

### Récupération des données

- **Telco** : téléchargement direct du CSV depuis la page Kaggle du dataset.
- **IEEE-CIS** : c'est une compétition, il faut d'abord la **rejoindre** et accepter ses règles. J'ai ensuite téléchargé `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv` et `test_identity.csv` dans `Big/ieee-fraud-detection/`.
  - Attention : le fichier `sample_submission.csv` seul ne suffit pas, il ne contient que le modèle de soumission.
  - Les données de la compétition (~1,3 Go) ne sont **pas** sur GitHub : elles dépassent la limite de 100 Mo par fichier et les règles de Kaggle interdisent de les diffuser.

---

# Partie 1 — Small : Telco Customer Churn

### 1. Data
- Chargement du CSV : 7 043 clients, 21 colonnes (1 identifiant, 19 variables explicatives, 1 cible `Churn`).
- Inspection des types : `TotalCharges` est stockée comme texte alors qu'elle est numérique.

### 2. EDA (analyse exploratoire)
- **Cible déséquilibrée** : environ 73 % de clients restent et 27 % partent. L'accuracy seule est donc trompeuse.
- **Variables numériques** : les clients qui partent ont une **ancienneté faible** (médiane de 10 mois contre 38) et des **factures mensuelles plus élevées**.
- **Variables catégorielles** (taux de churn par modalité) :
  - contrat **mensuel** : environ 43 % de churn, contre 3 % pour un contrat de 2 ans ;
  - **fibre optique** : environ 42 % ;
  - paiement par **chèque électronique** : environ 45 % ;
  - le genre n'a quasiment aucun effet.
- Matrice de corrélation : `TotalCharges` est fortement corrélée à `tenure`.

### 3. Prétraitement
- Conversion de `TotalCharges` en numérique. Les **11 valeurs manquantes** correspondent à des clients avec `tenure = 0`, pas encore facturés : je les ai remplacées par **0**.
- Suppression de `customerID` et vérification des doublons et des valeurs aberrantes (règle IQR : aucune).
- Harmonisation : `No internet service` et `No phone service` deviennent `No`, et `SeniorCitizen` passe de 0/1 à No/Yes.
- Encodage de la cible : `Churn` Yes/No devient 1/0.

### 4. Transformation
- **Feature engineering** : nombre de services souscrits, dépense mensuelle moyenne réelle, tranche d'ancienneté, indicateur « contrat mensuel + chèque électronique ».
- **Normalisation** (StandardScaler) des variables numériques et **One-Hot Encoding** des catégorielles.
- Tout est regroupé dans un **Pipeline scikit-learn**, pour éviter que des informations du test servent pendant l'entraînement et pour pouvoir réutiliser le même objet au déploiement.
- Découpage train/test **stratifié** 80 / 20.

### 5. ML
- Comparaison de 4 modèles en **validation croisée stratifiée à 5 plis** : régression logistique, Random Forest, Gradient Boosting et XGBoost.
- Gestion du déséquilibre avec `class_weight="balanced"` et `scale_pos_weight`.
- Optimisation des hyperparamètres avec **GridSearchCV** (critère : ROC-AUC).

### 6. Évaluation
- Métriques sur le jeu de test : Accuracy, Precision, Recall, F1, ROC-AUC, matrices de confusion et courbes ROC.
- **Choix du seuil** de décision par validation croisée sur le train (et non sur le test).
- **Modèle retenu : XGBoost optimisé**

| Métrique (test) | Valeur |
|---|---|
| ROC-AUC | 0,847 |
| Recall (churn) | 0,77 |
| F1 (churn) | 0,63 |
| Accuracy | 0,76 |

- Facteurs principaux : contrat mensuel, faible ancienneté, fibre optique, chèque électronique, absence de support technique.

### 7. Déploiement
- Sauvegarde du pipeline complet et du seuil avec **joblib**.
- Fonction `predire_churn()` testée sur un client à risque (probabilité de 90 %) et sur un client fidèle (5 %).
- **Application Streamlit** (`app.py`) : un formulaire pour saisir un client, qui affiche sa probabilité de churn.

---

# Partie 2 — Big : IEEE-CIS Fraud Detection

### 1. Data
- Deux tables à fusionner sur `TransactionID` : `train_transaction` (394 colonnes) et `train_identity` (41 colonnes). L'identité n'existe que pour environ un quart des transactions.
- Dans le fichier de test, les colonnes s'appellent `id-01` au lieu de `id_01` : je les ai renommées.
- **Réduction de la mémoire** (float64 → float32) pour pouvoir travailler sur un PC.
- Option `SAMPLE_ROWS` pour travailler sur une partie des données si la RAM est insuffisante.

### 2. EDA
- **Cible très déséquilibrée** : environ 3,5 % de fraudes. Un modèle qui répond toujours « légitime » aurait 96,5 % d'accuracy : on utilise donc le **ROC-AUC** (métrique officielle) et le **PR-AUC**.
- **Montant** très asymétrique : j'ai travaillé aussi sur son logarithme.
- **Temps** : `TransactionDT` est un nombre de secondes, d'où j'ai déduit le jour et l'heure. Les données couvrent environ 6 mois, et le test Kaggle se situe **après** : le découpage doit donc être temporel.
- **Variables à risque** : produit `C`, cartes de crédit, transactions sur mobile, certains domaines e-mail (ex. protonmail).
- **Valeurs manquantes** très nombreuses. Leur absence est elle-même une information.
- Les **339 colonnes V** forment des blocs très redondants (corrélation > 0,95).

### 3. Prétraitement
Toutes ces décisions sont prises **sans utiliser la cible**, donc sans risque de fuite de données.
- Vérification des doublons et **tri chronologique**.
- Suppression des colonnes **vides à plus de 90 %** et des colonnes **quasi constantes**.
- **Réduction des colonnes V** : dans chaque bloc de colonnes ayant les mêmes valeurs manquantes, je ne garde qu'une colonne parmi celles corrélées à plus de 0,95.
- Harmonisation du texte (minuscules, espaces).
- Les valeurs manquantes sont **conservées** pour LightGBM et XGBoost, qui les gèrent nativement.

### 4. Transformation
Toutes les transformations sont dans une classe scikit-learn `FraudFeatures` (fichier `fraud_features.py`). Elle apprend uniquement sur le train et est réutilisée telle quelle pour le test, la soumission et l'application.
- **Identifiant client approximatif** `uid` = `card1` + `addr1` + `D1n`, où `D1n = jour − D1` correspond à peu près à la date d'ouverture du compte.
- **Encodage par fréquence** : nombre d'apparitions d'une carte, d'une adresse, d'un e-mail, d'un client…
- **Agrégations par client** : montant moyen et écart-type, et ratio « montant / montant habituel du client ».
- Variables temporelles (heure, jour de la semaine), partie décimale du montant, simplification de l'appareil, de l'OS et du navigateur.
- Encodage des catégorielles par entier. Une modalité inconnue vaut −1.
- **Découpage temporel** : les 80 % de transactions les plus anciennes servent au train, les 20 % les plus récentes au test.

### 5. ML
- Une validation est réservée : les 15 % les plus récents du train.
- Modèles de référence sur un échantillon : régression logistique et Random Forest.
- **XGBoost** et **LightGBM** sur toutes les données, avec arrêt anticipé.
- Optimisation de LightGBM (grille sur `num_leaves` et `colsample_bytree`).
- **Validation croisée temporelle** (`TimeSeriesSplit`) pour vérifier la stabilité.
- Modèle final ré-entraîné sur tout le train.

### 6. Évaluation
- Seuil de décision choisi sur la **validation** (maximisation du F1).
- ROC-AUC, PR-AUC, Precision, Recall, F1, courbes ROC et Precision-Recall, matrice de confusion.
- **Vue métier** : quelle part des fraudes et du montant fraudé on bloque en contrôlant les 0,5 %, 1 %, 2 %, 5 % ou 10 % de transactions les plus suspectes.
- **Stabilité** : AUC semaine par semaine sur la période de test.
- **Importance des variables** et **explication d'une prédiction** avec les valeurs SHAP de LightGBM.

| Métrique (test) | Valeur |
|---|---|
| ROC-AUC | *à compléter* |
| PR-AUC | *à compléter* |
| Recall (fraude) | *à compléter* |
| Score Kaggle (public / privé) | *à compléter* |

### 7. Déploiement
- Sauvegarde du pipeline (`FraudFeatures` + LightGBM + seuil) avec **joblib**.
- Fonction `predire_fraude()` testée sur un fichier d'exemple de 300 transactions.
- **Application Streamlit** (`app.py`) : on charge un CSV de transactions, l'application affiche le score de fraude de chacune et les alertes, avec un seuil réglable et le téléchargement des résultats.
- **Soumission Kaggle** (`submission.csv`) : validation officielle du modèle sur le jeu de test caché de la compétition.

---

## Exécution

```bash
cd ~/Bureau/Big_data/TP3
source venv/bin/activate

# Partie Small
cd small
jupyter notebook            # exécuter telco_churn_pipeline.ipynb
streamlit run app.py

# Partie Big (placer d'abord les CSV Kaggle dans Big/ieee-fraud-detection/)
cd ../Big
jupyter notebook            # exécuter ieee_fraud_pipeline.ipynb
streamlit run app.py
```

## Difficultés rencontrées et solutions

| Problème | Solution |
|---|---|
| `pip install` bloqué (`externally-managed-environment`) | Création d'un environnement virtuel `venv` |
| Mot de passe refusé lors du `git push` | Utilisation d'un **token d'accès personnel** GitHub |
| GitHub n'accepte pas `/` dans un nom de dépôt (`TP3/small`) | Un seul dépôt `TP3` avec des **sous-dossiers** `small/` et `Big/` |
| Données IEEE-CIS inaccessibles | Il faut **rejoindre la compétition** sur Kaggle avant de télécharger |
| Seul `sample_submission.csv` téléchargé | Téléchargement des fichiers `train_*` et `test_*` |
| Données trop volumineuses pour la RAM | Réduction mémoire (float32) et option `SAMPLE_ROWS` |
| Fichiers de données > 100 Mo | Exclusion des CSV avec `.gitignore` |

## Conclusion

Ce TP m'a permis d'appliquer la même chaîne de Machine Learning à deux échelles :
- sur un **petit jeu de données** propre (Telco), l'accent est mis sur l'analyse, l'encodage et la comparaison de modèles ;
- sur un **gros jeu de données** réel (IEEE-CIS), il faut aussi gérer la mémoire, les centaines de colonnes anonymes et redondantes, les valeurs manquantes, le fort déséquilibre des classes et la dimension **temporelle** (découpage passé → futur, stabilité du modèle).

Dans les deux cas, regrouper les transformations dans un **pipeline** sauvegardé permet de passer directement de l'expérimentation au **déploiement** avec Streamlit.
