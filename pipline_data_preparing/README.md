# Pipeline CubeSat EPS – Estimation du SOH batterie

Ce dépôt contient le pipeline de traitement de données pour l’estimation de l’état de santé (State of Health, **SOH**) des batteries d’un CubeSat.  
Il transforme les mesures brutes des capteurs en séquences temporelles normalisées, prêtes à alimenter un réseau LSTM.

La philosophie est **« configurer une fois, exécuter partout »** :  
seul le fichier `config.py` est à adapter à la mission. Les autres scripts (`feature_engineering.py`, `split.py`, `normalize.py`, `sequences_construction.py`) doivent rester **inchangés**.

---

## Structure du projet

| Fichier                      | Rôle |
|------------------------------|------|
| `config.py`                  | **Point central** : chemins, paramètres physiques, méthode de split, type de normalisation. À modifier pour chaque mission. |
| `feature_engineering.py`     | Construction des 20 features à partir des données brutes. |
| `split.py`                   | Découpage temporel (ou aléatoire) en ensembles d’entraînement et de test. |
| `normalize.py`               | Normalisation des features (StandardScaler ou MinMaxScaler) *apprise sur l’entraînement uniquement*. |
| `sequences_construction.py`  | Création des séquences glissantes `(fenêtre, features)` pour le LSTM. |
| `README.md`                  | Ce document. |

---

## Configuration

Ouvrez **`config.py`** et adaptez les sections suivantes à votre jeu de données :

### 1. Dataset
```python
DATA_PATH   = "processed/features.csv"   # CSV brut en entrée
ID_COL      = "battery_id"               # identifiant de la batterie
CYCLE_COL   = "cycle"                    # numéro de cycle (ordre temporel)
TARGET_COL  = "SOH"                      # variable à prédire

2.Paramètres physiques
ROLLING_WINDOW – taille de la fenêtre glissante (en cycles) pour les moyennes mobiles.

Seuils thermiques (COLD_THRESHOLD_C, ECLIPSE_THRESHOLD_K).

Tension nominale (NOMINAL_VOLTAGE_V).

Capacité nominale initiale (détection automatique si None).

3. Split train / test
SPLIT_METHOD : "temporal" (recommandé) ou "random".

TRAIN_RATIO : proportion des cycles les plus anciens pour l’entraînement.

RANDOM_STATE : graine pour le split aléatoire (si utilisé).

4. Normalisation
SCALER_TYPE : "standard" (StandardScaler, μ=0 σ=1) ou "minmax".

COLS_NO_SCALE : colonnes à exclure de la normalisation (par défaut cycle et eclipse_flag).

5. Sorties
Tous les fichiers produits seront sauvegardés dans le dossier défini par OUTPUT_DIR (par défaut "processed").

## Prérequis et installation
Python ≥ 3.8 avec les bibliothèques suivantes :

pandas

numpy

scikit-learn

joblib

Placez votre fichier CSV brut dans le répertoire défini par DATA_PATH (ou modifiez le chemin dans config.py).

## Pipeline d’exécution
L’ordre des scripts doit être respecté, car chaque étape consomme la sortie de la précédente.

## Étape 1 – Feature engineering
Construit les 20 features et les sauvegarde dans processed/features.csv.

Étape 2 – Split train / test
Découpe le jeu de données en deux fichiers train.csv et test.csv.


## Étape 3 – Normalisation
Fitte le scaler sur les données d’entraînement, puis normalise l’entraînement et le test.
Sauvegarde également le scaler pour une utilisation ultérieure (inférence embarquée).



## Étape 4 – Construction des séquences LSTM
Crée les tenseurs (N, fenêtre, features) pour le LSTM et les sauvegarde au format .npy.


Les paramètres du sliding window (taille de fenêtre, pas, horizon de prédiction) sont ajustables directement en haut de sequences_construction.py.


Construction des séquences LSTM et justification de la taille de fenêtre
Le script sequences_construction.py utilise un sliding window pour transformer la série temporelle en échantillons supervisés.

Les paramètres principaux sont :

WINDOW_SIZE = 30 – nombre de cycles passés utilisés pour prédire le SOH futur.

STEP_SIZE = 1 – pas du sliding window (1 = toutes les positions possibles).

HORIZON = 1 – prédiction du SOH au cycle suivant (t + 1).

## Pourquoi WINDOW_SIZE = 30 ?
Contexte orbital
Un CubeSat en orbite basse (LEO) effectue environ 15 orbites par jour. Chaque orbite comporte une phase d’éclipse et une phase de soleil, donc 30 cycles correspondent typiquement à 2 jours d’opération. Ce laps de temps capture des variations thermiques et électriques significatives sans être trop long.

## Dynamique de dégradation des batteries
Les mécanismes de vieillissement (SEI, lithium plating) évoluent sur plusieurs cycles, mais des tendances court-terme (ex. effet de la température moyenne) sont déjà visibles sur une trentaine de cycles. Une fenêtre plus courte (ex. 10) ne fournirait pas assez de contexte temporel, tandis qu’une fenêtre trop grande (ex. 100) diluerait les variations récentes et augmenterait inutilement la complexité calculatoire.

## Contrainte mémoire / embarquée
La longueur de séquence détermine la taille des tenseurs d’entrée du LSTM. Avec 20 features, 30 pas de temps représentent 600 valeurs par échantillon, ce qui reste compatible avec un microcontrôleur durci de type CubeSat. Une fenêtre plus grande rendrait le modèle difficile à embarquer.

## Validation empirique
Des tests préliminaires (non inclus) avec des données simulées et réelles ont montré qu’une fenêtre de 30 à 50 cycles offre un bon compromis entre stabilité de l’apprentissage et erreur de validation. La valeur 30 a été retenue comme un bon point de départ avant une éventuelle optimisation par recherche bayésienne.

## Règle d’or pratique
La taille de fenêtre est souvent choisie égale à la période dominante de la série (ici, une à deux journées) multipliée par le nombre d’observations par période. WINDOW_SIZE = 30 reflète cette heuristique tout en restant simple.

## Autres paramètres
STEP_SIZE = 1 : on génère tous les échantillons glissants possibles, ce qui maximise la taille du jeu d’entraînement. Aucune décimation n’est nécessaire pour les séries de quelques centaines de cycles.

HORIZON = 1 : on prédit le SOH au cycle immédiatement suivant la fenêtre. Ce choix correspond à la tâche de pronostic « un pas en avant », standard pour le suivi en temps réel de l’état de santé.

Adaptation : ces valeurs peuvent être modifiées directement dans sequences_construction.py. Pour une autre mission, il est conseillé de re-valider le WINDOW_SIZE en fonction de la durée typique des cycles et des objectifs de prédiction.


