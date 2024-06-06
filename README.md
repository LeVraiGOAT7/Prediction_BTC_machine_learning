# Projet de Prédiction des Prix des Cryptomonnaies (BTC, BNB, ETH)

Ce projet utilise des techniques de deep learning pour prédire les prix de clôture des cryptomonnaies Bitcoin (BTC), Binance Coin (BNB) et Ethereum (ETH).

## Table des Matières

1. [Description](#description)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Contributions](#contributions)
5. [Licence](#licence)
6. [Contact](#contact)

## Description

Ce projet vise à prédire les prix de clôture des cryptomonnaies BTC, BNB et ETH en utilisant des modèles de deep learning tels que LSTM et GRU. Les données sont récupérées via l'API Yahoo Finance et les modèles sont implémentés en Python. Une application Streamlit permet aux utilisateurs de choisir entre différents modèles et de varier les dates de prédiction.

## Installation

### Prérequis

- Python 3.12.1
- [Git](https://git-scm.com/)
- [pip](https://pip.pypa.io/en/stable/)

### Étapes d'installation

1. Clonez le dépôt :

```bash
git clone https://github.com/username/Prediction_BTC_machine_learning.git
cd Prediction_BTC_machine_learning

2. Installez les dépendances :

pip install -r requirements.txt

## Usage

Exécution de l'application Streamlit
Pour lancer l'application Streamlit :

streamlit run app.py


Configuration des dates
L'application permet de sélectionner des dates de début et de fin pour les prédictions. Les dates doivent être comprises entre le 30 novembre 2023 et une période de 2 ans après la date de début sélectionnée.

Modèles disponibles
L'application propose plusieurs modèles de prédiction parmi lesquels choisir, développés par vous et votre partenaire. Les modèles sont sauvegardés sous forme de fichiers .ipynb et peuvent être sélectionnés dans l'application.

Data Profiling
L'application inclut également une fonctionnalité de data profiling sur la page d'accueil pour analyser les données

## Contributions

Les contributions sont les bienvenues ! Pour contribuer :

Fork le projet.
Créez une branche pour votre fonctionnalité (git checkout -b feature/NouvelleFonctionnalité).
Committez vos changements (git commit -m 'Ajout de NouvelleFonctionnalité').
Poussez à la branche (git push origin feature/NouvelleFonctionnalité).
Ouvrez une Pull Request.
