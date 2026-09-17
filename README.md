# 🎬 Movie Age Predictor (Scraping & NLP)

Ce projet d'Ingénierie des Langues (Université Paris 8) combine la collecte de données web (scraping) et le Traitement Automatique du Langage Naturel (NLP)[cite: 16]. L'objectif est de prédire automatiquement l'âge conseillé d'un film en se basant uniquement sur son synopsis[cite: 16].

## 📌 Architecture du projet
Le projet se divise en deux étapes majeures :
1. **Création du corpus (`scraper.py`)** : Extraction de 15 000 films depuis une base de données de cinéma en ligne, répartis en trois genres (Action, Romance, Horreur) à l'aide de `BeautifulSoup` et `requests`[cite: 12, 16].
2. **Classification Textuelle (`trainer.py`)** : Utilisation du modèle pré-entraîné **DistilBERT** d'HuggingFace[cite: 10]. Les étiquettes d'âge sont mappées sur 8 classes numériques[cite: 10]. Le modèle est fine-tuné sur 80% des données et évalué sur les 20% restants.

## 🚀 Installation & Prérequis

Assurez-vous d'avoir Python 3.8+ installé avec un accès à un GPU (CUDA) pour accélérer l'entraînement[cite: 10]. 
Installez les dépendances requises :

> pip install -r requirements.txt

## 🧠 Utilisation

**1. Récolter de nouvelles données (Optionnel)**
Si vous souhaitez générer vos propres fichiers JSON depuis le web :

> python scraper.py

**2. Entraîner et tester le modèle**
Assurez-vous d'avoir le fichier global `merged_movies.json` contenant la fusion de vos données[cite: 10], puis lancez :

> python trainer.py

Le modèle et le tokenizer seront automatiquement sauvegardés dans le dossier `./saved_model` à la fin des 5 epochs[cite: 10].

## 📊 Résultats & Métriques
Le modèle a été évalué en mesurant l'Accuracy, la Précision, le Recall et le F1-Score[cite: 10]. L'approche bidirectionnelle de BERT permet de saisir les nuances de violence ou de langage dans un synopsis afin de le classer avec pertinence[cite: 16]. Les métriques détaillées et les limites du système (taille des synopsis) sont discutées dans le `Rapport.pdf` inclus dans ce dépôt[cite: 16].
