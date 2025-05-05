# NERurse_Fr – Extraction d'Entités Nommées Médicales en Français

*NERurse_Fr** est un projet de traitement automatique du langage naturel visant à extraire des entités nommées du domaine médical dans des textes en français.

---

## Métadonnées

language: fr
multilinguality: monolingual
pretty_name: NERurse_Fr
size_categories: 10K<n<100K
task_categories:
    - named-entity-recognition
task_ids:
    - named-entity-recognition
tags:
    - medical
    - ner

---

# Analyse des Entités Nommées dans un Corpus Médical

## Introduction

Ce projet analyse un corpus de textes appartenant au domaine de la santé en français pour permettre l'extraction d'entités nommées spécialisées au domaine médical.

## Structure du projet

- `main.py` : Script principal pour le traitement et l'analyse.
- `src/scraper.py` : Classe et fonctions pour scraper des articles sur internet.
- `src/preprocess.py` : Classe et fonctions pour pré-traiter le corpus.
- `src/stats.py` : Classe et fonctions pour calculer des statistiques.
- `data/clean/corpus.csv` : Corpus nettoyé au format CSV.

## Description

Ce corpus **NERurse_Fr** est constitué d'articles français issus de la section [Santé](https://theconversation.com/fr/sante) du site [The Conversation](https://theconversation.com/fr) avec pour objectif l'**extraction d'entités nommées** spécialisées dans le **domaine médical**.

## Structure du corpus

Le fichier `corpus.csv` contient les colonnes suivantes :

1. **filename**: nom du document.
2. **text**: texte brut de l'article.
3. **clean_text**: text nettoyé de l'article.
3. **ner_tags**: labels des entités nommées présentes.

## Compte des labels

**Nombre de documents** : 3350
**Nombre total d'entités** : 71300
**Nombre moyen d'entités par document** : 21.28

- MISC: 13350
- LOC: 28050
- ORG: 14800
- PER: 15100

## Installation

1. Cloner le dépôt :

   ```bash
   git clone https://github.com/lenabrun/Outil-Traitement-Corpus.git
   cd Outil-Traitement-Corpus
   ```

2. Créer un environnement virtuel :

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. Installer les dépendances :
    ```bash
    pip install -r requirements.txt
    python -m spacy download fr_core_news_sm
    ```

4. Exécuter le script principal :
    ```bash
    python main.py
    ```





