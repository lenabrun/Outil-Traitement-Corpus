language: fr
multilinguality: monolingual
pretty_name: NERurse_Fr
task_categories:
    - named-entity-recognition
task_ids:
    - named-entity-recognition
tags:
    - medical
    - ner

---

# NERurse_Fr – Extraction d'Entités Nommées Médicales en Français

*NERurse_Fr* est un projet de traitement automatique du langage naturel visant à extraire des entités nommées du domaine médical dans des textes en français.

## Structure du projet

- `main.py` : Script principal pour le traitement et l'analyse.
- `train_ner.py` : Script pour l'entraînement du modèle.
- `evaluate_ner.py` : Script pour l'évaluation du modèle.

- `src/scraper.py` : Classe et fonctions pour scraper des articles sur internet.
- `src/utils.py` : Classe et fonctions pour pré-traiter et segmenter le corpus.
- `src/stats.py` : Classe et fonctions pour calculer des statistiques.
- `src/annotator.py` : Classe et fonctions pour annoter automatiquement le corpus.
- `src/data_augmentation.py` : Classe et fonctions pour augmenter le jeu de données.

- `data/` : Dossier contenant le jeu de données aux différentes étapes du traitement.
- `outputs/` : Dossier contenant les statistiques, résultats et graphiques liés aux différentes étapes du traitement.

## Description

Ce corpus *NERurse_Fr* est constitué d'articles français issus de la section [Santé](https://theconversation.com/fr/sante) du site [The Conversation](https://theconversation.com/fr) avec pour objectif l'extraction d'entités nommées spécialisées dans le **domaine médical.



