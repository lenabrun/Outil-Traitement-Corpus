"""
Script principal pour le traitement d'un corpus médical en français.

Ce script exécute les étapes suivantes :
1. Scraping d'articles depuis la section santé du site The Conversation.
2. Prétraitement du texte : nettoyage, tokenisation, segmentation en train, dev, test.
3. Extraction d'entités nommées à l'aide de spaCy et calcul de statistiques.
4. Visualisation de la distribution des entités extraites.

Modules utilisés :
- ArticleScraper : pour collecter et sauvegarder les articles.
- TextPreprocessor : pour nettoyer et segmenter le corpus.
- EntityStats : pour extraire les entités nommées et générer des statistiques.
"""

from pathlib import Path
from src.scraper import ArticleScraper
from src.utils import TextPreprocessor
from src.stats import EntityStats
from src.annotator import MedicalAnnotator
from src.data_augmentation import NERDataAugmentor

import pandas as pd # à supprimer
import ast # à supprimer


def main():
    # Étape 1 : scraping
    #scraper = ArticleScraper(
    #    base_url = "https://theconversation.com/fr/sante?page={}",
    #    output_dir=Path("data/raw")
    #)
    #scraper.collect_urls(num_pages=25)
    #scraper.scrape_articles()
    #scraper.save_to_csv(scraper.output_dir/"raw_corpus.csv")

    # Étape 2 : pré-traitement
    # preprocessor = TextPreprocessor(
    # #    input_csv_path=Path("data/raw/raw_corpus.csv"),
    # #    output_csv_path=Path("data/clean/clean_corpus.csv")
    # )
    #preprocessor.process_corpus()

    # Étape 3 : tokenisation
    #preprocessor.tokenize_text(
    #    input_csv=Path("data/clean/clean_corpus.csv"),
    #    output_csv=Path("data/clean/tokenized_corpus.csv")
    #)

    # Étape 4 : annotation avec modèle médical Typica
    #annotator = MedicalAnnotator()
    #annotator.annotate_csv(
    #    input_csv_path="data/clean/tokenized_corpus.csv",
    #    output_csv_path="data/clean/auto-annotated.csv"
    #)

    # Étape 5 : segmentation
    # preprocessor.split_corpus(
    #    input_csv=Path("data/clean/full_augmented_dataset.csv"),
    #    output_dir=Path("data/clean")
    # )
    from sklearn.model_selection import train_test_split

    # Charger ton corpus augmenté
    df = pd.read_csv("data/clean/full_augmented_dataset.csv")
    df["tokens"] = df["tokens"].apply(ast.literal_eval)
    df["ner_tags"] = df["ner_tags"].apply(ast.literal_eval)
    df = df.sample(n=5000, random_state=42)


    # Split train/dev/test
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Sauvegarder
    train_df.to_csv("data/splits/train.csv", index=False)
    val_df.to_csv("data/splits/dev.csv", index=False)
    test_df.to_csv("data/splits/test.csv", index=False)


    # # Étape 6 : augmentation des données
    # df = pd.read_csv("data/clean/auto_annotated.csv")

    # lexicons = {
    # "Disease": ["diabète", "grippe", "COVID-19", "hypertension", "asthme", "cancer"],
    # "Treatment": ["paracétamol", "chimiothérapie", "repos"],
    # "Symptom": ["fièvre", "maux de tête", "toux persistante", "perte d'appétit"]
    # }

    # augmentor = NERDataAugmentor(lexicons=lexicons, augmentation_prob=0,6)
    # augmented_df = augmentor.augment_dataset(df)

    # combined = pd.concat([df, augmented_df], ignore_index=True)
    # combined.to_csv("data/clean/full_augmented_dataset.csv", index=False)
    # print("\nJeu de données augmenté sauvegardé.")

    # Étape 7 : statistiques
    # original_path = Path("data/clean/auto_annotated.csv")
    # augmented_path = Path("data/clean/full_augmented_dataset.csv")

    # stats = EntityStats(augmented_path)  # On analyse le corpus final
    # stats.process_bio_annotated_csv()

    # num_docs, avg_entities, label_counts, text_counts = stats.compute_statistics()
    # stats.display_results(num_docs, avg_entities, label_counts, text_counts)
    # stats.plot_entity_distribution(label_counts)

    # # Comparaison avec le corpus original
    # EntityStats.augmentation_stats(
    #     original_csv=original_path,
    #     augmented_csv=augmented_path,
    #     save_to=Path("outputs/augmentation_report.txt")
    # )
    # Étape 6 : évaluation

if __name__ == "__main__":
    main()