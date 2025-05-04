"""
Script principal pour le traitement d'un corpus médical en français.

Ce script exécute les étapes suivantes :
1. Scraping d'articles depuis la section santé du site The Conversation.
2. Prétraitement du texte : nettoyage, lemmatisation, suppression des stop words et de la ponctuation.
3. Extraction d'entités nommées à l'aide de spaCy et calcul de statistiques.
4. Visualisation de la distribution des entités extraites.

Modules utilisés :
- ArticleScraper : pour collecter et sauvegarder les articles.
- TextPreprocessor : pour nettoyer et lemmatiser le texte.
- EntityStats : pour extraire les entités nommées et générer des statistiques.
"""

from pathlib import Path
from src.scraper import ArticleScraper
from src.preprocess import TextPreprocessor
from src.stats import EntityStats

def main():
    # Scraper les articles
    scraper = ArticleScraper(
        base_url = "https://theconversation.com/fr/sante?page={}",
        output_dir=Path("../data/raw")
    )
    scraper.collect_urls(num_pages=50)
    scraper.scrape_articles()
    scraper.safe_to_csv(scraper.output_dir/"corpus.csv")

    # Prétraiter le texte
    preprocessor = TextPreprocessor(input_csv_path=Path("../data/raw/corpus.csv"), output_csv_path=Path("../data/clean/corpus.csv"))
    preprocessor.process_corpus()

    # Calculer des statistiques
    stats = EntityStats(Path("../data/clean/corpus.csv"))
    stats.process_corpus()
    num_docs, avg_entities, label_counts, text_counts = stats.compute_statistics()
    stats.display_results(num_docs, avg_entities, label_counts, text_counts)
    stats.plot_entity_distribution(label_counts)

if __name__ == "__main__":
    main()