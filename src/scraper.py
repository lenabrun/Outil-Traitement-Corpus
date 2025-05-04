"""
Module de scraping d'articles depuis la section santé du site The Conversation.

Ce module définit la classe ArticleScraper qui permet de :
- collecter les URLs des articles à partir d'une URL paginée,
- extraire et sauvegarder localement le contenu textuel des articles,
- compiler les articles dans un fichier CSV pour un traitement ultérieur.
"""

import requests
from bs4 import BeautifulSoup
import time
import os
from urllib.parse import urljoin
import pandas as pd
from pathlib import Path

class ArticleScraper:
    """
    Scraper d'articles depuis la section santé du site The Conversation.
    """

    def __init__(self, base_url: str, output_dir: Path, headers: dict = None, delay: float = 1.0):
        """
        Initialise le scraper.
        
        :param base_url: URL avec une variable pagination.
        :param output_dir: Dossier où sauvegarder les articles.
        :param headers: Headers pour les requêtes HTTP.
        :param delay: Délai entre les requêtes en secondes.
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.headers = headers or {'User-Agent': 'Mozilla/5.0'}
        self.delay = delay
        self.article_urls = []
        self.docs = []

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_urls(self, num_pages: int = 50):
        """
        Récupère les URLs des articles sur plusieurs pages.

        :param num_pages: Nombre de pages à parcourir.
        """
        for page_num in range(1, num_pages + 1):
            url = self.base_url.format(page_num)
            try:
                res = requests.get(url, headers=self.headers)
                res.raise_for_status()
                soup = BeautifulSoup(res.text, "html.parser")
                articles = soup.find_all("div", class_="relative")
                for article in articles:
                    link = article.find("a")
                    if link:
                        article_url = urljoin("https://theconversation.com", link['href'])
                        self.article_urls.append(article_url)
            except requests.RequestException as e:
                print(f"Erreur page {page_num}: {e}")
            time.sleep(1)  # éviter de spammer le serveur

        print(f"{len(self.article_urls)} articles trouvés.")

    def scrape_articles(self):
        """
        Scrape le contenu de chaque article et le sauvegarde dans un fichier.
        """
        # Scraper les articles
        for idx, url in enumerate(self.article_urls, start=1):
            try:
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extraction du contenu
                content_div = soup.find('div', itemprop='articleBody')
                if not content_div:
                    print(f"Contenu non trouvé pour {url}")
                    continue

                paragraphs = content_div.find_all('p')
                article_text = '\n'.join([para.get_text() for para in paragraphs])

                # Création du nom de fichier à partir du titre
                title_tag = soup.find('h1')
                title = title_tag.get_text().strip() if title_tag else "article"
                filename = f"{idx:03d}_{title[:50].replace(' ', '_').replace('/', '-')}.txt"
                filepath = self.output_dir/filename

                # Sauvegarde dans un fichier texte
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(article_text)

                # Ajout au DataFrame
                if article_text.strip():
                    self.docs.append({
                        "filename": filename,
                        "text": article_text
                    })
                    print(f"Article sauvegardé : {filepath}")
            except requests.RequestException as e:
                print(f"Erreur requête pour {url}: {e}")
    
    def safe_to_csv(self, csv_path: Path):
        """
        Sauvegarde le corpus en DataFrame CSV.
        
        :param csv_path: Chemin vers le fichier CSV.
        """
        df = pd.DataFrame(self.docs)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"{len(df)} documents enregistrés dans {csv_path}")

    
def main():
    scraper = ArticleScraper(
        base_url = "https://theconversation.com/fr/sante?page={}",
        output_dir=Path("../data/raw")
    )
    scraper.collect_urls(num_pages=50)
    scraper.scrape_articles()
    scraper.safe_to_csv(scraper.output_dir/"corpus.csv")

if __name__ == "__main__":
    main()