import requests
from bs4 import BeautifulSoup
import time
import os
from urllib.parse import urljoin

base_url = "https://theconversation.com/fr/sante?page={}"
headers = {
    'User-Agent': 'Mozilla/5.0'
}

# Dossier où enregistrer le contenu textuel des articles
output_folder = "../data"

# Initialiser une liste d'urls d'articles
article_urls = []

# Parcourir les 5 premières pages par exemple
for page_num in range(1, 50):
    url = base_url.format(page_num)
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        soup = BeautifulSoup(res.text, "html.parser")
        articles = soup.find_all("div", class_="relative")
        for article in articles:
            link = article.find("a")
            if link:
                article_url = urljoin("https://theconversation.com", link['href'])
                article_urls.append(article_url)
    else:
        print(f"Erreur page {page_num}: {res.status_code}")
    time.sleep(1)  # éviter de spammer le serveur

print(f"{len(article_urls)} articles trouvés.")

# Scrapper les articles
for idx, url in enumerate(article_urls, start=1):
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Supposons que le contenu principal de l'article est dans une balise <div> avec la classe 'article-body'
        content_div = soup.find('div', itemprop='articleBody')
        if content_div:
            paragraphs = content_div.find_all('p')
            article_text = '\n'.join([para.get_text() for para in paragraphs])

            # Nettoyer le titre pour en faire un nom de fichier
            title_tag = soup.find('h1')
            if title_tag:
                title = title_tag.get_text().strip()
                filename = f"{idx:03d}_{title[:50].replace(' ', '_').replace('/', '-')}.txt"
            else:
                filename = f"{idx:03d}_article.txt"

            filepath = os.path.join(output_folder, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(article_text)
            
            print(f"Article sauvegardé : {filepath}")
        else:
            print(f"Contenu principal non trouvé pour l'article {url}.")
    else:
        print(f"Erreur lors de la requête pour l'article {url}: {response.status_code}")