## TP 1
### Partie 1 | Étude de cas CoNLL 2003
**Quel type de tâche propose CoNLL 2003 ?**  
⇒ CoNLL 2003 propose la tâche de reconnaissance d’entités nommées, en particulier les personnes, les lieux, les organisations et les autres entités qui ne font pas parties des trois catégories précédentes.  

**Quel type de données y a-t-il dans CoNLL 2003 ?**  
⇒ Les données du CoNLL 2003 se présentent en 4 colonnes, chacune séparée par un simple espace. La première colonne contient un mot, la deuxième son étiquette de partie du discours POS, la troisième son étiquette de groupe syntaxique (GN, GV…), et la quatrième son type d’entité nommée.  

**À quel besoin répond CoNLL 2003 ?**  
⇒ CoNLL 2003 propose un corpus annoté pour des modèles qui cherchent à faire de la reconnaissance d’entités nommées.  

**Quels types de modèles ont été entraînés sur CoNLL 2003 ?**  
⇒ Des modèles de reconnaissances d’entités nommées (NER) comme bert-base-NER et des modèles utilisant Flair ont été entrainés sur le CoNLL 2003.  

**Est-ce un corpus monolingue ou multilingue ?**  
⇒ CoNLL 2003 est un corpus multilingue qui a permis à des modèles de s’entraîner dessus pour plusieurs langues comme l’anglais, l’allemand et le français.  
#
### Partie 2 | Projet   
**Dans quel besoin vous inscrivez-vous ?**  
⇒ Le besoin est d'automatiser l'extraction d'informations structurées à partir d'articles scientifiques, afin de faciliter la recherche, l'indexation et l'analyse de contenus spécialisés. Cela répond à une problématique de surcharge informationnelle dans le domaine scientifique, où la quantité de publications rend difficile la veille et la synthèse manuelle.  

**Quel sujet allez-vous traiter ?**  
⇒ L'extraction d'entités nommées (NER) dans des articles scientifiques en français, avec un accent particulier sur les entités spécifiques au domaine, telles que les noms de maladies, de composés chimiques, d'organismes de recherche, etc.  

**Quel type de tâche allez-vous réaliser ?**  
⇒  Une tâche de reconnaissance d'entités nommées (NER), qui est une tâche supervisée de traitement automatique du langage.  

**Quel type de données allez-vous exploiter ?**  
⇒ Des textes scientifiques en français, extraits d'articles académiques au format .txt, nettoyés et prétraités pour l'analyse linguistique.  

**Où allez vous récupérer vos données ?**  
⇒ Les données proviennent de la section Santé de The Conversation France, une plateforme en ligne qui publie des articles rédigés par des chercheurs et des universitaires, offrant des analyses approfondies sur divers sujets d'actualité dans le domaine des sciences et de la santé.  

**Sont-elles libres d'accès ?**  
⇒ Oui, les articles de The Conversation sont publiés sous la licence Creative Commons CC-BY-ND, ce qui permet leur réutilisation sans modification, à condition de citer correctement la source.  
   
## TP 2
### Récupération du corpus

Pour récupérer des articles scientifiques, j'ai d'abord consulté le `robots.txt` du site theconversations.com pour ensuite pouvoir scrapper les articles de la section santé. Pour ce script `scrapper.py`, j'ai utilisé les librairies python `requests` et `BeautifulSoup`. J'ai récupéré les articles des 50 premières pages pour avoir un peu plus de 3000 articles dans mon corpus. J'ai ensuite ouvert le code source d'une des pages pour pouvoir retenir quelle balise html contient les urls des articles. J'espace les requêtes d'une seconde chacune pour respecter les bonnes pratiques.

```
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
```

Grâce à cette liste d'urls, je peux ensuite ouvrir la page d'un article et identifier quelle balise contient le contenu textuel. Je réutilise le titre des articles pour les utiliser dans le nom des fichiers. Puis je crée un Dataframe avec la librairie Pandas qui permet d'accéder aux données facilement. Une première colonne `filename` du fichier au format CSV est dédiée au nom de l'article et la deuxième `text` à son contenu.

```
for idx, url in enumerate(article_urls, start=1):
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
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

            filepath = output_folder / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(article_text)

            if article_text:
                docs.append({
                    "filename": filename,
                    "text": article_text
                })

            print(f"Article sauvegardé : {filepath}")
        else:
            print(f"Contenu principal non trouvé pour l'article {url}.")
    else:
        print(f"Erreur lors de la requête pour l'article {url}: {response.status_code}")
```

### Évaluation des données

1. Pertinence des données
Les articles provenant de la section santé de The Conversation sont généralement rédigés par des universitaires et des chercheurs, ce qui garantit une certaine qualité et pertinence pour l'extraction d'entités nommées dans le domaine médical.

2. Types de données présentes
 Le corpus est constitué de textes journalistiques traitant de sujets médicaux et de santé publique. Ces textes peuvent contenir des entités telles que des noms de maladies, de médicaments, d'institutions médicales, etc.

3. Statistiques exploitables
Après le prétraitement, il serait intéressant de calculer des statistiques telles que la fréquence des entités nommées, la distribution des longueurs des articles, ou encore la densité d'entités par article.

4. Attributs majeurs
Les attributs clés du corpus sont les colonnes `filename` et `text`. Après le nettoyage, il serait utile d'ajouter de nouvelles colonnes, comme `clean_text` pour le texte prétraité, ou `entities` pour les entités extraites.


###

## TP 3
### Visualisation des statistiques

