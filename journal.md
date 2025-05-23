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

Pour récupérer des articles scientifiques, je consulte d'abord le `robots.txt` du site theconversations.com pour ensuite pouvoir scrapper les articles de la section santé. Pour ce module `scraper.py`, je crée une classe `ArticleScraper` et j'utilise  les librairies python `requests` et `BeautifulSoup`. Dans une fonction `collect_urls()`, je récupère les articles des 25 premières pages pour avoir un peu plus de 1500 articles dans mon corpus, en ouvrant au préalable le code source d'une des pages pour pouvoir retenir quelle balise html contient les urls des articles. J'espace les requêtes d'une seconde chacune pour respecter les bonnes pratiques.

```
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
```

Grâce à cette liste d'urls, je peux ensuite ouvrir la page d'un article et identifier quelle balise contient le contenu textuel pour l'extraire dans une autre fonction `scrape_articles()`. Je réutilise le titre des articles pour les utiliser dans le nom des fichiers. Puis je crée un Dataframe avec la librairie Pandas qui permet d'accéder aux données facilement. Une première colonne `filename` du fichier au format CSV est dédiée au nom de l'article et la deuxième `text` à son contenu. Je sauvegarde le contenu textuel à la fois au format .txt mais aussi .csv grâce à une autre fonction `save_to_csv()`.

```
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

```

#
### Évaluation des données

1. Pertinence des données :  
Les articles provenant de la section santé de The Conversation sont généralement rédigés par des universitaires et des chercheurs, ce qui garantit une certaine qualité et pertinence pour l'extraction d'entités nommées dans le domaine médical.

2. Types de données présentes :  
 Le corpus est constitué de textes journalistiques traitant de sujets médicaux et de santé publique. Ces textes peuvent contenir des entités telles que des noms de maladies, de médicaments, d'institutions médicales, etc.

3. Statistiques exploitables :  
Après le prétraitement, il serait intéressant de calculer des statistiques telles que la fréquence des entités nommées, la distribution des longueurs des articles, ou encore la densité d'entités par article.

4. Attributs majeurs :  
Les attributs clés du corpus sont les colonnes `filename` et `text`. Après le nettoyage, il serait utile d'ajouter de nouvelles colonnes, comme `clean_text` pour le texte prétraité, ou `entities` pour les entités extraites.

#
### Pré-traitement des données

Pour le pré-traitement des données je crée une nouvelle classe `TextPreprocessor` dans le module `utils.py` où j'utilise quelques fonctions dans les premières étapes de mon projet :  
- `clean_text()` : fonction qui nettoie et normalise un texte donné (balises HTML, guillemets, apostrophes, espaces et marqueurs éditoriaux).
- `process_corpus()` : fonction qui lit le corpus depuis un fichier .csv, applique le nettoyage à chaque texte et sauvegarde le résultat dans un nouveau fichier .csv.
- `tokenize_text()` : fonction qui segmente les articles en phrases et tokens, initialise une colonne de 'O' pour l'annotation au format BIO, puis sauvegarde dans un fichier .csv.
- `split_corpus()` : fonction qui segmente le corpus en trois sous-corpus d'entraînement, de validation et de test.

#
### Annotation des données

Pour l'annotation des données dans un format BIO, j'annote d'abord 10 premières phrases manuellement en reprenant les labels d'un jeu de données médicales pour la même tâche de NER trouvé sur Hugging Face : [MedicalNER_FR](https://huggingface.co/datasets/TypicaAI/MedicalNER_Fr/viewer/default/train?q=vih&row=1837&views%5B%5D=train). Le processus étant extrêmement long, je décide d'essayer d'utiliser un modèle, qui a déjà été entraîné pour annoter ce genre de données, sur mon corpus. Ainsi je choisis le modèle [MedicalNER](https://huggingface.co/blaze999/Medical-NER) qui a été entraîné sur le jeu de données cité juste au dessus.  

Je fais cela en créant une classe `MedicalAnnotator` dans le module `annotator.py` en chargeant le modèle, le tokenizer et le dictionnaire permettant de traduire les indices de prédiction en labels BIO.

```
class MedicalAnnotator:
    def __init__(self, model_name="TypicaAI/HealthcareNER-Fr"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.id2label = self.model.config.id2label
```

J'utilise ensuite trois autres fonctions dans mon processus d'annotation :  
- `annotate_csv()` : fonction qui lit un corpus et y applique l'annotation automatique NER phrase par phrase.
- `fix_bio_labels()` : fonction qui corrige les séquences BIO invalides en appliquant deux règles principales :
   - si plusieurs B-XXX consécutifs sont détectés pour le même type d’entité, les suivants sont convertis en I-XXX.
   - si un I-XXX est mal positionné (ex : au début ou après un O), il est corrigé en B-XXX.
- `correct_bio_in_csv()` : fonction qui permet de corriger un fichier déjà annoté.

## TP 3
### Visualisation et statistiques

Pour pouvoir effectuer des statistiques sur mon corpus, je commence par créer une classe `EntityStats` dans un module `stats.py`. J'initialise cinq listes pour stocker les résultats des statistiques calculés :  
```
self.total_entities = 0
self.entities_per_doc: List[int] = []
self.entity_labels: List[str] = []
self.entity_texts: List[str] = []
```
Je crée ensuite trois fonctions :  
- `process_corpus()` : fonction qui parcourt tout le corpus et extrait les entités nommées, mettant ainsi à jour les listes créées précédemment.
- `compute_statistics()` : fonction qui calcule les statistiques globales à partir des données extraites précédemment, dont le nombre total de documents traités, la moyenne d'entités nommées par document, la répartition par type d’entité sous forme de Counter, la fréquence des entités textuelles les plus présentes.
- `display_results()` : fonction qui affiche les résultats statistiques dans la console.
- `plot_entity_distribution()` : fonction qui affiche un graphique de la répartition des types d'entités dans le corpus.


## TP 4
### Augmentation des données

J'implémente l'augmentation à travers la classe `NERDataAugmentor` du module `data_augmentation.py`. J'initialise deux attributs de la classe : un dictionnaire contenant pour chaque type d'entité une liste de variantes ou synonymes utilisés comme base de remplacement, et une probabilité que chaque entité éligible soit remplacée, ce qui permet de contrôler l'intensité de l'augmentation des données.  

Je crée ensuite une fonction `substitute_entities()` pour appliquer le remplacement des entités avec une série de conditions : 
 
1. Détection des entités commençant par un tag B-.
2. Regroupement des tokens B- et I- pour reconstituer chaque entité complète.
3. Si le type d’entité existe dans le lexique et que la probabilité tirée au sort est inférieure à augmentation_prob :
   - Un remplacement est effectué par une version alternative tirée du lexique.
   - Les nouveaux tokens sont correctement réétiquetés en BIO.
   - Un mécanisme est prévu pour éviter les doublons immédiats en contrôlant que deux remplacements successifs ne soient pas identiques.
  
**Exemple :**
```
tokens = ["le", "patient", "souffre", "de", "grippe"]
labels = ["O", "O", "O", "O", "B-DISEASE"]
```
⇒ *ici, "grippe" pourrait être remplacé par "pneumonie" pour crééer une nouvelle phrase si "pneumonie" fait partie du lexique des entités de type `DISEASE` passé à la classe.*

Puis j'utilise une deuxième fonction `augment_dataset()` qui applique la logique à tout un DataFrame et un nouveau est généré avec les données augmentées.


## TP 5
### Fine-tuning du modèle

Pour cette étape de fine-tuning, je choisis le modèle [CamemBERT NER](https://huggingface.co/Jean-Baptiste/camembert-ner) sur Hugging Face et je crée un script `train_ner.py` dédié à cela.

Les jeux d’entraînement et de validation sont chargés et les colonnes tokens et ner_tags sont converties depuis des chaînes en listes, puis transformées en objets de type Dataset (de la bibliothèque `datasets` de Hugging Face) dans un fonction `load_csv_ner()`. Cela permet de bénéficier des méthodes optimisées de Hugging Face pour le traitement par lot et l'entraînement.

```
def load_csv_ner(path):
    df = pd.read_csv(path)
    df["tokens"] = df["tokens"].apply(ast.literal_eval)
    df["ner_tags"] = df["ner_tags"].apply(ast.literal_eval)
    return Dataset.from_pandas(df)

train_dataset = load_csv_ner("data/splits/train.csv")
val_dataset = load_csv_ner("data/splits/dev.csv")

datasets = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})
```

Les mappings `label2id` et `id2label` sont ensuite créés. Ces dictionnaires assurent la conversion entre les étiquettes BIO (B-SYMPTOM, I-DISEASE, etc.) et des indices numériques, requis par le modèle lors de l’entraînement.

```
unique_labels = sorted({label for seq in datasets["train"]["ner_tags"] for label in seq})
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}
```

Puis dans une fonction `tokenize_and_align_labels()`, il est question d’aligner correctement les étiquettes BIO sur les sous-tokens générés par le tokenizer (CamemBERT utilise WordPiece). Le label est attribué uniquement au premier sous-token d’un mot et les autres sous-tokens reçoivent l’étiquette -100, pour être ignorés dans la fonction de perte.

```
def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            label_ids.append(label2id[example["ner_tags"][word_idx]])
        else:
            label_ids.append(label2id[example["ner_tags"][word_idx]] if example["ner_tags"][word_idx].startswith("I-") else -100)
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = label_ids
    return tokenized_inputs
```

J'utilise ensuite la métrique `SeqEval` qui calcule la précision, le rappel et le F1-score au niveau des entités. La fonction `compute_metrics()` permet de comparer les labels véridiques aux labels prédits par le modèle et d'ainsi calculer les résultats de l'entraînement.

```
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [id2label[label] for (pred, label) in zip(prediction, label_seq) if label != -100]
        for prediction, label_seq in zip(predictions, labels)
    ]
    true_predictions = [
        [id2label[pred] for (pred, label) in zip(prediction, label_seq) if label != -100]
        for prediction, label_seq in zip(predictions, labels)
    ]
    return metric.compute(predictions=true_predictions, references=true_labels)
```

Ayant un ordinateur peu puissant, j'ajuste les hyperparamètres pour que le temps de traitement soit plus raisonnable. J'ai limité le nombre d'epochs à 1 mais cela diminue drastiquement les performances du modèles, chose qui se reflète dans les résultats que j'ai pu obtenir à la fin de cet entraînement.

```
args = TrainingArguments(
    output_dir="./camembert-ner",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
)
```


## TP 6
### Évaluation du modèle

Je teste maintenant mon modèle sur le sous-ensemble de test pour juger sa robustesse dans des situations réelles. Je commence dans un nouveau script `evaluate_ner.py` par charger le modèle que j'avais enregistré après l'entraînement pour pouvoir le réutiliser, puis je fais le mapping et j'applique la fonction `tokenize_and_align_labels()`. J'utilise les mêmes métriques que durant l'entraînement et j'obtiens des résultats assez mitigés.

**Analyse globale**

| **Métrique**      | **Valeur** |
| ----------------- | ---------- |
| F1-score global   | **33.1 %** |
| Précision globale | 44.6 %     |
| Rappel global     | 26.4 %     |
| Accuracy globale  | 90.5 %     |
| Perte (loss)      | 0.485      |

- Le F1-score global est relativement faible (33.1 %), ce qui indique que le modèle a des difficultés à prédire correctement les entités nommées dans de nombreuses catégories, malgré une précision modérée.
- L’accuracy (90.5 %) est élevée, mais ce chiffre peut être trompeur dans les tâches de NER : il inclut aussi tous les tokens non annotés (typiquement "O") qui sont majoritaires.
- Le rappel global très bas (26.4 %) montre que le modèle rate une majorité des entités attendues.

**Analyse par entité**

- *Disease* : le modèle parvient relativement bien à détecter les entités de type "Disease".
- *MedicalProcedure* : Le modèle identifie incorrectement ou trop peu de procédures médicales. Il y a peut-être une trop grande variabilité lexicale.
- *Symptom* : Beaucoup de faux positifs (bonne précision, faible rappel), ce qui suggère que le modèle devine trop souvent des symptômes sans base suffisante. Cela indique un surapprentissage possible sur certaines expressions symptomatiques.
- *AnatomicalStructure, Medication/Vaccine, MISC* : Le modèle échoue totalement à détecter ces entités car soit  elles sont absentes ou rares dans l'entraînement, soit leur vocabulaire est trop variable pour être capté sans plus de données. Nous pouvons voir dans le fichier `/outputs/label_distribution.png` que ces trois labels sont en effet les moins représentés sur tout le corpus.

**Hypothèses**

À partir de ces résultats, nous pouvons formuler plusieurs hypothèses :  
1. Déséquilibre des classes :  
Certaines entités comme  *Medication/Vaccine* et *AnatomicalStructure* sont probablement sous-représentées en opposition aux autres classes, ce qui provoque un déséquilibre dans les performances du modèle qui favorise les classes sur-représentées au détriment des classes plus rares. Cela pourrait être amélioré en s'assurant que chaque label occupe une proportion équitable du corpus.

2. Qualité ou volume du corpus d'entraînement :  
Le nombre d'exemples annotés par type pourrait être insuffisant pour permettre une généralisation efficace, notamment pour les catégories à F1=0. Ayant été limitée par le matériel utilisé, le modèle pourrait être largement amélioré en augmentant simplement le nombre d'épochs et la taille des sous-corpus d'entraînement, de validation et de test.

3. Étiquettes mal encodées (BIO) :  
Malgré les fonctions de correction de l'annotation BIO, il reste encore de nombreuses erreurs comme des I- isolés ou mal formés dû à l'utilisation de l'annotation automatique d'un modèle pré-entrainé, ce qui nuit à l’apprentissage de mon modèle. Si le temps le permettait, en annotant manuellement une quantité suffisante de phrases et sans avoir recours à l'annotation automatique d'un modèle pré-entraîné, la fiabilité de l'annotation aurait pu être mieux assurée pour ensuite obtenir des résultats plus représentatifs des performances du modèle même.
