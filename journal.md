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

Pour récupérer des articles scientifiques, j'ai d'abord consulté le `robots.txt` du site theconversations.com pour ensuite pouvoir scrapper les articles de la section santé. Pour ce module `scrapper.py`, j'ai utilisé les librairies python `requests` et `BeautifulSoup`. J'ai récupéré les articles des 50 premières pages pour avoir un peu plus de 3000 articles dans mon corpus. J'ai ensuite ouvert le code source d'une des pages pour pouvoir retenir quelle balise html contient les urls des articles. J'espace les requêtes d'une seconde chacune pour respecter les bonnes pratiques.

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

### Pré-traitement des données

```
class TextPreprocessor:
    """
    Classe pour le prétraitement de texte, incluant le nettoyage et la lemmatisation.
    """

    def __init__(self, input_csv_path: Path, output_csv_path: Path, model: str = "fr_core_news_sm"):
        """
        Initialise le préprocesseur avec les chemins d'entrée et de sortie, et charge le modèle spaCy.

        :param input_csv_path: Chemin vers le fichier CSV d'entrée contenant le corpus.
        :param output_csv_path: Chemin vers le fichier CSV de sortie pour sauvegarder le corpus nettoyé.
        :param model: Nom du modèle spaCy à utiliser pour la lemmatisation.
        """
        self.input_csv_path = input_csv_path
        self.output_csv_path = output_csv_path
        self.nlp = spacy.load(model)
        self.html_tags = re.compile(r'<[^>]+>')
        self.ui_elements = re.compile(r'(Abonnez|Partager|Lire aussi)', re.IGNORECASE)

    def clean_text(self, text: str) -> str:
        """
        Nettoie et normalise un texte donné.
        
        :param text: Texte brut.
        :return: Texte nettoyé et normalisé.
        """
        if pd.isnull(text):
            return ""
        text = self.html_tags.sub('', text) # Suppression des balises HTML
        text = self.ui_elements.sub('', text)   # Suppression des éléments d'interface utilisateur
        text = text.lower() # Convertir en minuscules
        text = text.replace('«', '"').replace('»', '"').replace('’', "'") # Remplacer les guillemets français par des guillemets anglais
        text = re.sub(r'\s+', ' ', text)    # Supprimer les espaces multiples
        text = text.strip() # Supprimer les espaces en début et fin de texte
        return text
    
    def process_corpus(self):
        """
        Lit le corpus depuis le fichier CSV, applique le nettoyage à chaque texte, et sauvegarde le résultat dans un nouveau fichier CSV.
        """
        df = pd.read_csv(self.input_csv_path, encoding="utf-8") # Lecture du fichier CSV
        df['clean_text'] = df['text'].apply(self.clean_text)    # Application du nettoyage à la colonne 'text'
        self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)  # Création du répertoire de sortie si necessaire
        df.to_csv(self.output_csv_path, index=False, encoding="utf-8")  # Sauvegarde du DataFrame nettoyé
        print(f"Corpus nettoyé enregistré dans {self.output_csv_path}")

    def tokenize_text(self, input_csv: Path, output_csv: Path):
        """
        Segmente les articles en phrases et tokens, puis sauvegarde dans un CSV.
        
        :param input_csv: Chemin vers le fichier CSV d'entrée.
        :param output_csv: Chemin vers le fichier CSV de sortie.
        """
        df = pd.read_csv(input_csv, encoding="utf-8")

        if 'clean_text' not in df.columns:
            print("La colonne 'clean_text' est absente du fichier.")
            return
        
        lines = []
        for idx, row in df.iterrows():
            text = row['clean_text']
            filename = row.get('filename', f"doc{idx}")
            doc = self.nlp(text)

            for i, sent in enumerate(doc.sents):
                tokens = [token.text for token in sent]
                ner_tags = ["O"] * len(tokens)  # initialiser tous les tags à "O"
                lines.append({
                    "filename": filename,
                    "sentence_id": f"{filename}_s{i}",
                    "sentence": sent.text.strip(),
                    "tokens": json.dumps(tokens, ensure_ascii=False),
                    "ner_tags": json.dumps(ner_tags, ensure_ascii=False)
                })

        df_output = pd.DataFrame(lines)
        df_output.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"{len(df_output)} phrases segmentées sauvegardées dans {output_csv}")
```


### Annotation des données

```
class MedicalAnnotator:
    def __init__(self, model_name="TypicaAI/HealthcareNER-Fr"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.id2label = self.model.config.id2label

    
    def annotate_csv(self, input_csv_path, output_csv_path, text_column="sentence"):
        df = pd.read_csv(input_csv_path)
        annotated_rows = []

        for sentence in tqdm(df[text_column], desc="Annotating"):
            encoding = self.tokenizer(sentence, return_tensors="pt", truncation=True)
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)

            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            labels = [self.id2label[label.item()] for label in predictions[0]]

            final_tokens, final_labels = [], []
            for token, label in zip(tokens, labels):
                if token.startswith("▁") or not token.startswith("<"):
                    if token not in self.tokenizer.all_special_tokens:
                        cleaned_token = token.replace("▁", "")
                        final_tokens.append(cleaned_token)
                        final_labels.append(label)
            
            fixed_labels = self.fix_bio_labels(final_labels)

            annotated_rows.append({
                "tokens": final_tokens,
                "ner_tags": fixed_labels
            })

        output_df = pd.DataFrame(annotated_rows)
        output_df.to_csv(output_csv_path, index=False)
        print(f"Corpus annoté sauvegardé : {output_csv_path}")


    def fix_bio_labels(self, labels: List[str]) -> List[str]:
        """
        Corrige les séquences BIO invalides :
        - Chaînes de B-XXX consécutifs → B-XXX, I-XXX, I-XXX, ...
        - I-XXX isolés ou mal enchaînés → B-XXX

        :param labels: Liste des étiquettes prédites
        :return: Liste corrigée des étiquettes BIO
        """
        fixed_labels = []
        prev_entity = None
        prev_label = "O"

        for I, label in enumerate(labels):
            if label == "O":
                fixed_labels.append("O")
                prev_label = "O"
                prev_entity = None
                continue

            # Corriger les labels mal formés (ex: "BDisease", "ISymptom")
            if "-" in label:
                tag, entity = label.split("-", 1)
            else:
                tag = label[0]
                entity = label[1:]

            if tag == "B":
                if prev_label in {"B", "I"} and prev_entity == entity:
                    # Si on enchaîne plusieurs B-XXX du même type -> on remplace par I-XXX
                    fixed_labels.append(f"I-{entity}")
                    prev_label = "I"
                else:
                    fixed_labels.append(f"B-{entity}")
                    prev_label = "B"
            elif tag == "I":
                if prev_label not in {"B", "I"} or prev_entity != entity:
                    # I-XXX mal enchaîné → devient B-XXX
                    fixed_labels.append(f"B-{entity}")
                    prev_label = "B"
                else:
                    fixed_labels.append(f"I-{entity}")
                    prev_label = "I"
            else:
                fixed_labels.append(label)
                prev_label = tag

            prev_entity = entity

        return fixed_labels
    
    def correct_bio_in_csv(self, input_csv_path: str, output_csv_path: str):
        """
        Corrige les erreurs dans les étiquettes BIO d'un fichier CSV déjà annoté.

        :param input_csv_path: Chemin vers le fichier annoté (ex: auto-annotated.csv, train.csv)
        :param output_csv_path: Chemin où sauvegarder le fichier corrigé
        """
        df = pd.read_csv(input_csv_path)

        if "ner_tags" not in df.columns:
            raise ValueError(f"Colonne 'ner_tags' manquante dans {input_csv_path}")

        corrected_rows = []

        for _, row in df.iterrows():
            tokens = eval(row["tokens"]) if isinstance(row["tokens"], str) else row["tokens"]
            labels = eval(row["ner_tags"]) if isinstance(row["ner_tags"], str) else row["ner_tags"]

            fixed_labels = self.fix_bio_labels(labels)

            corrected_rows.append({
                "tokens": tokens,
                "ner_tags": fixed_labels
            })

        corrected_df = pd.DataFrame(corrected_rows)
        corrected_df.to_csv(output_csv_path, index=False)
        print(f"BIO labels corrigés et enregistrés dans : {output_csv_path}")
```

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

J'ai implémenté l'augmentation à travers la classe `NERDataAugmentor` du module `data_augmentation.py`. J'initialise deux attributs de la classe : un dictionnaire contenant pour chaque type d'entité une liste de variantes ou synonymes utilisés comme base de remplacement, et une probabilité que chaque entité éligible soit remplacée, ce qui permet de contrôler l'intensité de l'augmentation des données.  

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

```
import pandas as pd
import ast
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments
from transformers import CamembertTokenizerFast, CamembertForTokenClassification, Trainer, DataCollatorForTokenClassification
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import evaluate

# 1. Chargement des données depuis CSV
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

# 2. Créer un mapping id <-> label
unique_labels = sorted({label for seq in datasets["train"]["ner_tags"] for label in seq})
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

# 3. Tokenisation
tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")

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

tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=False)

# 🧾 4. Modèle
model = CamembertForTokenClassification.from_pretrained(
    "camembert-base",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# 5. Évaluation
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

# 6. Entraînement
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

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("camembert-ner-finetuned")
tokenizer.save_pretrained("camembert-ner-finetuned")

# Évaluation après entraînement
metrics = trainer.evaluate()
print(metrics)

# Sauvegarde des métriques
import json
with open("camembert-ner-finetuned/train_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
```

## TP 6
### Évaluation du modèle
```
import ast
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import CamembertTokenizerFast, CamembertForTokenClassification, Trainer, DataCollatorForTokenClassification
import evaluate
import json

# Charger données de test
def load_csv_ner(path):
    df = pd.read_csv(path)
    df["tokens"] = df["tokens"].apply(ast.literal_eval)
    df["ner_tags"] = df["ner_tags"].apply(ast.literal_eval)
    return Dataset.from_pandas(df)

test_dataset = load_csv_ner("data/splits/test.csv")

# Charger tokenizer et modèle fine-tuné
tokenizer = CamembertTokenizerFast.from_pretrained("camembert-ner-finetuned")
model = CamembertForTokenClassification.from_pretrained("camembert-ner-finetuned")

# Mapping id2label
label_list = model.config.id2label
label_list = [label_list[i] for i in range(len(label_list))]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

# Tokenisation avec alignement
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

tokenized_test = test_dataset.map(tokenize_and_align_labels, batched=False)

# Métrique
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

# Évaluation avec Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
)

metrics = trainer.evaluate(tokenized_test)
print(metrics)

with open("camembert-ner-finetuned/test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
```
