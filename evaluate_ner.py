"""
Script d'évaluation d'un modèle CamemBERT fine-tuné pour la reconnaissance d'entités nommées (NER) en français.

Ce script effectue les opérations suivantes :
1. Chargement du jeu de test (format CSV avec colonnes 'tokens' et 'ner_tags').
2. Chargement du modèle fine-tuné et de son tokenizer.
3. Tokenisation avec alignement des étiquettes BIO.
4. Évaluation à l'aide de la métrique seqeval.
5. Sauvegarde des résultats dans un fichier JSON.

Modèle attendu : camembert-ner-finetuned
Format des données : colonnes 'tokens' et 'ner_tags' contenant des listes.
"""

import ast
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import CamembertTokenizerFast, CamembertForTokenClassification, Trainer, DataCollatorForTokenClassification
import evaluate
import json

def load_csv_ner(path):
    """
    Charge un fichier CSV contenant des annotations NER et retourne un objet HuggingFace Dataset.

    :param path: Chemin du fichier CSV à charger.
    :return: Dataset contenant les colonnes 'tokens' et 'ner_tags'.
    """
    df = pd.read_csv(path)
    df["tokens"] = df["tokens"].apply(ast.literal_eval)
    df["ner_tags"] = df["ner_tags"].apply(ast.literal_eval)
    return Dataset.from_pandas(df)

# Chargement du dataset de test
test_dataset = load_csv_ner("data/splits/test.csv")

# Chargement du tokenizer et du modèle fine-tuné
tokenizer = CamembertTokenizerFast.from_pretrained("camembert-ner-finetuned")
model = CamembertForTokenClassification.from_pretrained("camembert-ner-finetuned")

# Création des mappings entre labels et IDs
label_list = model.config.id2label
label_list = [label_list[i] for i in range(len(label_list))]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

def tokenize_and_align_labels(example):
    """
    Tokenise une phrase mot à mot, puis aligne les étiquettes NER sur les sous-tokens.

    - Applique la stratégie -100 pour ignorer les sous-tokens non initiaux.
    - Aligne correctement les labels de type B-XXX / I-XXX avec les sous-mots.

    :param example: Dictionnaire avec les clés 'tokens' et 'ner_tags'.
    :return: Dictionnaire avec 'input_ids', 'attention_mask' et 'labels'.
    """
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

# Tokénisation et alignement des labels
tokenized_test = test_dataset.map(tokenize_and_align_labels, batched=False)

metric = evaluate.load("seqeval")

def compute_metrics(p):
    """
    Calcule les métriques NER sur les prédictions du modèle à l'aide de `seqeval`.

    :param p: Tuple contenant (predictions, labels) produit par Trainer.
    :return: Dictionnaire contenant les scores de performance (accuracy, F1, etc.).
    """
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

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
)

# Évaluation du modèle sur le jeu de test
metrics = trainer.evaluate(tokenized_test)
print(metrics)

# Sauvegarde des métriques dans un fichier JSON
with open("camembert-ner-finetuned/test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
