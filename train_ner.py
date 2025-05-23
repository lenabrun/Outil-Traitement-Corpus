"""
Script de fine-tuning d'un modèle CamemBERT pour la reconnaissance d'entités nommées (NER) en français.

Étapes couvertes :
1. Chargement et préparation des données BIO annotées.
2. Tokenisation avec alignement des labels.
3. Définition du modèle CamemBERT pour la classification de tokens.
4. Entraînement avec la bibliothèque HuggingFace Trainer.
5. Évaluation et sauvegarde du modèle et des métriques.
"""

import pandas as pd
import ast
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments
from transformers import CamembertTokenizerFast, CamembertForTokenClassification, Trainer, DataCollatorForTokenClassification
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import evaluate

def load_csv_ner(path):
    """
    Charge un fichier CSV contenant des annotations NER (tokens, ner_tags) et le convertit en objet Dataset.

    :param path: Chemin du fichier CSV.
    :return: Un objet HuggingFace Dataset avec les colonnes 'tokens' et 'ner_tags'.
    """
    df = pd.read_csv(path)
    df["tokens"] = df["tokens"].apply(ast.literal_eval)
    df["ner_tags"] = df["ner_tags"].apply(ast.literal_eval)
    return Dataset.from_pandas(df)

# Chargement des datasets d'entraînement et de validation
train_dataset = load_csv_ner("data/splits/train.csv")
val_dataset = load_csv_ner("data/splits/dev.csv")

datasets = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})

# Création des mappings entre labels et IDs
unique_labels = sorted({label for seq in datasets["train"]["ner_tags"] for label in seq})
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

# Chargement du tokenizer
tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")

def tokenize_and_align_labels(example):
    """
    Tokenise une phrase (liste de mots) en sous-tokens et aligne les étiquettes NER en format BIO avec les sous-tokens.

    - Utilise -100 pour masquer les sous-tokens supplémentaires lors de l'entraînement.
    - Gère correctement les labels de type I-XXX pour les sous-tokens internes.

    :param example: Un dictionnaire avec les clés 'tokens' et 'ner_tags'.
    :return: Dictionnaire enrichi avec 'input_ids', 'attention_mask' et 'labels' alignés.
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
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=False)

model = CamembertForTokenClassification.from_pretrained(
    "camembert-base",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

metric = evaluate.load("seqeval")

def compute_metrics(p):
    """
    Calcule les métriques de performance NER à partir des prédictions et des vraies étiquettes.

    Utilise la librairie `seqeval` pour obtenir précision, rappel, F1 par entité.

    :param p: Tuple contenant (logits, labels) fournis par le Trainer.
    :return: Dictionnaire des scores de performance par classe.
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

# Définition des hyperparamètres et des options de logging pour l'entraînement du modèle
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

# Entraînement du modèle sur le jeu de train
trainer.train()

trainer.save_model("camembert-ner-finetuned")
tokenizer.save_pretrained("camembert-ner-finetuned")

# Évaluation finale du modèle et sauvegarde des métriques dans un fichier JSON
metrics = trainer.evaluate()
print(metrics)

# Sauvegarde des métriques
import json
with open("camembert-ner-finetuned/train_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)