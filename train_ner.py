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