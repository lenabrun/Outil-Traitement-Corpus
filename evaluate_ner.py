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
