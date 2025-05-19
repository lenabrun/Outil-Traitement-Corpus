from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import pandas as pd
from tqdm import tqdm
from typing import List

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
    
