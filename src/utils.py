"""
Module de prétraitement de texte pour l'extraction d'entités nommées dans des articles scientifiques.

Ce module lit un corpus depuis un fichier CSV, nettoie le texte en supprimant les balises HTML et les éléments d'interface utilisateur,
et applique une tokenisation. Le résultat est sauvegardé dans un nouveau fichier CSV.
"""

import re
import ast
import json
from pathlib import Path
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split

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
        text = self.html_tags.sub('', text)
        text = self.ui_elements.sub('', text)
        text = text.lower()
        text = text.replace('«', '"').replace('»', '"').replace('’', "'")
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
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

    def split_corpus(self, input_csv: str, output_dir: str, test_size: float = 0.2, dev_size: float = 0.1, random_state: int = 42):
        """
        Divise le corpus en sous-ensembles de développement, d'entraînement et de test.

        :param input_csv: Chemin vers le fichier CSV contenant le corpus.
        :param output_dir: Répertoire de sortie pour les fichiers divisés.
        :param test_size: Proportion du corpus à inclure dans l'ensemble de test.
        :param dev_size: Proportion du corpus à inclure dans l'ensemble de développement.
        :param random_state: Graine pour le générateur de nombres aléatoires.
        """
        df = pd.read_csv(input_csv)
        df["tokens"] = df["tokens"].apply(ast.literal_eval)
        df["ner_tags"] = df["ner_tags"].apply(ast.literal_eval)
        df = df.sample(n=5000, random_state=42)
        train_dev, test = train_test_split(df, test_size=test_size, random_state=random_state)
        train, dev = train_test_split(train_dev, test_size=dev_size / (1 - test_size), random_state=random_state)

        train.to_csv(output_dir/"train.csv", index=False)
        dev.to_csv(output_dir/"dev.csv", index=False)
        test.to_csv(output_dir/"test.csv", index=False)
        print(f"Corpus divisé en train ({len(train)}), dev ({len(dev)}), test ({len(test)})")

if __name__ == "__main__":
    input_path = Path("data/raw/raw_corpus.csv")
    cleaned_path = Path("data/clean/clean_corpus.csv")
    tokenized_path = Path("data/clean/tokenized_corpus.csv")
    output_dir = Path("data/clean")

    preprocessor = TextPreprocessor(input_csv_path=input_path, output_csv_path=cleaned_path)
    preprocessor.process_corpus()
    preprocessor.tokenize_text(
        input_csv=cleaned_path,
        output_csv=tokenized_path)
    preprocessor.split_corpus(input_csv=tokenized_path, output_dir=output_dir)