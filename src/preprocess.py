"""
Module de prétraitement de texte pour l'extraction d'entités nommées dans des articles scientifiques.

Ce module lit un corpus depuis un fichier CSV, nettoie le texte en supprimant les balises HTML et les éléments d'interface utilisateur,
et applique une lemmatisation tout en supprimant les stop words et la ponctuation. Le résultat est sauvegardé dans un nouveau fichier CSV.
"""

import re
from pathlib import Path
import pandas as pd
import spacy

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
        Nettoie et lemmatise un texte donné.
        
        :param text: Texte brut à nettoyer.
        :return: Texte nettoyé et lemmatisé.
        """
        if pd.isnull(text):
            return ""
        # Suppression des balises HTML
        text = self.html_tags.sub('', text)
        # Suppression des éléments d'interface utilisateur
        text = self.ui_elements.sub('', text)
        # Traitement avec spaCy
        doc = self.nlp(text)
        # Lemmatisation et suppression des stop words et de la ponctuation
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        cleaned_text = ' '.join(tokens)
        return cleaned_text
    
    def process_corpus(self):
        """
        Lit le corpus depuis le fichier CSV, applique le nettoyage à chaque texte, et sauvegarde le résultat dans un nouveau fichier CSV.
        """
        # Lecture du fichier CSV
        df = pd.read_csv(self.input_csv_path, encoding="utf-8")
        # Application du nettoyage à la colonne 'text'
        df['clean_text'] = df['text'].apply(self.clean_text)
        # Création du répertoire de sortie si necessaire
        self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        # Sauvegarde du DataFrame nettoyé
        df.to_csv(self.output_csv_path, index=False, encoding="utf-8")
        print(f"Corpus nettoyé enregistré dans {self.output_csv_path}")

def main():
    """
    Point d'entrée principal du script.
    """
    input_csv = Path("../data/raw/corpus.csv")
    output_csv = Path("../data/clean/corpus_clean.csv")
    preprocessor = TextPreprocessor(input_csv, output_csv)
    preprocessor.process_corpus()

if __name__ == "__main__":
    main()