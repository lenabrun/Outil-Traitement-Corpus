import spacy
from pathlib import Path
import re

# Charger le modèle français de spaCy
nlp = spacy.load("fr_core_news_sm")

# Définir le chemin vers le dossier contenant les fichiers .txt
data_dir = Path("../data/raw")
output_dir = Path("../data/clean")

# Expressions régulières pour nettoyer le texte
html_tags = re.compile(r'<[^>]+>')
ui_elements = re.compile(r'(Abonnez|Partager|Lire aussi)', re.IGNORECASE)

# Parcourir tous les fichiers .txt
for file_path in data_dir.glob("*.txt"):
    with file_path.open(encoding="utf-8") as f:
        text = f.read()
        # Nettoyage du texte
        text = html_tags.sub('', text)
        text = ui_elements.sub('', text)
        # Traitement avec spaCy
        doc = nlp(text)
        # Lemmatisation et suppression des stop words
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        cleaned_text = ' '.join(tokens)
        # Sauvegarder le texte nettoyé
        output_file = output_dir / file_path.name
        with output_file.open("w", encoding="utf-8") as out_f:
            out_f.write(cleaned_text)
