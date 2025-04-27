import spacy
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

# Charger le modèle français de spaCy
nlp = spacy.load("fr_core_news_sm")

data = Path("../data/clean")

# Initialiser les compteurs
total_entities = 0
entities_per_doc = []
entity_labels = []
entity_texts = []

# Lire les fichiers .txt du corpus
corpus = list(data.glob("*.txt"))

# Traiter chaque document
for file in corpus:
    with file.open(encoding="utf-8") as f:
        doc = f.read().strip()
        parsed_doc = nlp(doc)
        entities = parsed_doc.ents
        total_entities += len(entities)
        entities_per_doc.append(len(entities))
        entity_labels.extend([ent.label_ for ent in entities])
        entity_texts.extend([ent.text for ent in entities])

# Statistiques
num_documents = len(corpus)
average_entities = total_entities / num_documents if num_documents > 0 else 0
label_counts = Counter(entity_labels)
text_counts = Counter(entity_texts)

# Afficher les résultats
print(f"Nombre de documents : {num_documents}")
print(f"Nombre total d'entités : {total_entities}")
print(f"Nombre moyen d'entités par document : {average_entities:.2f}")
print("\nRépartition des types d'entités :")
for label, count in label_counts.items():
    print(f"  {label}: {count}")
print("\nEntités les plus fréquentes :")
for text, count in text_counts.most_common(10):
    print(f"  {text}: {count}")

# Visualiser la répartition des types d'entités
labels, counts = zip(*label_counts.items())
plt.figure(figsize=(10, 6))
plt.bar(labels, counts)
plt.title("Répartition des types d'entités")
plt.xlabel("Type d'entité")
plt.ylabel("Fréquence")
plt.show()
