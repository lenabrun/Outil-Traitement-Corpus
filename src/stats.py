"""
Module d'analyse statistique des entités nommées dans un corpus de textes.

Ce script charge des fichiers texte, extrait les entités nommées avec spaCy, calcule des statistiques globales et affiche les résultats, y compris des visualisations.
"""

import spacy
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from typing import List, Tuple

class EntityStats:
    """
    Classe pour le traitement de corpus et le calcul de statistiques sur les entités nommées.
    """

    def __init__(self, corpus_dir: Path, model: str = "fr_core_news_sm"):
        """
        Initialise l'analyseur avec un dossier de corpus et un modèle spaCy.
        
        :param corpus_dir: Répertoire contenant les fichiers .txt du corpus.
        :param model: Nom du modèle spaCy à utiliser pour l'extraction d'entités.
        """
        self.corpus_dir = corpus_dir
        self.model = model
        self.nlp = spacy.load(model)

        # Variables internes pour stocker les résultats
        self.total_entities = 0
        self.entities_per_doc: List[int] = []
        self.entity_labels: List[str] = []
        self.entity_texts: List[str] = []

    def process_corpus(self):
        """
        Parcourt tous les fichiers texte du corpus et extrait les entités nommées.
        """
        txt_files = list(self.corpus_dir.glob("*.txt"))
        if not txt_files:
            print(f"Aucun fichier trouvé dans {self.corpus_dir}")
            return
        
        for file in txt_files:
            with file.open(encoding="utf-8") as f:
                text = f.read().strip()
                doc = self.nlp(text)
                ents = doc.ents

                self.total_entities += len(ents)
                self.entities_per_doc.append(len(ents))
                self.entity_labels.extend([ent.label_ for ent in ents])
                self.entity_texts.extend([ent.text for ent in ents])

    def compute_statistics(self) -> Tuple[int, float, Counter, Counter]:
        """
        Calcule les statistiques sur les entités extraites.
        
        :return: Tuple contenant :
                - nombre de documents,
                - moyenne d'entités par documents,
                - fréquence par type d'entité (label),
                - entités les plus fréquentes.
        """
        num_docs = len(self.entities_per_doc)
        avg_entities = self.total_entities / num_docs if num_docs > 0 else 0
        label_counts = Counter(self.entity_labels)
        text_counts = Counter(self.entity_texts)
        return num_docs, avg_entities, label_counts, text_counts
    
    def display_results(self, num_docs: int, avg_entities: float, label_counts: Counter, text_counts: Counter):
        """
        Affiche les résultats statistiques dans la console.
        
        :param num_docs: Nombre total de documents.
        :param avg_entities: Nombre moyen d'entités par document.
        :param label_counts: Fréquence des types d'entités.
        :param text_counts: Fréquence des entités spécifiques.
        """
        print(f"Nombre de documents : {num_docs}")
        print(f"Nombre total d'entités : {self.total_entities}")
        print(f"Nombre moyen d'entités par document : {avg_entities:.2f}")

        print ("\nRépartition des types d'entités :")
        for label, count in label_counts.items():
            print(f"{text}: {count}")

    def plot_entity_distribution(self, label_counts: Counter):
        """
        Affiche un graphique de la répartition des types d'entités.
        
        :param label_counts: Fréquence des types d'entités.
        """
        if not label_counts:
            print("Aucune entité à afficher.")
            return
        
        labels, counts = zip(*label_counts.items())
        plt.figure(figsize=(10, 6))
        plt.bar(labels, counts, color="steelblue")
        plt.title("Répartition des types d'entités")
        plt.xlabel("Type d'entité")
        plt.ylabel("Fréquence")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def main():
    """
    Point d'entrée du script pour le calcul de statistiques d'entités nommées.
    """
    corpus_path = Path("../data/clean")
    stats = EntityStats(corpus_path)
    stats.process_corpus()
    num_docs, avg_entities, label_counts, text_counts = stats.compute_statistics()
    stats.display_results(num_docs, avg_entities, label_counts, text_counts)
    stats.plot_entity_distribution(label_counts)

if __name__ == "__main__":
    main()

