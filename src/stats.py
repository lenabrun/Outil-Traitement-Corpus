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
import ast

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
        Parcourt tout le corpus et extrait les entités nommées.
        """
        df = pd.read_csv(self.corpus_dir, encoding="utf-8")
        if 'clean_text' not in df.columns:
            print(f"La colonne 'clean_text' est absente du fichier.")
            return
        
        for text in df['clean_text'].dropna():
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
            print(f"{label}: {count}")

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

    @staticmethod
    def augmentation_stats(original_csv: Path, augmented_csv: Path, save_to: Path = None):
        """
        Compare la taille du corpus original et du corpus augmenté.

        :param original_csv: chemin vers le fichier CSV original
        :param augmented_csv: chemin vers le fichier CSV augmenté
        """
        df_orig = pd.read_csv(original_csv)
        df_aug = pd.read_csv(augmented_csv)

        n_orig = len(df_orig)
        n_aug = len(df_aug)

        augmentation_absolute = n_aug - n_orig
        augmentation_relative = (n_aug / n_orig) if n_orig > 0 else float('inf')

        report = (
            f"\nStatistiques d'augmentation du corpus :\n"
            f"Corpus original : {n_orig} documents\n"
            f"Corpus augmenté : {n_aug} documents\n"
            f"Augmentation absolue : +{augmentation_absolute} documents\n"
            f"Augmentation relative : x{augmentation_relative:.2f} fois\n"
        )
    
        print(report)

        if save_to:
            with open(save_to, "w", encoding="utf-8") as f:
                f.write(report)

    def process_bio_annotated_csv(self):
        """
        Extrait les entités nommées à partir d'un fichier CSV contenant les colonnes 'tokens' et 'ner_tags'.
        """
        df = pd.read_csv(self.corpus_dir, encoding="utf-8")
        if not {"tokens", "ner_tags"}.issubset(df.columns):
            print("Les colonnes 'tokens' et 'ner_tags' sont absentes du fichier.")
            return

        for tokens, tags in zip(df["tokens"], df["ner_tags"]):
            tokens = ast.literal_eval(tokens)
            tags = ast.literal_eval(tags)

            current_entity = []
            current_label = ""

            for token, tag in zip(tokens, tags):
                if tag.startswith("B-"):
                    if current_entity:
                        self.entities_per_doc.append(1)
                        self.entity_labels.append(current_label)
                        self.entity_texts.append(" ".join(current_entity))
                        self.total_entities += 1
                        current_entity = []

                    current_label = tag[2:]
                    current_entity = [token]

                elif tag.startswith("I-") and current_entity:
                    current_entity.append(token)

                elif tag == "O":
                    if current_entity:
                        self.entities_per_doc.append(1)
                        self.entity_labels.append(current_label)
                        self.entity_texts.append(" ".join(current_entity))
                        self.total_entities += 1
                        current_entity = []

            if current_entity:
                self.entities_per_doc.append(1)
                self.entity_labels.append(current_label)
                self.entity_texts.append(" ".join(current_entity))
                self.total_entities += 1


def main():
    """
    Point d'entrée du script pour le calcul de statistiques d'entités nommées.
    """
    corpus_path = Path("../data/clean/corpus.csv")
    stats = EntityStats(corpus_path)
    stats.process_corpus()
    num_docs, avg_entities, label_counts, text_counts = stats.compute_statistics()
    stats.display_results(num_docs, avg_entities, label_counts, text_counts)
    stats.plot_entity_distribution(label_counts)

if __name__ == "__main__":
    main()

