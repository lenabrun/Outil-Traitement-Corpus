"""
Module d'augmentation d'un jeu de données NER dans un corpus de textes.
"""

import random
import pandas as pd
import ast
from typing import List, Tuple, Dict


class NERDataAugmentor:
    """
    Classe pour augmenter un jeu de données NER en remplaçant certains types d'entités
    par des exemples d’un lexique, avec un contrôle de probabilité.
    """

    def __init__(self, lexicons: Dict[str, List[str]], augmentation_prob: float = 1.0):
        """
        :param lexicons: Dictionnaire {entité: liste de termes} pour le remplacement.
        :param augmentation_prob: Probabilité de remplacer une entité (entre 0 et 1).
        """
        self.lexicons = lexicons
        self.augmentation_prob = augmentation_prob

    from typing import List, Tuple

    def substitute_entities(
        self,
        tokens: List[str],
        labels: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Remplace certaines entités nommées par des exemples du lexique, avec une probabilité donnée,
        en évitant les doublons immédiats.

        :param tokens: Liste de tokens de la phrase.
        :param labels: Liste de labels NER (format BIO) associés à chaque token.
        :return: Tuple contenant la nouvelle liste de tokens et les labels correspondants.
        """
        new_tokens, new_labels = [], []
        last_replacement = None
        i = 0

        while i < len(tokens):
            label = labels[i]

            if label.startswith("B-"):
                entity_type = label[2:]
                entity_tokens = [tokens[i]]
                j = i + 1
                while j < len(labels) and labels[j] == f"I-{entity_type}":
                    entity_tokens.append(tokens[j])
                    j += 1

                if (
                    entity_type in self.lexicons and
                    random.random() < self.augmentation_prob
                ):
                    replacement = random.choice(self.lexicons[entity_type]).split()

                    # Si la dernière entité insérée était la même, on évite la répétition
                    if replacement != last_replacement:
                        new_tokens.extend(replacement)
                        new_labels.extend(
                            ["B-" + entity_type] + ["I-" + entity_type] * (len(replacement) - 1)
                        )
                        last_replacement = replacement
                    else:
                        new_tokens.extend(entity_tokens)
                        new_labels.extend(
                            ["B-" + entity_type] + ["I-" + entity_type] * (len(entity_tokens) - 1)
                        )
                else:
                    new_tokens.extend(entity_tokens)
                    new_labels.extend(
                        ["B-" + entity_type] + ["I-" + entity_type] * (len(entity_tokens) - 1)
                    )
                i = j
            elif label.startswith("I-"):
                # Cas ignoré car déjà traité via le B-
                i += 1
            else:
                new_tokens.append(tokens[i])
                new_labels.append(labels[i])
                i += 1

        return new_tokens, new_labels


    def augment_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applique l’augmentation sur un DataFrame.

        :param df: DataFrame avec colonnes 'tokens' et 'ner_tags'.
        :return: DataFrame augmenté.
        """
        augmented_rows = []
        for _, row in df.iterrows():
            tokens = ast.literal_eval(row["tokens"])
            labels = ast.literal_eval(row["ner_tags"])
            new_tokens, new_labels = self.substitute_entities(tokens, labels)

            augmented_rows.append({
                "tokens": new_tokens,
                "ner_tags": new_labels
            })

        return pd.DataFrame(augmented_rows)
