# GlycanGT
GlycanGT is a graph transformer–based pre-trained model for glycan analysis.
Built upon the TokenGT architecture, GlycanGT is pre-trained with masked language modeling (MLM) on glycan structures. Using the embeddings obtained from the pre-trained model, GlycanGT achieves state-of-the-art (SOTA) performance across multiple benchmark tasks.

Thanks to MLM pre-training, GlycanGT can also predict missing components—such as ambiguous monosaccharides and glycosidic linkages—in incompletely characterized glycans. This capability is particularly valuable since the majority of glycans in existing databases are registered as ambiguous sequences.

This repository provides instructions for environment setup, glycan embedding extraction with GlycanGT, and methods for predicting ambiguous glycan sequences.

# Installation

