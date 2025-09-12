# GlycanGT
GlycanGT is a graph transformer–based pre-trained model for glycan analysis.
Built upon [the TokenGT architecture](https://arxiv.org/abs/2207.02505), GlycanGT is pre-trained with masked language modeling (MLM) on glycan structures. Using the embeddings obtained from the pre-trained model, GlycanGT achieves state-of-the-art (SOTA) performance across multiple benchmark tasks.

Thanks to MLM pre-training, GlycanGT can also predict missing components—such as ambiguous monosaccharides and glycosidic linkages—in incompletely characterized glycans. This capability is particularly valuable since the majority of glycans in existing databases are registered as ambiguous sequences.

This repository provides instructions for environment setup, glycan embedding extraction with GlycanGT, and methods for predicting ambiguous glycan sequences.

# Model architecture / Training details
GlycanGT is built upon the Tokenized Graph Transformer (TokenGT) architecture, in which both monosaccharides (nodes) and glycosidic linkages (edges) are treated as independent tokens. Each token embedding combines three components:  
1. A linear projection of token content features,  
2. Node identifiers encoded as orthogonal random features (ORFs), and  
3. Trainable type embeddings distinguishing nodes from edges.  

A special [graph] token is prepended, and all tokens are passed through stacked Transformer encoder layers with multi-head self-attention. The final [graph] token representation is used as the glycan embedding for downstream tasks.

Pretraining was performed using masked language modeling (MLM) on 83,740 glycans curated from GlyCosmos/GlyTouCan, with glycans containing ambiguous symbols excluded to avoid data leakage. Both node and edge tokens were masked at varying ratios (5–55%) and predicted from surrounding context. Among four model scales (ss, small, medium, large), the large model with a balanced 35% masking ratio provided consistently strong performance and was adopted for downstream evaluations. Optimization used AdamW (learning rate 1e−6, weight decay 0.01) with early stopping, and training was conducted on NVIDIA L40 GPUs.

# Performance benchmarks
We benchmarked GlycanGT against existing graph-based approaches, including SweetNet, GlycanAA, and relational graph convolutional networks (RGCN), across three tasks using the GlycanML benchmark datasets:

- **Taxonomy classification (Domain → Species):**  
  GlycanGT achieved the best Macro-F1 at most taxonomic levels, demonstrating its ability to capture global structural features beyond local motifs.

- **Glycosylation type prediction (N-linked, O-linked, free):**  
  GlycanGT reached a Macro-F1 of 0.932.

- **Immunogenicity prediction (binary):**  
  GlycanGT achieved the highest performance, with an AUPRC of 0.844.

![Performance benchmarks]()

In addition, embeddings learned by GlycanGT revealed biologically meaningful clusters in motif enrichment analyses, separating glycan classes such as sialylated N-glycans and O-glycan cores. The model also demonstrated strong performance in predicting missing monosaccharides and linkages in ambiguous glycans, achieving >80% top-5 accuracy even under heavy masking.

# Installation
## Dependencies
- Python 3.12
- torch 2.5.1+cu121
- torchvision 0.20.1+cu121
- triton 3.1.0
- glycowork 1.6.2
- scikit-learn 1.7.0
- numpy 1.26.4
- scipy 1.15.3

## Install GlycanGT 
Create aconda enviroment:
```bash
conda create -n glycangt python=3.12 -y
conda activate glycangt
```

Install PyTorch (CUDA 12.1 version):
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Install other dependencies:
```bash
pip install triton==3.1.0 glycowork==1.6.2 scikit-learn==1.7.0 numpy==1.26.4 scipy==1.15.3
```

# Get embeddings from the glycan IUPAC condensed
In this script, you can provide glycans written in **IUPAC condensed format** as input in a `.csv` or `.txt` file.  
The script reads the glycans, converts them into graph structures, and then extracts their embeddings using GlycanGT.  

Run the following script to obtain the embeddings:
```bash

```
# License
This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).  
You are free to use, modify, and distribute this software under the terms of the license.

# Citation
Citation will be available after publication.

