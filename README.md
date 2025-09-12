# GlycanGT
GlycanGT is a graph transformer–based pre-trained model for glycan analysis.
Built upon [the TokenGT architecture](https://arxiv.org/abs/2207.02505), GlycanGT is pre-trained with masked language modeling (MLM) on glycan structures. Using the embeddings obtained from the pre-trained model, GlycanGT achieves state-of-the-art (SOTA) performance across multiple benchmark tasks.

Thanks to MLM pre-training, GlycanGT can also predict missing components—such as ambiguous monosaccharides and glycosidic linkages—in incompletely characterized glycans. This capability is particularly valuable since the majority of glycans in existing databases are registered as ambiguous sequences.

This repository provides instructions for environment setup, glycan embedding extraction with GlycanGT, and methods for predicting ambiguous glycan sequences.

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


# Citation
Citation will be available after publication.

