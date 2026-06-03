# RNA Structure Predictor

A bioinformatics machine learning project aimed at accurately predicting the 3D structural formations and folding patterns of RNA sequences. 

## Overview

Understanding RNA structure is critical for developing targeted therapeutics and understanding cellular functions. This project leverages Multiple Sequence Alignment (MSA) techniques and deep learning to predict structural labels from raw nucleotide sequences.

## Dataset

The repository handles massive genomic datasets:
- `train_sequences.csv` & `validation_sequences.csv`: Raw RNA sequences.
- `train_labels.csv` & `validation_labels.csv`: Ground truth structural annotations.
*(Note: V2 datasets represent extended, high-density data splits reaching hundreds of megabytes in size).*

## Architecture

- **`rna_structure_predictor.py`**: The core deep learning pipeline. It ingests the CSV sequences, applies necessary tokenization/MSA preprocessing, and trains the predictive model.
- **`MSA/`**: Contains helper modules and cached alignments for Multiple Sequence Alignment processing, crucial for identifying evolutionary conserved structural motifs.

## Getting Started

Due to the massive size of the `.v2.csv` files, ensure you are running the training scripts on a machine with sufficient RAM and GPU memory.
```bash
python rna_structure_predictor.py
```
