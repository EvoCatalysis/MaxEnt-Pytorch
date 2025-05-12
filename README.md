# MaxEnt Mutation Energy Estimation

This repository provides a 3-step workflow for mutation energy analysis using a Maximum Entropy (MaxEnt) model trained on multiple sequence alignment (MSA) data.

## 🔧 Installation

```bash
git clone https://github.com/EvoCatalysis/MaxEnt-Pytorch.git
cd Maxent-Pytorch
pip install -r requirements.txt
```

## 🧬 Workflow Overview

### 1. Preprocess MSA
Convert a FASTA file into one-hot encoded matrix and compute sequence weights.

```bash
python msa_lib.py --msa input.fasta --gap_cutoff 0.7
```

### 2. Train MaxEnt Model
Train a MaxEnt model using the processed MSA.

```bash
python train_maxent.py --device 'cuda' --epoch 1000
```

### 3. Predict Mutational Energies
Predict mutation energy changes based on the trained model.

```bash
python mut_energy_maxent_single_all.py
```

## 🧾 Files

| File                             | Description                                |
|----------------------------------|--------------------------------------------|
| `msa_lib.py`                     | Preprocess MSA file                        |
| `train_maxent.py`                | Train MaxEnt model on one-hot encoded MSA  |
| `mut_energy_maxent_single_all.py`| Compute mutational energies                |
| `requirements.txt`               | Python dependencies                        |


