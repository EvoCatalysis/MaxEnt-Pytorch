# MaxEnt Mutation Energy Estimation

This repository provides a 3-step workflow for mutation energy analysis using a Maximum Entropy (MaxEnt) model trained on multiple sequence alignment (MSA) data. 
More details of the MaxEnt model can be found at [Enhancing computational enzyme design by a maximum entropy strategy
](https://www.pnas.org/doi/abs/10.1073/pnas.2122355119)

## 🔧 Installation

```bash
git clone https://github.com/EvoCatalysis/MaxEnt-Pytorch.git
cd MaxEnt-Pytorch
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
python train_maxent.py --device 'cuda' --n_epochs 1000
```

### 3. Predict Mutation Energies
Predict mutation energy changes compared to the wild-type based on the trained model.

```bash
python mut_energy_maxent_single_all.py
```

## 🧾 Files

| File                             | Description                                |
|----------------------------------|--------------------------------------------|
| `msa_lib.py`                     | Preprocess MSA file                        |
| `train_maxent.py`                | Train MaxEnt model on one-hot encoded MSA  |
| `mut_energy_maxent_single_all.py`| Compute mutation energies                  |
| `requirements.txt`               | Python dependencies                        |

## 📚 References
Please cite: Xie, W. J., Asadi, M., & Warshel, A. (2022). [Enhancing computational enzyme design by a maximum entropy strategy
](https://www.pnas.org/doi/abs/10.1073/pnas.2122355119). *Proceedings of the National Academy of Sciences USA*, 119(7), e2122355119.
