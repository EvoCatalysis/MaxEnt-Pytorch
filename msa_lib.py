"""
- Process MSA into one-hot encoding and effective sequence weights.

Command-line Arguments:
  --msa <path>         Input MSA file in FASTA format
  --gap_cutoff <float> Threshold for filtering gapped positions (default: 0.7)
  --output <path>      Output pickle file path
"""

import numpy as np
from scipy.spatial.distance import pdist,squareform
from Bio import SeqIO, AlignIO
import pandas as pd
import pickle
import argparse

alphabet = 'ACDEFGHIKLMNPQRSTVWY-'
states = len(alphabet)

a2n = {}
for a,n in zip(alphabet,range(states)):
    a2n[a] = n

def aa2num(aa):
    '''convert amino acid into num'''
    if aa in a2n: return a2n[aa]
    else: return a2n['-']

def fasta2mat(fasta_map):
    '''convert MSA into matrix'''
    sequence = []
    for value in fasta_map.values():
        sequence.append([aa2num(i) for i in value])
    return np.array(sequence)

def get_eff_chunk(msa, eff_cutoff=0.8, chunk_size=20000):
    nseq = msa.shape[0]
    ncol = msa.shape[1]

    # Initialize the sequence weight matrix to zeros
    msa_w = np.zeros((nseq, nseq))

    # Compute pairwise distances in chunks
    for i in range(0, nseq, chunk_size):
        for j in range(i, nseq, chunk_size):
            print (i,j)
            start_i = i
            end_i = min(i + chunk_size, nseq)

            start_j = j
            end_j = min(j + chunk_size, nseq)

            chunk1 = msa[start_i:end_i]
            chunk2 = msa[start_j:end_j]

            if start_i == start_j:
                d = 1.0 - squareform(pdist(chunk1, 'hamming'))
                msa_w[start_i:end_i, start_i:end_i] = d
            else:
                d = 1.0 - np.array([[np.sum(a != b) / ncol for b in chunk2] for a in chunk1])
                msa_w[start_i:end_i, start_j:end_j] = d
                msa_w[start_j:end_j, start_i:end_i] = d.T

    msa_w = np.where(msa_w >= eff_cutoff, 1, 0)
    msa_w = 1 / np.sum(msa_w, -1)

    return msa_w


def filt_gap(msa, gap_cutoff=0.3):
    '''remove gap sites'''
    tmp = np.zeros_like(msa)
    tmp[np.where(msa == 20)] = 1
    non_gap = np.where(np.sum(tmp,0)/tmp.shape[0] < gap_cutoff)[0]
    return msa[:,non_gap], non_gap

def read_fasta_file(fasta_file):
    records = {}

    with open(fasta_file, 'r') as file:
        # Initialize empty sequence and ID variables
        sequence = ''
        seq_id = ''

        # Loop over each line in the file
        for line in file:
            # If line is a header line
            if line.startswith('>'):
                # If we have a sequence already, add it to the records list
                if sequence:
                    records[seq_id] = sequence.upper()

                # Get the sequence ID and reset the sequence variable
                seq_id = line.strip()[1:]
                sequence = ''
            else:
                # Append the line to the current sequence
                sequence += line.strip()

        # Add the last sequence to the records list
        if sequence:
            records[seq_id] = sequence.upper()

    return records

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process MSA')
    parser.add_argument('--msa', type=str, required=True, help='MSA file path')
    parser.add_argument('--gap_cutoff', type=float, default='0.7', help='criteria to filt gap')
    parser.add_argument('--output', type=str, default='msa_processed.pkl', help='Output file path')

    args = parser.parse_args()
    print ('gap_cutoff', args.gap_cutoff)

    fasta_map = read_fasta_file(args.msa)
    print ('number of MSA:', len(fasta_map))
    
    #process MSA
    msa = fasta2mat(fasta_map)
    num_seq, num_aa_msa = msa.shape
    msa_filted, non_gap = filt_gap(msa, args.gap_cutoff)
    msa_eff = get_eff_chunk(msa_filted)       #used to down-weight similar sequences
    msa_filted_onehot = np.eye(states)[msa_filted]
    _, num_aa_filted = msa_filted.shape
    print ('number of non_gap sites:', len(non_gap))
    
    msa_processed = {'gap_cutoff':args.gap_cutoff, 'non_gap':non_gap, 'msa_eff':msa_eff, 'msa_filted':msa_filted}
    with open(args.output, 'wb') as f:
        pickle.dump(msa_processed, f)
