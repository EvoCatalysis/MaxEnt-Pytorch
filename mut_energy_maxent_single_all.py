"""
- Predict mutational energies from trained MaxEnt model.

Command-line Arguments:
  --msa <path>     Default: msa_processed.pkl
  --model <path>   Default: h_J.pkl
"""

import sys
from msa_lib import *
import pandas as pd

with open('msa_processed.pkl','rb') as f:
    msa_processed = pickle.load(f)
non_gap = msa_processed['non_gap']
initial = msa_processed['msa_filted'][0]

with open("h_J.pkl", "rb") as f:
    h_J = pickle.load(f)
h = h_J['h']
J = h_J['J']

states = len(h) // len(non_gap) #21
h = h.reshape(-1, states)
triu = np.triu_indices(len(non_gap), 1)
J_full = np.zeros((len(non_gap),states,len(non_gap),states))
J_full[triu[0],:,triu[1],:] = J
print ('J_full.shape', J_full.shape)

# Generate all possible single mutations
mutations = []
energies = []

energy_wt = sum(h[i][initial[i]] + sum(J_full[i][initial[i]][j][initial[j]] for j in range(i+1, len(non_gap))) for i in range(len(non_gap)))

for pos,aa in enumerate(initial):
    print (pos)
    for idx in range(states):
        mut = copy.deepcopy(initial)
        mut[pos] = idx
        mutation = ''.join([alphabet[aa], str(non_gap[pos]+1), alphabet[idx]])
        energy = sum(h[i][mut[i]] + sum(J_full[i][mut[i]][j][mut[j]] for j in range(i+1, len(non_gap))) for i in range(len(non_gap)))

        mutations.append(mutation)
        energies.append(energy-energy_wt)


# Construct the DataFrame using mutations and energies
df = pd.DataFrame({
    'mutant': mutations,
    'maxent': energies
})

# Save the DataFrame to a CSV file
df.to_csv('energy_all_single_mutations.csv', index=False)

