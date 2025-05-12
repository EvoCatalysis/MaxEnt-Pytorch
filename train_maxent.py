"""
- Train a MaxEnt model using MSA one-hot input.

Default Parameters:
  --device cuda
  --epoch 1000
"""

from msa_lib import *

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import argparse
import time

class MSADataset(Dataset):
    def __init__(self, msa_onehot, msa_eff):
        self.msa_onehot = msa_onehot
        self.msa_eff = msa_eff

    def __len__(self):
        return len(self.msa_onehot)

    def __getitem__(self, idx):
        return self.msa_onehot[idx], self.msa_eff[idx]


class MaxEntModel(nn.Module):
    '''pseudo-lihelihood approximation'''
    def __init__(self, num_aa_filted, states):
        super(MaxEntModel, self).__init__()
        self.num_aa_filted = num_aa_filted
        self.states = states
        self.h = nn.Parameter(torch.zeros(num_aa_filted*states), requires_grad=True)
        self.J = nn.Parameter(torch.zeros(num_aa_filted*states, num_aa_filted*states), requires_grad=True)

    def forward(self, msa_filted_onehot, msa_eff):
        #J = (self.J+self.J.T)/2*(1-torch.eye(self.num_aa_filted*self.states).to(self.J.device)) #symmetrized J
        J = (self.J+self.J.T)/2*(1-torch.eye(self.num_aa_filted*self.states).to(self.J)) #symmetrized J
        msa_filted_onehot = msa_filted_onehot.reshape(-1, self.num_aa_filted*self.states)
        H = -(self.h+torch.mm(msa_filted_onehot,J)).reshape(-1, self.num_aa_filted, self.states)
        pll = torch.sum(torch.sum(-msa_filted_onehot.reshape(-1,self.num_aa_filted, self.states)*F.log_softmax(H, -1), dim=(-2,-1))*msa_eff)
        L2_h = 0.01*torch.sum(self.h**2)
        L2_J = 0.01*torch.sum(J**2)*0.5*(self.num_aa_filted-1)*(self.states-1)
        return pll+L2_h+L2_J


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--device', type=str, help='device used to train the model')
    parser.add_argument('--msa_processed', type=str, default='msa_processed.pkl', help='processed MSA file path')
    parser.add_argument('--n_epochs', type=int, default='1000', help='number of epochs')

    args = parser.parse_args()

    with open(args.msa_processed,'rb') as f:
        msa_processed = pickle.load(f)
    
    msa_filted = msa_processed['msa_filted']
    msa_filted_onehot = np.eye(states)[msa_filted]
    msa_eff = msa_processed['msa_eff']
    num_aa_filted = len(msa_processed['non_gap'])
    
    msa_filted_onehot = torch.from_numpy(msa_filted_onehot).type(torch.FloatTensor)
    msa_eff = torch.from_numpy(msa_eff).type(torch.FloatTensor)
    
    device = torch.device(args.device)
    msa_filted_onehot = msa_filted_onehot.to(device)
    msa_eff = msa_eff.to(device)
    batch_size = 20000  # You can adjust this value as needed
    dataset = MSADataset(msa_filted_onehot, msa_eff)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = MaxEntModel(num_aa_filted, states).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    
    start_time = time.time()
    
    with open('loss.txt','w') as f:
        for epoch in range(args.n_epochs):
            for batch_onehot, batch_eff in data_loader:
                optimizer.zero_grad()
                loss = model(batch_onehot, batch_eff)
                loss.backward()
                optimizer.step()
        
            if epoch%1==0:
                print(f"Epoch {epoch}/{args.n_epochs}, Loss: {loss.item()}")
                f.write(f"{epoch} {loss.item():.6f}\n")
    
    print(f"time used: {time.time() - start_time:.2f}")

    h_opt = model.h.cpu().data.numpy()
    J_opt = model.J.cpu().data.numpy()
    J_opt = J_opt.reshape(num_aa_filted,states,num_aa_filted,states)
    
    tri = np.triu_indices(num_aa_filted,1)
    J_opt = J_opt[tri[0],:,tri[1],:]
    
    h_J = {'h':h_opt, 'J':J_opt}
    with open("h_J.pkl", "wb") as f:
        pickle.dump(h_J, f)
