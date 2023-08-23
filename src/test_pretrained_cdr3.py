#!/usr/bin/env python 

import torch
if torch.cuda.is_available():    
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name())
else:
    print('No GPU available, using the CPU instead.')
device = "cuda:0" if torch.cuda.is_available() else "cpu"

import numpy as np
import pandas as pd 
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from nettcr_archs import NetTCR_CDR3_pep, ConvBlock

import utils
import random

# Set random seed
seed=15
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Define network
class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(in_channels=20, out_channels=16, kernel_size=1, padding='same')
        self.c3 = nn.Conv1d(in_channels=20, out_channels=16, kernel_size=3, padding='same')
        self.c5 = nn.Conv1d(in_channels=20, out_channels=16, kernel_size=5, padding='same')
        self.c7 = nn.Conv1d(in_channels=20, out_channels=16, kernel_size=7, padding='same')
        self.c9 = nn.Conv1d(in_channels=20, out_channels=16, kernel_size=9, padding='same')

        self.activation = nn.Sigmoid()

    def forward(self, x):
        c1 = torch.max(self.activation(self.c1(x)), 2)[0]
        c3 = torch.max(self.activation(self.c3(x)), 2)[0]
        c5 = torch.max(self.activation(self.c5(x)), 2)[0]
        c7 = torch.max(self.activation(self.c7(x)), 2)[0]
        c9 = torch.max(self.activation(self.c9(x)), 2)[0]

        out = torch.cat((c1, c3, c5, c7, c9), 1)

        return out
    
class CatConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a3, b3):
        out = torch.cat((a3, b3), 1)

        return out

class NetTCR_CDR3_pep(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = ConvBlock()

        self.linear = nn.Linear(in_features=80*2, out_features=32)
        self.activation = nn.Sigmoid()
        self.out = nn.Linear(32, 1)
        self.cat_conv = CatConv()

    def forward(self, a3, b3):
        # Transpose tensors
        a3 = torch.permute(a3, (0, 2, 1))

        b3 = torch.permute(b3, (0, 2, 1))
        

        a3_cnn = self.cnn(a3)
        b3_cnn = self.cnn(b3)

        cat = self.cat_conv(a3_cnn, b3_cnn)

        hid = self.activation(self.linear(cat))
        out = self.activation(self.out(hid))

        return out

# Define function to make the dataset
def make_tensor_ds(df, encoding, cdr_max_len=30, device='cpu'):
    encoded_a3 = torch.tensor(utils.enc_list_bl_max_len(df.A3, encoding, cdr_max_len)/5, dtype=float)
    encoded_b3 = torch.tensor(utils.enc_list_bl_max_len(df.B3, encoding, cdr_max_len)/5, dtype=float)
    targets = torch.tensor(df.binder.values, dtype=float)

    tensor_ds = TensorDataset(encoded_a3.float().to(device),
                              encoded_b3.float().to(device),
                              torch.unsqueeze(targets.float().to(device), 1))

    return tensor_ds

def test(args):    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # Make sure peptide, A3, B3 columns are in the data
    x_test = pd.read_csv(args.test_data)
    assert 'A3' in x_test.columns, "Couldn't find A3 in the data"
    assert 'B3' in x_test.columns, "Couldn't find B3 in the data"
    assert 'peptide' in x_test.columns, "Couldn't find peptide in the data"
    
    test_tensor = make_tensor_ds(x_test, encoding = utils.blosum50, device=device)
    
    a3_test = test_tensor[:][0]
    b3_test = test_tensor[:][1]
    
    partitions = {1, 2, 3, 4, 5}
    preds = 0
    for t in partitions:
        for v in partitions:
            if t!=v:
                net = torch.load(f'{args.trained_models_dir}/checkpoint_t.{str(t)}.v.{str(v)}.pt',
                                 map_location=torch.device('cpu'))
                net.eval()
                preds += net(a3_test, b3_test)
        
    x_test['nettcr_prediction'] = preds.cpu().detach().numpy()/20
    
    x_test.to_csv(args.outdir+'/pretrained_nettcr_preds_cdr3_ab.csv', index=False)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data")
    parser.add_argument("--trained_models_dir")
    parser.add_argument("--outdir")
    parser.add_argument("--device", default='cpu')
    
    args = parser.parse_args()

    test(args)
