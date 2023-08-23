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

    def forward(self, a1, a2, a3, b1, b2, b3):
        out = torch.cat((a1, a2, a3, b1, b2, b3), 1)

        return out

class NetTCR_CDR123_pep(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = ConvBlock()

#         self.linear = nn.LazyLinear(out_features=64)
        self.linear = nn.Linear(in_features=80*6, out_features=32)
        self.activation = nn.Sigmoid()
#         self.hid = nn.Linear(128,64)
        self.out = nn.Linear(32, 1)
        self.cat_conv = CatConv()

    def forward(self, a1, a2, a3, b1, b2, b3):
        # Transpose tensors
        a1 = torch.permute(a1, (0, 2, 1))
        a2 = torch.permute(a2, (0, 2, 1))
        a3 = torch.permute(a3, (0, 2, 1))
        b1 = torch.permute(b1, (0, 2, 1))
        b2 = torch.permute(b2, (0, 2, 1))
        b3 = torch.permute(b3, (0, 2, 1))
        
        a1_cnn = self.cnn(a1)
        a2_cnn = self.cnn(a2)
        a3_cnn = self.cnn(a3)
        b1_cnn = self.cnn(b1)
        b2_cnn = self.cnn(b2)
        b3_cnn = self.cnn(b3)

        cat = self.cat_conv(a1_cnn, a2_cnn, a3_cnn, b1_cnn, b2_cnn, b3_cnn)

        hid = self.activation(self.linear(cat))
        out = self.activation(self.out(hid))

        return out

# Define function to make the dataset
def make_tensor_ds(df, encoding, cdr1_max_len=10, cdr2_max_len=20, cdr3_max_len=20, device='cpu'):
    encoded_a1 = torch.tensor(utils.enc_list_bl_max_len(df.A1, encoding, cdr1_max_len)/5, dtype=float)
    encoded_a2 = torch.tensor(utils.enc_list_bl_max_len(df.A2, encoding, cdr2_max_len)/5, dtype=float)
    encoded_a3 = torch.tensor(utils.enc_list_bl_max_len(df.A3, encoding, cdr3_max_len)/5, dtype=float)
    encoded_b1 = torch.tensor(utils.enc_list_bl_max_len(df.B1, encoding, cdr1_max_len)/5, dtype=float)
    encoded_b2 = torch.tensor(utils.enc_list_bl_max_len(df.B2, encoding, cdr2_max_len)/5, dtype=float)
    encoded_b3 = torch.tensor(utils.enc_list_bl_max_len(df.B3, encoding, cdr3_max_len)/5, dtype=float)
    targets = torch.tensor(df.binder.values, dtype=float)

    tensor_ds = TensorDataset(encoded_a1.float().to(device), encoded_a2.float().to(device), encoded_a3.float().to(device), 
                              encoded_b1.float().to(device), encoded_b2.float().to(device), encoded_b3.float().to(device), 
                              torch.unsqueeze(targets.float().to(device), 1))

    return tensor_ds

def test(args):
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    # Make sure peptide, A3, B3 columns are in the data
    x_test = pd.read_csv(args.test_data)
    assert "A1" and "A2" and "A3" in x_test.columns, "Make sure the test file contains all the CDRs"
    assert "B1" and "B2" and "B3" in x_test.columns, "Make sure the test file contains all the CDRs"
    assert 'peptide' in x_test.columns, "Couldn't find peptide in the test data"
    
    test_tensor = make_tensor_ds(x_test, encoding = utils.blosum50, device=device)
    
    a1_test = test_tensor[:][0]
    a2_test = test_tensor[:][1]
    a3_test = test_tensor[:][2]
    b1_test = test_tensor[:][3]
    b2_test = test_tensor[:][4]
    b3_test = test_tensor[:][5]

    partitions = {1, 2, 3, 4, 5}
    preds = 0
    for t in partitions:
        for v in partitions:
            if t!=v:
                net = torch.load(f'{args.trained_models_dir}/checkpoint_t.{str(t)}.v.{str(v)}.pt',
                                 map_location=torch.device('cpu'))
                net.eval()
                preds += net(a1_test, a2_test, a3_test, b1_test, b2_test, b3_test)
        
    x_test['nettcr_prediction'] = preds.cpu().detach().numpy()/20
    
    x_test.to_csv(args.outdir+'/pretrained_nettcr_preds_cdr123_ab.csv', index=False)
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data")
    parser.add_argument("--trained_models_dir")
    parser.add_argument("--outdir")
    parser.add_argument("--device", default='cpu')
    
    args = parser.parse_args()
    
    test(args)
