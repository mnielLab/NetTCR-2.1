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

import torch
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
    # Make sure peptide, A3, B3 columns are in the data
    x_test = pd.read_csv(args.test_data)
    assert 'A3' in x_test.columns, "Couldn't find A3 in the data"
    assert 'B3' in x_test.columns, "Couldn't find B3 in the data"
    assert 'peptide' in x_test.columns, "Couldn't find peptide in the data"
    
    test_tensor = make_tensor_ds(x_test, encoding = utils.blosum50, device=device)
    
    # Load model
    net = torch.jit.load(args.trained_model, map_location=torch.device(device))
    net.to(device)
    
    a3_test = test_tensor[:][0]
    b3_test = test_tensor[:][1]
    
    pred = net(a3_test, b3_test)

        
    x_test['nettcr_prediction'] = pred.cpu().detach().numpy()
    
    x_test.to_csv(args.outdir+'/nettcr_preds_cdr3_ab.csv', index=False)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data")
    parser.add_argument("--trained_model")
    parser.add_argument("--outdir")
    parser.add_argument("--device", default='cpu')
    
    args = parser.parse_args()

    test(args)
