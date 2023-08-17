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
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from nettcr_archs import NetTCR_CDR123, NetTCR_CDR123_singlechain


import utils
import time, random

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
def make_tensor_ds(df, encoding, pep_max_len=30, cdr1_max_len=10, cdr2_max_len=20, cdr3_max_len=20, device='cpu'):
    encoded_pep = torch.tensor(utils.enc_list_bl_max_len(df.peptide, encoding, pep_max_len)/5, dtype=float)
    encoded_a1 = torch.tensor(utils.enc_list_bl_max_len(df.A1, encoding, cdr1_max_len)/5, dtype=float)
    encoded_a2 = torch.tensor(utils.enc_list_bl_max_len(df.A2, encoding, cdr2_max_len)/5, dtype=float)
    encoded_a3 = torch.tensor(utils.enc_list_bl_max_len(df.A3, encoding, cdr3_max_len)/5, dtype=float)
    encoded_b1 = torch.tensor(utils.enc_list_bl_max_len(df.B1, encoding, cdr1_max_len)/5, dtype=float)
    encoded_b2 = torch.tensor(utils.enc_list_bl_max_len(df.B2, encoding, cdr2_max_len)/5, dtype=float)
    encoded_b3 = torch.tensor(utils.enc_list_bl_max_len(df.B3, encoding, cdr3_max_len)/5, dtype=float)
    targets = torch.tensor(df.binder.values, dtype=float)

    tensor_ds = TensorDataset(encoded_pep.float().to(device), 
                              encoded_a1.float().to(device), encoded_a2.float().to(device), encoded_a3.float().to(device), 
                              encoded_b1.float().to(device), encoded_b2.float().to(device), encoded_b3.float().to(device), 
                              torch.unsqueeze(targets.float().to(device), 1))

    return tensor_ds

def test(args):
    chain = args.trained_model.split('_')[-1][:-3]
    
    # Make sure peptide, A3, B3 columns are in the data
    x_test = pd.read_csv(args.test_data)
    assert "A1" and "A2" and "A3" in x_test.columns, "Make sure the test file contains all the CDRs"
    assert "B1" and "B2" and "B3" in x_test.columns, "Make sure the test file contains all the CDRs"
    assert 'peptide' in x_test.columns, "Couldn't find peptide in the test data"
    
    test_tensor = make_tensor_ds(x_test, encoding = utils.blosum50, device=device)
    
    # Load model
    net = torch.jit.load(args.trained_model, map_location=torch.device(device))
    net.to(device)
    
    pep_test = test_tensor[:][0]
    a1_test = test_tensor[:][1]
    a2_test = test_tensor[:][2]
    a3_test = test_tensor[:][3]
    b1_test = test_tensor[:][4]
    b2_test = test_tensor[:][5]
    b3_test = test_tensor[:][6]
    y_test = test_tensor[:][7]
    
    if args.chain == "ab":
        pred = net(pep_test, a1_test, a2_test, a3_test, b1_test, b2_test, b3_test)
    elif args.chain == "a":
        pred = net(pep_test, a1_test, a2_test, a3_test)
    elif args.chain == "b":
        pred = net(pep_test, b1_test, b2_test, b3_test)
        
    x_test['nettcr_prediction'] = pred.cpu().detach().numpy()
    
    x_test.to_csv(args.outdir+'/nettcr_preds_cdr123_'+chain+'.csv', index=False)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data")
    parser.add_argument("--trained_model")
    parser.add_argument("--outdir")
    parser.add_argument("--chain")
    parser.add_argument("--device", default='cpu')
    
    args = parser.parse_args()
    
    assert args.chain!=None, "Specify chain"
    assert args.chain in ["a","b","ab"], "Invalid chain"
    
    test(args)
