#!/usr/bin/env python 

import torch
if torch.cuda.is_available():    
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name())
    device = "cuda:0"
else:
    print('No GPU available, using the CPU instead.')
    device='cpu'

import os
import numpy as np
import pandas as pd 
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from nettcr_archs import NetTCR_CDR123_pep

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

def train(args):
    # Create output directory if it doesn't exist
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # Make sure peptide, A3, B3 columns are in the data
    x_train = pd.read_csv(args.train_data)
    assert "A1" and "A2" and "A3" in x_train.columns, "Make sure the training file contains all the CDRs"
    assert "B1" and "B2" and "B3" in x_train.columns, "Make sure the training file contains all the CDRs"
    assert 'peptide' in x_train.columns, "Couldn't find peptide in the training data"
    
    train_tensor = make_tensor_ds(x_train, encoding = utils.blosum50, device=device)
    train_loader = DataLoader(train_tensor, batch_size=args.batch_size)

    x_valid = pd.read_csv(args.val_data)
    assert "A1" and "A2" and "A3" in x_valid.columns, "Make sure the validation file contains all the CDRs"
    assert "B1" and "B2" and "B3" in x_valid.columns, "Make sure the validation file contains all the CDRs"
    assert 'peptide' in x_valid.columns, "Couldn't find peptide in the validation data"
    valid_tensor = make_tensor_ds(x_valid, encoding = utils.blosum50, device=device)
    valid_loader = DataLoader(valid_tensor, batch_size=args.batch_size)

    # Init the neural network
    net = NetTCR_CDR123_pep()
    net.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    early_stopping = utils.EarlyStopping(patience=20, 
                                   path=args.outdir+'/trained_model_cdr123_ab.pt',
                                   print_count=False)

    train_loss, valid_loss = [], []

    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        batch_loss = 0
        net.train()
        for a1_train, a2_train, a3_train, \
            b1_train, b2_train, b3_train, y_train in train_loader:

            net.zero_grad()
            output = net(a1_train, a2_train, a3_train, b1_train, b2_train, b3_train)

            loss = criterion(output, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
        train_loss.append(batch_loss/len(train_loader))

        val_batch_loss = 0
        net.eval()
        for a1_valid, a2_valid, a3_valid, \
            b1_valid, b2_valid, b3_valid, y_valid in valid_loader:

            pred = net(a1_valid, a2_valid, a3_valid, b1_valid, b2_valid, b3_valid)


            loss = criterion(pred, y_valid)
            val_batch_loss += loss.data
        valid_loss.append(val_batch_loss/len(valid_loader))

        if args.verbose:
            print('Epoch: %d\t Train Loss: %.6f\t valid Loss: %.6f\t Time: %.2f s'%(epoch, train_loss[-1], 
                                                                                     valid_loss[-1], 
                                                                                     time.time()-\
                                                                                     start_epoch_time))
        # Ealy stopping

        if utils.invoke(early_stopping, valid_loss[-1], net, implement=True)==True:
            print("Early stopping")
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data")
    parser.add_argument("--val_data")
    parser.add_argument("--outdir")
    parser.add_argument("--device", default='cpu')
    parser.add_argument("--learning_rate", "-lr", default=0.001)
    parser.add_argument("--batch_size", "-bs", default=64)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--verbose", default=1)
    
    args = parser.parse_args()
        
    train(args)
    

    
