#!/usr/bin/env python 

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define network architecture 
class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(in_channels=24, out_channels=16, kernel_size=1, padding='same')
        self.c3 = nn.Conv1d(in_channels=24, out_channels=16, kernel_size=3, padding='same')
        self.c5 = nn.Conv1d(in_channels=24, out_channels=16, kernel_size=5, padding='same')
        self.c7 = nn.Conv1d(in_channels=24, out_channels=16, kernel_size=7, padding='same')
        self.c9 = nn.Conv1d(in_channels=24, out_channels=16, kernel_size=9, padding='same')

        self.activation = nn.Sigmoid()

    def forward(self, x):
        c1 = torch.max(self.activation(self.c1(x)), 2)[0]
        c3 = torch.max(self.activation(self.c3(x)), 2)[0]
        c5 = torch.max(self.activation(self.c5(x)), 2)[0]
        c7 = torch.max(self.activation(self.c7(x)), 2)[0]
        c9 = torch.max(self.activation(self.c9(x)), 2)[0]

        out = torch.cat((c1, c3, c5, c7, c9), 1)

        return out
    
class NetTCR_CDR3(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = ConvBlock()

        self.linear = nn.Linear(in_features=80*3, out_features=32)
        self.activation = nn.Sigmoid()
        self.out = nn.Linear(32, 1)

    def forward(self, pep, a3, b3):
        # Transpose tensors
        pep = torch.permute(pep, (0, 2, 1))
        a3 = torch.permute(a3, (0, 2, 1))
        b3 = torch.permute(b3, (0, 2, 1))
        
        pep_cnn = self.cnn(pep)
        a3_cnn = self.cnn(a3)
        b3_cnn = self.cnn(b3)

        cat = torch.cat((pep_cnn, a3_cnn, b3_cnn),1)

        hid = self.activation(self.linear(cat))
        out = self.activation(self.out(hid))

        return out
    
class NetTCR_CDR3_singlechain(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = ConvBlock()

        self.linear = nn.Linear(in_features=80*2, out_features=32)
        self.activation = nn.Sigmoid()
        self.out = nn.Linear(32, 1)

    def forward(self, pep, cdr):
        # Transpose tensors
        pep = torch.permute(pep, (0, 2, 1))
        cdr = torch.permute(cdr, (0, 2, 1))
        
        pep_cnn = self.cnn(pep)
        cdr_cnn = self.cnn(cdr)

        cat = torch.cat((pep_cnn, cdr_cnn),1)

        hid = self.activation(self.linear(cat))
        out = self.activation(self.out(hid))

        return out
    
class NetTCR_CDR123(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = ConvBlock()

        self.linear = nn.Linear(in_features=80*7, out_features=32)
        self.activation = nn.Sigmoid()
        self.out = nn.Linear(32, 1)

    def forward(self, pep, a1, a2, a3, b1, b2, b3):
        # Transpose tensors
        pep = torch.permute(pep, (0, 2, 1))
        a1 = torch.permute(a1, (0, 2, 1))
        a2 = torch.permute(a2, (0, 2, 1))
        a3 = torch.permute(a3, (0, 2, 1))
        b1 = torch.permute(b1, (0, 2, 1))
        b2 = torch.permute(b2, (0, 2, 1))
        b3 = torch.permute(b3, (0, 2, 1))
        
        pep_cnn = self.cnn(pep)
        a1_cnn = self.cnn(a1)
        a2_cnn = self.cnn(a2)
        a3_cnn = self.cnn(a3)
        b1_cnn = self.cnn(b1)
        b2_cnn = self.cnn(b2)
        b3_cnn = self.cnn(b3)

        cat = torch.cat((pep_cnn, a1_cnn, a2_cnn, a3_cnn, b1_cnn, b2_cnn, b3_cnn), 1)

        hid = self.activation(self.linear(cat))
        out = self.activation(self.out(hid))

        return out

    
class NetTCR_CDR123_singlechain(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = ConvBlock()

        self.linear = nn.Linear(in_features=80*4, out_features=32)
        self.activation = nn.Sigmoid()
        self.out = nn.Linear(32, 1)

    def forward(self, pep, cdr1, cdr2, cdr3):
        # Transpose tensors
        pep = torch.permute(pep, (0, 2, 1))
        cdr1 = torch.permute(cdr1, (0, 2, 1))
        cdr2 = torch.permute(cdr2, (0, 2, 1))
        cdr3 = torch.permute(cdr3, (0, 2, 1))
        
        pep_cnn = self.cnn(pep)
        cdr1_cnn = self.cnn(cdr1)
        cdr2_cnn = self.cnn(cdr2)
        cdr3_cnn = self.cnn(cdr3)

        cat = torch.cat((pep_cnn, cdr1_cnn, cdr2_cnn, cdr3_cnn), 1)

        hid = self.activation(self.linear(cat))
        out = self.activation(self.out(hid))

        return out
    
    