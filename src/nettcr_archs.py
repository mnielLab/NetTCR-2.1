#!/usr/bin/env python 

import torch
import torch.nn as nn

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
    
class CatConv123(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a1, a2, a3, b1, b2, b3):
        out = torch.cat((a1, a2, a3, b1, b2, b3), 1)

        return out
    
class CatConv3(nn.Module):
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
        self.cat_conv = CatConv3()

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
    

class NetTCR_CDR123_pep(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = ConvBlock()

        self.linear = nn.Linear(in_features=80*6, out_features=32)
        self.activation = nn.Sigmoid()
        self.out = nn.Linear(32, 1)
        self.cat_conv = CatConv123()

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