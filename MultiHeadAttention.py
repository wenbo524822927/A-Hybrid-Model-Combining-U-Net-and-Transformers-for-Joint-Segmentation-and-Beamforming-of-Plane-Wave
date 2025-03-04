import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat as loadmat
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, seq_len, first_num,heads, k_num, stride):
        super().__init__()
        self.seq_len = seq_len
        self.heads = heads
        self.k_num = k_num
        self.f_num = first_num
        self.st = stride
        self.k_Linear = nn.Linear(self.f_num,self.f_num)
        self.q_Linear = nn.Linear(self.f_num,self.f_num)
        self.v_Linear = nn.Linear(self.f_num,self.f_num)
        self.para_linear1 = nn.Linear(seq_len*k_num,5000)
        self.para_linear2 = nn.Linear(5000,k_num)
        self.Dropout1 = nn.Dropout(p=0.5)
        self.BN_k = nn.BatchNorm2d(heads)
        self.BN_q = nn.BatchNorm2d(heads)
        self.BN_v = nn.BatchNorm2d(heads)
        self.BN1 = nn.BatchNorm2d(heads)
        self.unfold = nn.Unfold(kernel_size=(self.k_num, self.seq_len),dilation=1, stride=(self.st, 1))
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, q, k, v):
        bs = q.size(0)
        #Linear
        q = self.q_Linear(q.transpose(2,3)).transpose(2,3)
        k = self.k_Linear(k.transpose(2,3)).transpose(2,3)
        v = self.v_Linear(v.transpose(2,3)).transpose(2,3)
        #batch norm & unfold
        k = self.unfold(k).transpose(1,2)
        q = self.unfold(q).transpose(1,2)
        v = self.unfold(v).transpose(1,2)
        #
        k = self.BN_k(torch.stack(k.split(self.seq_len,dim=-1),dim=-1))
        q = self.BN_q(torch.stack(q.split(self.seq_len,dim=-1),dim=-1))
        v = self.BN_v(torch.stack(v.split(self.seq_len,dim=-1),dim=-1))
        # calculate attention
        scores = self.BN1(self.attention(q,k,v))
        # joint ParaLine
        output = self.ParaLine(scores)
        return output
        
    def attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(self.k_num)
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)
        return output

    def ParaLine(self, scores):
        output = self.Dropout1(self.leakyrelu(self.para_linear1(scores.flatten(start_dim=2,end_dim=-1)).unsqueeze(dim = 1)))
        output = self.sigmoid(self.para_linear2(output))
        return output
