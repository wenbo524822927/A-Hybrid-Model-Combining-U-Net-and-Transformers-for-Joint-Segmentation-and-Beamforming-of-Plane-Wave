import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.io import loadmat as loadmat
import torch.nn.functional as F
from Attention_Gate_block import Attention_Gate_block as AGB
from MultiHeadAttention import MultiHeadAttention as MHA
from BeamUNet import BeamUNet as BUN

class TransUnetBeam(nn.Module):
	def __init__(self):
		super(TransUnetBeam,self).__init__()
		self.BeamUNet = BUN()
		self.MultiHeadAttention = MHA(seq_len=128,first_num=4094,heads=370,k_num=400,stride=10)
	def forward(self,x0,x1,x2,x3,w,x_c):
		y1 = self.BeamUNet(x0,x1,x2,x3,w)
		y2 = self.MultiHeadAttention(x_c,x_c,x_c)
		y = y1*y2
		return y,y1,y2
