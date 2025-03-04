import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.io import loadmat as loadmat
import torch.nn.functional as F

class Attention_Gate_block(nn.Module):
	def __init__(self,in_cha_x,in_cha_g):
		super(Attention_Gate_block,self).__init__()
		#in_cha_x is the channel number of input tensor x
		#in_cha_g is the channel number of weight tensor g
		#
		self.in_cha_x = in_cha_x
		self.in_cha_g = in_cha_g
		self.Conv1 = nn.Conv2d(in_cha_g,in_cha_x,(1,1))
		self.Conv2 = nn.Conv2d(in_cha_x,in_cha_x,(1,1))
		self.Conv3 = nn.Conv2d(in_cha_x,1,(1,1))
		self.LeakyReLU = nn.LeakyReLU()
		self.sig = nn.Sigmoid()
	def forward(self,g,x):
		g0 = self.Conv1(g)
		x0 = self.Conv2(x)
		h = self.sig(self.Conv3(self.LeakyReLU(g0 + x0)))
		weight = h.expand(-1,self.in_cha_x,-1,-1)
		y = x * weight
		return y
