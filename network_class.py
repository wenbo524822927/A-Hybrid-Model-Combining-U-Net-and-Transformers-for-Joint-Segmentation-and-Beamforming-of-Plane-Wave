import os
import glob
import json
import argparse
import wandb
import numpy as np
import pytorch_ssim
import matplotlib.pyplot as plt
from scipy.signal import resample
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

class Attention_Gate_block(nn.Module):
	def __init__(self,in_cha_x):
		super(Attention_Gate_block,self).__init__()
		#in_cha_x is the channel number of input tensor x
		#
		self.in_cha_x = in_cha_x
		self.Conv1 = nn.Conv2d(in_cha_x,in_cha_x,(1,1))
		self.Conv2 = nn.Conv2d(in_cha_x,1,(1,1))
		self.LeakyReLU = nn.LeakyReLU()
		self.sig = nn.Sigmoid()
	def forward(self,x):
		x0 = self.Conv1(x)
		h = self.sig(self.Conv2(self.LeakyReLU(x0)))
		weight = h.expand(-1,self.in_cha_x,-1,-1)
		y = x * weight
		return y

def cbl_block(in_ch, out_ch, k=3, p=1, bias=False):
	return nn.Sequential(
		nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=bias),
		nn.BatchNorm2d(out_ch),
		nn.LeakyReLU()
		)


class BeamUNet(nn.Module):
	def __init__(self):
		super(BeamUNet,self).__init__()
		self.AGB0 = Attention_Gate_block(16)
		self.AGB1 = Attention_Gate_block(8)
		self.AGB2 = Attention_Gate_block(4)
		self.AGB3 = Attention_Gate_block(2)
		pool = nn.MaxPool2d(2)
		self.enblock_x3 = nn.Sequential(
			cbl(2, 16),
			pool,
			cbl(16, 16))
		self.enblock_x2 = nn.Sequential(
			cbl(4, 16),
			pool,
			cbl(16, 16))
		self.enblock_x1 = nn.Sequential(
			cbl(8, 32),
			pool,
			cbl(32,32),
			pool,
			cbl(32, 32))
		self.enblock_x0 = nn.Sequential(
			cbl(16, 32),
			pool,
			cbl(32, 64),
			pool,
			cbl(64, 64)
			pool,
			cbl(64, 64))
		self.enblock1 = nn.Sequential(
			cbl(32, 32),
			pool,
			cbl(32, 32))
		self.enblock2 = nn.Sequential(
			cbl(64, 64),
			pool,
			cbl(64, 64))
		self.enblock3 = nn.Sequential(
			cbl(128, 128),
			pool,
			cbl(128, 128))
		self.neck = nn.Sequential(
			cbl(128, 128),
			cbl(128, 128),
			cbl(128, 128))
		self.deblock1 = nn.Sequential(
			nn.Upsample(scale_factor=2),
			cbl(256, 128),
			cbl(128, 128))
		self.deblock2 = nn.Sequential(
			nn.Upsample(scale_factor=2)
			cbl(256, 64),
			cbl(64, 64))
		self.deblock3 = nn.Sequential(
			nn.Upsample(scale_factor=2),
			cbl(128, 32),
			cbl(32, 32))
		self.deblock4 = nn.Sequential(
			nn.Upsample(scale_factor=2),
			cbl(64, 16),
			cbl(16, 8))
		self.upblock1 = nn.Sequential(
			nn.Upsample(scale_factor=2),
			cbl(8, 8, p=0, bias=True),
			cbl(8, 8, bias=True)，
			cbl(8, 8, bias=True))
		self.upblock2 = nn.Sequential(
			nn.Upsample(scale_factor=1.98),
			cbl(8, 8, p=0, bias=True),
			cbl(8, 8, bias=True),
			cbl(8, 8, bias=True),
			cbl(8, 1, bias=True),
			cbl(1, 1, bias=True),
			cbl(1, 1, bias=True),
			nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
			nn.Sigmoid())

	def forward(self,x0,x1,x2,x3):
		x3 = self.AGB3(x3)
		x2 = self.AGB2(x2)
		x1 = self.AGB1(x1)
		x0 = self.AGB0(x0)
		h1 = torch.cat((self.enblock_x3(x3),self.enblock_x2(x2)),1)
		h2 = torch.cat((self.enblock1(h1),self.enblock_x1(x1)),1)
		h3 = torch.cat((self.enblock2(h2),self.enblock_x0(x0)),1)
		h4 = self.enblock3(h3)
		y1 = self.neck(h4)
		y2 = self.deblock1(torch.cat((y1,h4),1))
		y3 = self.deblock2(torch.cat((y2,h3),1))
		y4 = self.deblock3(torch.cat((y3,h2),1))
		y5 = self.deblock4(torch.cat((y4,h1),1))
		y6 = self.upSampleDecoder(y5)
		return y6

	def upSampleDecoder(self,x):
		y = self.upblock2(self.upblock1(x))
		return y

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
		self.para_linear2 = nn.Linear(5000,heads)
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

class TransUnetBeam(nn.Module):
	'''
	mode: 
	y1: UNet
	y2: Transformer
	both: all
	'''
	def __init__(self,mode):
		super(TransUnetBeam_t,self).__init__()
		self.BeamUNet = BUN()
		self.MultiHeadAttention = MHA(seq_len=128,first_num=4094,heads=370,k_num=400,stride=10)
		self.mode = mode
	def forward(self,x0,x1,x2,x3,w,x_c,mode: str = 'both'):
		mode = mode.lower()
		y1 = y2 = None
		if mode in ('y1', 'both'):
			y1 = self.BeamUNet(x0, x1, x2, x3, w)
		if mode in ('y2', 'product', 'both'):
			y2 = self.MultiHeadAttention(x_c, x_c, x_c)
		y2 = self.MultiHeadAttention(x_c,x_c,x_c)
		if mode == 'y1':
			return y1
		elif mode == 'y2':
			return y2
		elif mode == 'both':
			return y1,y2,y1*y2
		else : 
			raise ValueError(f"Unsupported mode: {mode}. "
							f"Choose from ['y1','y2','product','both'].")

