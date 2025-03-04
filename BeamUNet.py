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

class BeamUNet(nn.Module):
	def __init__(self):
		super(BeamUNet,self).__init__()
		self.AGB3 = AGB(4,30)
		self.AGB2 = AGB(8,30)
		self.AGB1 = AGB(15,30)
		self.AGB0 = AGB(30,30)
		self.enblock1 = nn.Sequential(
			nn.Conv2d(4,8,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(8),
			nn.LeakyReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(8,8,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(8),
			nn.LeakyReLU(),
			nn.Conv2d(8,8,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(8),
			nn.LeakyReLU())
		self.enblock2 = nn.Sequential(
			nn.Conv2d(16,16,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(16),
			nn.LeakyReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(16,16,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(16),
			nn.LeakyReLU(),
			nn.Conv2d(16,16,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(16),
			nn.LeakyReLU())
		self.enblock3 = nn.Sequential(
			nn.Conv2d(32,32,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(32,32,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(),
			nn.Conv2d(32,32,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(32),
			nn.LeakyReLU())
		self.enblock4 = nn.Sequential(
			nn.Conv2d(64,128,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(128,128,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(),
			nn.Conv2d(128,128,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU())
		self.enblock2x = nn.Sequential(
			nn.Conv2d(8,8,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(8),
			nn.LeakyReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(8,8,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(8),
			nn.LeakyReLU(),
			nn.Conv2d(8,8,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(8),
			nn.LeakyReLU())
		self.enblock1x = nn.Sequential(
			nn.Conv2d(15,16,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(16),
			nn.LeakyReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(16,16,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(16),
			nn.LeakyReLU(),
			nn.Conv2d(16,16,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(16),
			nn.LeakyReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(16,16,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(16),
			nn.LeakyReLU())
		self.enblock0x = nn.Sequential(
			nn.Conv2d(30,32,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(32,32,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(32,32,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(32,32,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(32),
			nn.LeakyReLU())
		self.neck = nn.Sequential(
			nn.Conv2d(128,128,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(),
			nn.Conv2d(128,128,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(),
			nn.Conv2d(128,128,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(),
			nn.Conv2d(128,128,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU())
		self.deblock1 = nn.Sequential(
			nn.Upsample(scale_factor=2),
			nn.Conv2d(256,64,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(),
			nn.Conv2d(64,64,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(),
			nn.Conv2d(64,64,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(64),
			nn.LeakyReLU())
		self.deblock2 = nn.Sequential(
			nn.Upsample(scale_factor=2),
			nn.Conv2d(128,128,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(),
			nn.Conv2d(128,32,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(),
			nn.Conv2d(32,32,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(32),
			nn.LeakyReLU())
		self.deblock3 = nn.Sequential(
			nn.Upsample(scale_factor=2),
			nn.Conv2d(64,64,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(),
			nn.Conv2d(64,16,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(16),
			nn.LeakyReLU(),
			nn.Conv2d(16,16,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(16),
			nn.LeakyReLU())
		self.deblock4 = nn.Sequential(
			nn.Upsample(scale_factor=2),
			nn.Conv2d(32,32,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(),
			nn.Conv2d(32,32,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(),
			nn.Conv2d(32,16,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(16),
			nn.LeakyReLU(),
			nn.Conv2d(16,16,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(16),
			nn.LeakyReLU(),
			nn.Conv2d(16,16,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(16),
			nn.LeakyReLU())
		self.deblock5 = nn.Sequential(
			nn.Conv2d(16,16,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(16),
			nn.LeakyReLU(),
			nn.Conv2d(16,16,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(16),
			nn.LeakyReLU(),
			nn.Conv2d(16,8,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(8),
			nn.LeakyReLU(),
			nn.Conv2d(8,8,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(8),
			nn.LeakyReLU(),
			nn.Conv2d(8,8,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(8),
			nn.LeakyReLU())
		self.deblock6_1 = nn.Sequential(
			nn.Upsample(scale_factor=(2,2.5)),
			nn.Conv2d(8,8,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(8),
			nn.LeakyReLU(),
			nn.Conv2d(8,8,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(8),
			nn.LeakyReLU())
		self.deblock6_2 = nn.Sequential(
			nn.Conv2d(8,8,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(8),
			nn.LeakyReLU(),
			nn.Conv2d(8,8,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(8),
			nn.LeakyReLU(),
			nn.Conv2d(8,8,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(8),
			nn.LeakyReLU())
		self.deblock7_1 = nn.Sequential(
			nn.Upsample(scale_factor=(1.44,1.25)),
			nn.Conv2d(8,8,(3,3),padding=(2,1),bias=False),
			nn.BatchNorm2d(8),
			nn.LeakyReLU(),
			nn.Conv2d(8,8,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(8),
			nn.LeakyReLU())
		self.deblock7_2 = nn.Sequential(
			nn.Conv2d(8,8,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(8),
			nn.LeakyReLU(),
			nn.Conv2d(8,8,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(8),
			nn.LeakyReLU(),
			nn.Conv2d(8,8,(3,3),padding=1,bias=False),
			nn.BatchNorm2d(8),
			nn.LeakyReLU())      
		self.deblock8 = nn.Sequential(
			nn.Conv2d(8,1,(3,3),padding=1),
			nn.BatchNorm2d(1),
			nn.LeakyReLU(),
			nn.Conv2d(1,1,(3,3),padding=1),
			nn.Sigmoid())
	def forward(self,x0,x1,x2,x3,w):
		x3 = self.AGB3(w,x3)
		x2 = self.AGB2(w,x2)
		x1 = self.AGB1(w,x1)
		x0 = self.AGB0(w,x0)
		h1 = torch.cat((self.enblock1(x3),self.enblock2x(x2)),1)
		h2 = torch.cat((self.enblock2(h1),self.enblock1x(x1)),1)
		h3 = torch.cat((self.enblock3(h2),self.enblock0x(x0)),1)
		h4 = self.enblock4(h3)
		y1 = self.neck(h4)
		y2 = self.deblock1(torch.cat((y1,h4),1))
		y3 = self.deblock2(torch.cat((y2,h3),1))
		y4 = self.deblock3(torch.cat((y3,h2),1))
		y5 = self.deblock4(torch.cat((y4,h1),1))
		y6 = self.upSampleDecoder(self.deblock5(y5))
		return y6
	def upSampleDecoder(self, x):
		h = self.deblock6_1(x)
		h1 = self.deblock7_1(self.deblock6_2(h) + h)
		h2 = self.deblock7_2(h1) + h1
		y = self.deblock8(h2)
		return y

    


