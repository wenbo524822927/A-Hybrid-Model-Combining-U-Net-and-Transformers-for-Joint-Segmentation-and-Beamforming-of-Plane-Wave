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

from network_class import TransUnetBeam as network
from dataset import dataset as dataset

def main(learning_rate,epochs,ar,br,cr,mode,BatchSize,cuda_num, Model_Saving_idx):
	#check the cuda available
	if torch.cuda.is_available() == True :
		print('GPU exist')
	else :
		print('No GPU')
		os.exit(0)
	#Initialize wandb
	#parameters :
	lr = learning_rate
	num_epochs = epochs
	batchsize = BatchSize
	a_loss=ar
	b_loss=br
	c_loss=cr
	mode_name = mode
	##
	train_path_name = 'train_path.json'
	vali_path_name = 'vali_path.json'
	test_path_name = 'test_path.json'
	Weight_all = np.load('weight_all_x.npy')
	with open(train_path_name,'r') as fa0:
		train_set_list = json.load(fa0)
	with open(vali_path_name,'r') as fb0:
		vali_set_list = json.load(fb0)
	device = torch.device(cuda_num if torch.cuda.is_available() else 'cpu')
	#setting train_dataset and test_dataset
	#train_set/validate_set/test_set
	train_dataset = dataset(train_set_list,Weight_all)
	vali_dataset = dataset(vali_set_list,Weight_all)
	#train_loader/validate_loader/test_loader
	train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = batchsize,shuffle = True,num_workers = 8,drop_last=True)
	vali_loader = torch.utils.data.DataLoader(dataset = vali_dataset,batch_size = 1,shuffle = True,num_workers = 8,drop_last=True)
	print('data loader done')
	#model
	model = network(mode_name).to(device)
	criterion1 = pytorch_ssim.SSIM()
	criterion2 = torch.nn.KLDivLoss(reduction='batchmean',log_target=True)
	optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
	scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=20,T_mult=1,eta_min=0.000001,last_epoch=-1)
	for epoch in range(num_epochs):
		for i,(RF0,RF1,RF2,RF3,w,BF,RF_c) in enumerate(train_loader):
			RF0 = RF0.to(device,dtype = torch.float)
			RF1 = RF1.to(device,dtype = torch.float)
			RF2 = RF2.to(device,dtype = torch.float)
			RF3 = RF3.to(device,dtype = torch.float)
			w = w.to(device,dtype = torch.float)
			RF_c = RF_c.to(device,dtype = torch.float)
			BF = BF.to(device,dtype = torch.float)
			y1,y2,y = model(RF0,RF1,RF2,RF3,w,RF_c)
			#unsqueeze for ssim calculating
			loss1 = 1 - criterion1(BF,y1)
			loss2 = 1 - criterion1(BF,y2)
			loss3 = 1 - criterion1(BF,output)
			loss4 = 10*criterion2(F.log_softmax(output.view(-1,1,370*400),dim=2),F.log_softmax(BF.view(-1,1,370*400),dim=2))
			loss = a_loss * loss1 + b_loss * loss2 + c_loss * loss3 + loss4
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		scheduler.step()
	PATH = "Model_Saving_PATH" # saving the model
	torch.save(model.state_dict(),PATH)
	print("Finish Training")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train a deep learning model.')
	parser.add_argument('--learning_rate', type=float, default=0.01)
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--ar', type=float,default=1.0)
	parser.add_argument('--br', type=float,default=1.0)
	parser.add_argument('--cr', type=float,default=1.0)
	parser.add_argument('mode', type=str,default='both')
	parser.add_argument('--BatchSize', type=int, default=16)
	parser.add_argument('--cuda_num', type=str, default="cuda:0")
	parser.add_argument('--Model_Saving_idx', type=str, default="C1")
	args = parser.parse_args()
	main(args.learning_rate, args.epochs, args.ar, args.br,args.cr,args.mode,args.BatchSize, args.cuda_num,args. Model_Saving_idx)

