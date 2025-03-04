import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from dataset import dataset as dataset
from TransUnetBeam import TransUnetBeam as network
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import glob
import json
import argparse
import pytorch_ssim
from scipy.signal import resample

def main(learning_rate,epochs,ar,br,cr,BatchSize,cuda_num, Model_Saving_idx):
	#check the cuda available
	if torch.cuda.is_available() == True :
		print('GPU exist')
	else :
		print('No GPU')
		os.exit(0)
	#parameters :
	lr = learning_rate
	num_epochs = epochs
	batchsize = BatchSize
	a_loss=ar
	b_loss=br
	c_loss=cr
	Model_Saving_PATH = '../Results/Model_Path/' + Model_Saving_idx + '.pth'
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
	model = network().to(device)
	criterion1 = nn.MSELoss()
	criterion2 = pytorch_ssim.SSIM()
	criterion3 = torch.nn.KLDivLoss(reduction='batchmean',log_target=True)
	total_step1 = len(train_loader)
	total_step2 = len(vali_loader)
	optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
	#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=20,T_mult=1,eta_min=0.000001,last_epoch=-1)
	for epoch in range(num_epochs):
		loss_temp1 = 0
		loss_temp2 = 0
		Lloss1 = 0
		Lloss2 = 0
		Lloss3 = 0
		Lloss4 = 0
		Lloss10 = 0
		Lloss20 = 0
		Lloss30 = 0
		Lloss40 = 0
		for i,(RF0,RF1,RF2,RF3,w,BF,RF_c) in enumerate(train_loader):
			RF0 = RF0.to(device,dtype = torch.float)
			RF1 = RF1.to(device,dtype = torch.float)
			RF2 = RF2.to(device,dtype = torch.float)
			RF3 = RF3.to(device,dtype = torch.float)
			w = w.to(device,dtype = torch.float)
			RF_c = RF_c.to(device,dtype = torch.float)
			BF = BF.to(device,dtype = torch.float)
			output,y1,y2 = model(RF0,RF1,RF2,RF3,w,RF_c)
			#y2 = model(RF0,RF1,RF2,RF3,w,RF_c)
			#unsqueeze for ssim calculating
			loss1 = 1 - criterion2(BF,y1)
			loss2 = 1 - criterion2(BF,y2)
			loss3 = 1 - criterion2(BF,output)
			loss4 = criterion3(F.log_softmax(output.view(-1,1,370*400),dim=2),F.log_softmax(BF.view(-1,1,370*400),dim=2))
			loss = a_loss * loss1 + b_loss * loss2 + c_loss * loss3 + loss4
			loss_temp1 += loss
			Lloss1 += loss1
			Lloss2 += loss2
			Lloss3 += loss3
			Lloss4 += loss4
			optimizer.zero_grad()
			loss.backward()
			#nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1, norm_type=1)
			optimizer.step()
			#print('training yes')
		loss_avg_ep1 = loss_temp1/total_step1
		Lloss1 = Lloss1/total_step1
		Lloss2 = Lloss2/total_step1
		Lloss3 = Lloss3/total_step1
		Lloss4 = Lloss4/total_step1
		print('Train:Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}, loss1: {:.4f}, loss2: {:.4f}, loss3: {:.4f}, loss4: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step1, loss_avg_ep1,Lloss1,Lloss2,Lloss3,Lloss4))
		with torch.no_grad():
			for k,(RF0,RF1,RF2,RF3,w,BF,RF_c) in enumerate(vali_loader):
				RF0 = RF0.to(device,dtype = torch.float)
				RF1 = RF1.to(device,dtype = torch.float)
				RF2 = RF2.to(device,dtype = torch.float)
				RF3 = RF3.to(device,dtype = torch.float)
				w = w.to(device,dtype = torch.float)
				BF = BF.to(device,dtype = torch.float)
				RF_c = RF_c.to(device,dtype = torch.float)
				output,y1,y2 = model(RF0,RF1,RF2,RF3,w,RF_c)
				#y2 = model(RF0,RF1,RF2,RF3,w,RF_c)
				loss10 = 1 - criterion2(BF,y1)
				loss20 = 1 - criterion2(BF,y2)
				loss30 = 1 - criterion2(BF,output)
				loss40 = criterion3(F.log_softmax(output.view(-1,1,370*400),dim=2),F.log_softmax(BF.view(-1,1,370*400),dim=2))
				loss0 = a_loss * loss10 + b_loss * loss20 + c_loss * loss30 + loss40
				loss_temp2 += loss0
				Lloss10 += loss10
				Lloss20 += loss20
				Lloss30 += loss30
				Lloss40 += loss40
			loss_avg_ep2 = loss_temp2/total_step2
			Lloss10 = Lloss10/total_step2
			Lloss20 = Lloss20/total_step2
			Lloss30 = Lloss30/total_step2
			Lloss40 = Lloss40/total_step2
			print('Vali:Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}, loss1: {:.4f}, loss2: {:.4f}, loss3: {:.4f}, loss4: {:.4f}'.format(epoch+1, num_epochs, k+1, total_step2,loss_avg_ep2,Lloss10,Lloss20,Lloss30,Lloss40))
		scheduler.step()
	PATH = Model_Saving_PATH
	torch.save(model.state_dict(),PATH)
	print("Finish Training")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train a deep learning model.')
	parser.add_argument('--learning_rate', type=float, default=0.01)
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--ar', type=float,default=1.0)
	parser.add_argument('--br', type=float,default=1.0)
	parser.add_argument('--cr', type=float,default=1.0)
	parser.add_argument('--BatchSize', type=int, default=16)
	parser.add_argument('--cuda_num', type=str, default="cuda:0")
	parser.add_argument('--Model_Saving_idx', type=str, default="C1")
	args = parser.parse_args()
	main(args.learning_rate, args.epochs, args.ar, args.br,args.cr,args.BatchSize, args.cuda_num,args. Model_Saving_idx)

