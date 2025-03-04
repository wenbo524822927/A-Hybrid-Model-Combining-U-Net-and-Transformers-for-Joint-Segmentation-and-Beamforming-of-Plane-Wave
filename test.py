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
import wandb
import pytorch_ssim

def test(cuda_num,Model_Saving_idx):
	#check whether the cuda is available
	if torch.cuda.is_available() == True :
		print('GPU exist')
	else :
		print('No GPU')
		os.exit(0)

	# Initialize
	Model_PATH = '../Results/Model_Path/' + Model_Saving_idx + '.pth'
	test_path_name = 'test_path.json'
	savepath = '../Results/Image_results/' + Model_Saving_idx + '/'
	with open(test_path_name,'r') as fc:
		test_set_list = json.load(fc)
	device = torch.device(cuda_num if torch.cuda.is_available() else 'cpu')
	model = network().to(device)
	model.load_state_dict(torch.load(Model_PATH))
	Weight_all = np.load('weight_all_x.npy')
	test_dataset = dataset(test_set_list,Weight_all)
	test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = 1,shuffle = False,num_workers = 8)
	print('data loader done')

	#test
	with torch.no_grad():
		for i,(RF0,RF1,RF2,RF3,w,BF,RF_c) in enumerate(test_loader):
			RF0 = RF0.to(device,dtype = torch.float)
			RF1 = RF1.to(device,dtype = torch.float)
			RF2 = RF2.to(device,dtype = torch.float)
			RF3 = RF3.to(device,dtype = torch.float)
			w = w.to(device,dtype = torch.float)
			BF = BF.to(device,dtype = torch.float)
			RF_c = RF_c.to(device,dtype = torch.float)
			output,y1,y2 = model(RF0,RF1,RF2,RF3,w,RF_c)
			Y1 = y1.cpu()
			Y1 = Y1.detach().numpy()
			Y2 = y2.cpu()
			Y2 = Y2.detach().numpy()
			output = output.cpu()
			output = output.detach().numpy()
			name = test_set_list[i].split('Simulation_data/')[1].split('.npz')[0]
			arr_name = savepath + name
			image_name = savepath + name + '.png'
			np.savez(arr_name, NNout=output[0,0,:,:], y1=Y1[0,0,:,:], y2=Y2[0,0,:,:])
			plt.imsave(image_name,output[0,0,:,:])
			print(i)
	print('Test done !')

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train a deep learning model.')
	parser.add_argument('--cuda_num', type=str, default="cuda:0")
	parser.add_argument('--Model_Saving_idx', type=str, default="C1")
	args = parser.parse_args()
	test(args.cuda_num, args.Model_Saving_idx)


