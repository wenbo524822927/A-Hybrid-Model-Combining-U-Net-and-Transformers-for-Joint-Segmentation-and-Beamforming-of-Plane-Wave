import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from scipy.signal import resample

class dataset(Dataset):
    def __init__(self, files, weight):
    # files is a list of mat files path
        self.files = files
        self.weight = weight

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        onepath = self.files[idx]
        data = np.load(onepath)
        RF_ori = data['RF_in']#RF_in for phantom, RF_ori for simulation data.
        weight1 = np.vstack((self.weight,np.zeros([3840-3804,128])))
        weight0 = self.split_data(weight1)
        weight0 = torch.from_numpy(weight0)
        RF0,RF1,RF2,RF3 = self.downsample_split_RF(RF_ori)
        RF_c = np.vstack((RF_ori,np.zeros([4094-3804,128])))
        RF_c = torch.from_numpy(RF_c)
        RF_c = torch.unsqueeze(RF_c,dim=0)
        RF0 = torch.from_numpy(RF0)
        RF1 = torch.from_numpy(RF1)
        RF2 = torch.from_numpy(RF2)
        RF3 = torch.from_numpy(RF3)
        #RF_ori0 = torch.unsqueeze(RF_ori0,dim = 0)
        BF_out = data['BF_out']
        BF_out0 = torch.from_numpy(BF_out)
        BF_out0 = torch.unsqueeze(BF_out0,dim = 0)
        #RF_out0 = torch.unsqueeze(RF_out0,dim = 0)
        return RF0,RF1,RF2,RF3,weight0,BF_out0,RF_c

    def comb_weight(self,x,w):
        # x is channel data
        # w is delay weight of one line
        # np.stack((x,w))
        # np.concatenate((x,w),axis=0)
        return np.concatenate((x,w),axis=0)

    def split_data(self,x):
        return x.reshape((30,128,128))

    def downsample_split_RF(self,x):
        x = np.vstack((x,np.zeros([3840-3804,128])))
        x1 = resample(x,num=15*128,axis=0)
        x2 = resample(x,num=8*128,axis=0)
        x3 = resample(x,num=4*128,axis=0)
        return x.reshape((30,128,128)),x1.reshape((15,128,128)),x2.reshape((8,128,128)),x3.reshape((4,128,128))
