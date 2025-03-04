# USTF: Ultrasound Test Function
# Author : Bo Wen

import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
from statistics import mean
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d
from scipy import ndimage
#
def name2CXZR(file_name):
    file_clip = file_name.split('_')
    x = float(file_clip[1])
    z = float(file_clip[2])
    r = float(file_clip[3])
    return x,z,r

def GT_mask(x,z,r):
	#create ground truth mask
	pos_x = np.linspace(-19,19,400)
	pos_z = np.linspace(5,70,370)
	gt = np.zeros([370,400])
	for i in range(400):
		for k in range(370):
			d = math.sqrt((pos_x[i] - x)**2 + (pos_z[k] - z)**2)
			if d >= r:
				gt[k,i] = 1.0
	return gt

def DICE(mask,gt,t): 
	mask_x = np.where(mask > t, 1.0, 0.0)
	intersection = np.sum(mask_x * gt)
	size_img1 = np.sum(mask_x)
	size_img2 = np.sum(gt)
	F1 = (2 * intersection) / (size_img1 + size_img2)
	return F1



def xzr2in_and_out(x,z,r,arr):
    pos_x = np.linspace(-19,19,400)
    pos_z = np.linspace(5,70,370)
    inside = arr.copy()
    outside = arr.copy()
    for i in range(400):
        for k in range(370):
            d = math.sqrt((pos_x[i] - x)**2 + (pos_z[k] - z)**2)
            if d <= r :
                outside[k,i] = None
            else :
                inside[k,i] = None
    inside_x = inside[~np.isnan(inside)]
    outside_x = outside[~np.isnan(outside)]
    return inside_x,outside_x

#MSE
def MSE(X,Y):
	return np.mean((X - Y)**2)

#SSIM
def SSIM(X,Y):
	return ssim(X,Y,win_size=11)

#Contrast
def Contrast(inside,outside):
	Si = mean(inside)
	So = mean(outside)
	return 20*math.log10(Si/So)
#SNR

def SNR(inside,outside):
    So = np.mean(outside)
    tho_o = np.std(outside)
    return So/tho_o
#PMF
def PMF(X):
	hist, bins = np.histogram(X, bins=256,range=(0,1),density=True)
	pmf = hist/np.sum(hist)
	return pmf,bins

def gCNR(inside,outside):
    # Calculate mean and variance for each region
    bins=100
    hist1, _ = np.histogram(inside, bins=bins, range=(0,1))
    pmf1 = hist1 / np.sum(hist1)
    hist2, _ = np.histogram(outside, bins=bins, range=(0,1))
    pmf2 = hist2 / np.sum(hist2)
    pmf_min = []
    for i in range(100):
        pmf_min.append(min(pmf1[i],pmf2[i]))
    gCNR = 1 - sum(pmf_min)
    return gCNR  # Ensure gCNR is between 0 and 1

def NCC(X,Y):
	X_mean = np.mean(X)
	Y_mean = np.mean(Y)
	X_std = np.std(X)
	Y_std = np.std(Y)
	X_norm = X - X_mean
	Y_norm = Y - Y_mean
	cross_corr = correlate2d(X_norm,Y_norm,mode='same')
	return cross_corr / (X_std*Y_std*X.size)

def MI(image1,image2):
	# Flatten the images to 1D arrays
	image1_flat = image1.flatten()
	image2_flat = image2.flatten()

	# Calculate the joint histogram
	joint_hist, _, _ = np.histogram2d(image1_flat, image2_flat, bins=256)

	# Normalize the joint histogram
	joint_hist_normalized = joint_hist / np.sum(joint_hist)

	# Calculate the marginal histograms
	hist1, _ = np.histogram(image1_flat, bins=256)
	hist2, _ = np.histogram(image2_flat, bins=256)

	# Normalize the marginal histograms
	hist1_normalized = hist1 / np.sum(hist1)
	hist2_normalized = hist2 / np.sum(hist2)

	# Compute the mutual information
	mi = 0
	for i in range(256):
		for j in range(256):
			if joint_hist_normalized[i, j] != 0 and hist1_normalized[i] != 0 and hist2_normalized[j] != 0:
				mi += joint_hist_normalized[i, j] * np.log2(joint_hist_normalized[i, j] / (hist1_normalized[i] * hist2_normalized[j]))

	return mi