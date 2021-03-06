import os
import pandas as pd
import numpy as np
from tifffile import imread
import itertools

import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageTargetData(Dataset):
    '''
    Custom Pytorch Dataset object. 
    
    Input:
    datapoints -- list of datapoints generated by the 'get_datapoints' function
    transform  -- list of transformations
    '''
    
    def __init__(self, datapoints, transform=None):
        
        self.data = datapoints
        self.transform = transform
        self.slab = len(datapoints[0][0])
        
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        
        D = self.slab
        
        img, tgt = self.data[idx]
        
        img_target = imread(tgt)
        img_target = np.expand_dims(img_target, axis=0)
    
        _, H, W = img_target.shape
        img_input = np.zeros((D, H, W))
        
        
        
        for i in range(len(img)):
            img_input[i] = imread(img[i])
            
        sample = (img_input, img_target)
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
class Normalize(object):
    '''
    Transformation that divides all values in the image by "denom"
    '''
    def __call__(self, sample, denom=255):
        img_input, img_target = sample

        img_input = img_input / denom
        img_target =  img_target / denom
        
        sample = (img_input, img_target)
        
        return sample
    
class FlipVertical(object):
    '''
    Transformation that flips the image vertically with probablity p
    '''
    def __call__(self, sample, p=0.5):
        rand = np.random.random_sample()
        
        if p >= rand:
            img_input, img_target = sample
            
            for i in range(len(img_input)):
                img_input[i] = np.flipud(img_input[i])
                
            img_target[0] = np.flipud(img_target[0])
            sample = (img_input, img_target)
            return sample
        
        else:
            return sample
        
class FlipHorizontal(object):
    '''
    Transformation that flips the image horizontally with probablity p
    '''
    def __call__(self, sample, p=0.5):
        rand = np.random.random_sample()
        
        if p >= rand:
            img_input, img_target = sample
            
            for i in range(len(img_input)):
                img_input[i] = np.fliplr(img_input[i])
                
            img_target[0] = np.fliplr(img_target[0])
            sample = (img_input, img_target)
            return sample
        
        else:
            return sample
            
    
class ToTensor(object):
    '''
    Transformation that changes an ndarray to a tensor
    '''
    
    def __call__(self, sample):

        img_input, img_target = sample
        
        img_input = torch.from_numpy(img_input.copy())
        img_target = torch.from_numpy(img_target.copy())
        
        return img_input, img_target
    
    
# ######################################################################################################################
def get_slab_idx(i, slab, total_files):
    '''
    Function to get indexes of a slab
    
    INPUT:
    i           -- index
    slab        -- the slab thickness in slices for the entire input
    total_files -- total number of files available
    
    OUTPUT:
    idx_slab -- slab indicies corresponding to input i
    '''
    
    # find half the slab
    slab_half = slab//2

    # get range of half the slab
    idx = np.arange(1 + slab_half)

    # get index for lower and upper slices
    lower = np.flip(i - idx)
    upper = i + idx

    # boolean logic to perform reflection at the beginning and end of a scan
    # beginning
    if i < slab_half:
        idx_slab = np.concatenate([np.flip(upper[1:]), upper])

    # end
    elif i > total_files - (slab_half+1):
        idx_slab = np.concatenate([lower, np.flip(lower[:-1])])

    # other
    else:
        idx_slab = np.concatenate([lower, upper[1:]])

    return idx_slab




def get_datapoints(input_target, slab=1, split=0, crop_file=[]):
    '''
    Function to create datapoints for training
    
    INPUT:
    input_target -- tuple containing (input folder, target folder)
    slab         -- how thick of a slab to be used for the input must be odd >= 1
    split        -- how much of the datapoints to be used in validation float < 1.0
    crop_file    -- 
    
    OUTPUT:
    datapoint_train -- list of training datapoints
    datapoint_valid -- list of validation datapoints
    '''
    
    # initialize lists
    datapoint_train=[]
    datapoint_valid=[]
    
    # loop through list of tuples
    for sample in input_target:
        
        # unpack tuple
        img_in, img_tg = sample
        
        # get list of files
        fls_in = os.listdir(img_in)
        fls_tg = os.listdir(img_tg)
        
        # join list of files
        fls_in = np.array([os.path.join(img_in, fl) for fl in fls_in])
        fls_tg = np.array([os.path.join(img_tg, fl) for fl in fls_tg])
        
        if len(crop_file) > 0:
            crop_key = img_in.split('/')[-1]
            start, stop = crop_file.loc[crop_key]
            fls_in = fls_in[int(start):int(stop)]
            fls_tg = fls_tg[int(start):int(stop)]
        
        total_files = len(fls_in)
        
        # initialize list for current folders datapoints
        dt=[]
        dv=[]
        
        # set up 'mod' based on the 'split' value
        if split > 0:
            mod = total_files // int(split * total_files)
        else:
            mod = int(total_files + 1)
            
        for i in range(total_files):
            
            if (i+1) % mod == 0:
                dv.append((list(fls_in[get_slab_idx(i, slab, total_files)]), fls_tg[i]))
            else:
                dt.append((list(fls_in[get_slab_idx(i, slab, total_files)]), fls_tg[i]))
                
        datapoint_train = list(itertools.chain(datapoint_train, dt))
        datapoint_valid = list(itertools.chain(datapoint_valid, dv))
        
    return datapoint_train, datapoint_valid
        
        
def get_datapoints_apply(image_fldrs, slab=1, crop_file=[]):
    '''
    Function to create datapoints for prediction
    
    INPUT:
    image_fldrs -- list containing image folders
    slab        -- how thick of a slab to be used for the input must be odd >= 1
    crop_file   --
    
    OUTPUT:
    dats_all -- list of datapoints
    '''

    dats_all=[]

    for fldr in image_fldrs:

        flsin = os.listdir(fldr)

        flsin = np.array([os.path.join(fldr, fl) for fl in flsin])
        
        if len(crop_file) > 0:
            crop_key = fldr.split('/')[-1]
            start, stop = crop_file.loc[crop_key]
            flsin = flsin[int(start):int(stop)]

        total_files = len(flsin)

        d=[]

        for i in range(total_files):
            d.append(list(flsin[get_slab_idx(i, slab, total_files)]))


        dats_all = list(itertools.chain(dats_all, d))
        
    return dats_all


class ImageDataApply(Dataset):
    '''
    Custom Pytorch Dataset object. 
    
    Input:
    datapoints -- list of datapoints generated by the 'get_datapoints_apply' function
    transform  -- list of transformations
    '''
    
    def __init__(self, datapoints, transform=None):
        
        self.data = datapoints
        self.transform = transform
        self.slab = len(datapoints[0])
        
        self.H, self.W = imread(datapoints[0][0]).shape
        
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        
        D = self.slab
        
        img = self.data[idx]
        
        img_input = np.zeros((D, self.H, self.W))

        for i in range(len(img)):
            img_input[i] = imread(img[i])
        
        if self.transform:
            img_input = self.transform(img_input)
            
        return img_input
    
    
class ToTensorApply(object):
    '''
    Transformation that changes an ndarray to a tensor
    '''
    
    def __call__(self, img_input):
        
        img_input = torch.from_numpy(img_input.copy())
        
        return img_input
    
    
class NormalizeApply(object):
    '''
    Transformation that divides all values in the image by "denom"
    '''
    def __call__(self, img_input, denom=255):

        img_input = img_input / denom
        
        return img_input