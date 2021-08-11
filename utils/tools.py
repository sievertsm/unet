import os
import numpy as np
from tifffile import imread, imwrite
import matplotlib.pyplot as plt

import torch 
import torch.nn.functional as F


def get_true_index(arr, slab=1):
    '''
    helper function to split data with slabs
    '''
    idx=[]
    for s in arr:
        s *= slab
        temp = list(np.arange(s, s+slab))
        idx += temp
    return np.array(idx)


def display_predicted_mask(model, loader, idx=None, device='cpu', dtype=torch.float, ltype=torch.long):
    '''
    Function to make a prediction and show the image overlaid with the mask
    
    Input:
    model  -- trained Pytorch model
    loader -- data loader
    idx    -- index to display. If "None" a random index is chosen
    device -- device for model to make predictions
    dtype  -- dtype for model
    ltype  -- ltype for model
    '''

    if not idx:
        idx = np.random.randint(0, len(loader))

    print(f"Image Index: {idx}")


    for b, (x, y) in enumerate(loader):
        # move data to device
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=ltype)

        if b==idx:
            break

    model.eval()
    with torch.no_grad():
        pred = model(x)
        pred = F.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)



    img = x[0, 0, :, :].to(device='cpu')
    msk = y[0, 0, :, :].to(device='cpu')
    net = pred.to(device='cpu')[0]

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    for i in range(3):
        ax[i].imshow(img, cmap='bone')

    ax[1].imshow(msk, interpolation='nearest', alpha=0.5)
    ax[2].imshow(net, interpolation='nearest', alpha=0.5)

    for a in ax:
        a.axis('off')

    plt.tight_layout()

    
def predict_seg(model, x):
    
    model.eval()
    
    with torch.no_grad():
        pred = model(x)
        pred = F.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        
    return pred

def predict_denoise(model, x):
    model.eval()
    
    with torch.no_grad():
        pred = model(x)
        
    return pred

def write_predict_denoise(pred, i, fldr_output):
    
    # move predicion to cpu and save
    pred = pred.to(device='cpu')[0][0]
    pred = pred.numpy()
    imwrite(os.path.join(fldr_output, f"{i:04}.tiff"), pred)
    

def write_predict_seg(pred, i, fldr_output):
    
    # move predicion to cpu and save
    pred = pred.to(device='cpu')[0][0]
    pred = pred.numpy()
    plt.imsave(os.path.join(fldr_output, f"{i:04}.tiff"), pred, cmap='gray')
    
    
    
def apply_network_loader(model, loader, fldr_output, device='cpu', dtype=torch.float, segmentation=True):
    '''
    Function to apply a trained model to a stack of images
    
    Input:
    model       -- pytorch model with trained weights
    fldr_input  -- folder containing the input images 
    fldr_output -- name of folder for the output images (will create folder if one doesn't exist)
    device      -- device to be used
    dtype       -- data type to be used
    '''
    
    # create folder if there isn't one
    if not os.path.isdir(fldr_output):
        os.mkdir(fldr_output)
        print(f"Directory Created For: {fldr_output}\n")
        
    if segmentation:
        fn_predict = predict_seg
        fn_write = write_predict_seg
    else:
        fn_predict = predict_denoise
        fn_write = write_predict_denoise

    # get input files
    total_files = len(loader)
    
    # move model to device
    model = model.to(device=device)
    
    print(f"Beginning Predictions")
    
    # loop through all files 
    for i, x in enumerate(loader):
        
        x = x.to(device=device, dtype=dtype)

        pred = fn_predict(model, x)
        fn_write(pred, i, fldr_output)
        
        i+=1
        
        if i % 100 == 0:
            print(f"[{i}/{total_files}]")
            
    print('Complete!\n')