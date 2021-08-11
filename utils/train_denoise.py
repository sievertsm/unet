import pandas as pd
import numpy as np
from tifffile import imread
import os
import matplotlib.pyplot as plt

from datetime import datetime
import time

import torch 
import torch.nn.functional as F

def check_loss(model, loader, device='cpu', dtype=torch.float):
    '''
    Function that checks loss
    
    Input:
    model  -- Pytorch model
    loader -- Pytorch data loader
    device -- device for the computation
    '''
    model.eval()
    
    total_loss=[]
    with torch.no_grad():
        for b, (x, y) in enumerate(loader):
            
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)
#             y = y[:, 0, :, :]
            
            scores = model(x)
#             scores = F.softmax(scores, dim=1)
            loss = F.mse_loss(scores, y).item()
            total_loss.append(loss)
            
        current_loss = np.mean(total_loss)
        
    return current_loss

def save_model(file_name, model, optimizer=None, results_folder=None, **kwargs):
    '''
    Function to save the model parameters as a dictionary
    
    Input:
    file_name      -- (str) file name for the dictionary (.tar will be added automatically)
    model          -- Pytorch model
    optimizer      -- Pytorch optimizer
    results_folder -- (str) results folder name
    **kwargs
    '''
    
    kwargs['model_state_dict'] = model.state_dict()
    
    if optimizer:
        kwargs['optimizer_state_dict'] = optimizer.state_dict()
        
    if results_folder:
        file_name = os.path.join(results_folder, file_name)
    
    torch.save(kwargs, f'{file_name}.tar')
    
    
def display_images(images, titles=['Input', 'Target', 'Network Output'], cmap='bone', height=10, save_path=None, show=True):
    '''
    Function to display and/or saves images
    
    Input:
    images    -- (list) images of shape (H, W) to be included in the figure
    titles    -- (list) title for each image in images
    cmap      -- (string) colormap for the figure
    height    -- (number) value for the height of the figure. figure size and text scale by this value
    save_path -- (string/path) path to save the figure. Must include a valid extension (.png, .jpg, etc.)
    show      -- (bool) whether the image should be displayed in the notebook
    '''
    
    num_img = len(images)
    
    fig, ax = plt.subplots(nrows=1, ncols=num_img, figsize=(height, height*num_img))
    
    for i, img in enumerate(images):
        ax[i].imshow(img, cmap=cmap)
#         ax[i].imshow(img, cmap=cmap, interpolation='nearest')
        ax[i].set_title(titles[i], fontsize=2*height)
        
    for a in ax:
        a.axis('off')
        
    plt.tight_layout()
    
    if save_path != None:
        plt.savefig(save_path, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
        
    
    
def train(model, optimizer, loader_train, loader_val, epochs=1, print_every=5, device='cpu', dtype=torch.float, early_stop=None, save_results=False, display_input=None):
    '''
    Define training loop
    
    Inputs:
    model             -- model to be trained
    optimizer         -- optimizer to be used
    loader_train      -- data loader for training data
    loader_val        -- data loader for validation data
    epochs            -- number of epochs to train for
    print_every       -- print every number of batches
    device            -- device used for training (cpu or cuda)
    dtype             -- data type to be used
    early_stop        -- stop early if no improvement for so many epochs
    save_results      -- boolean wheter to save results
    display_input     -- option to specify an image to display at the end of each epoch
    
    Outputs:
    loss_train        -- list of the training loss
    loss_val          -- list of the validation loss
    '''
    
    if save_results:
        # create results folder
        today = datetime.today()
        results_fldr = 'results_' + today.strftime('%y%m%d_%H%M')
        if not os.path.isdir(results_fldr):
            os.mkdir(results_fldr)
            print(f'Directory "{results_fldr}" Created\n')
        else:
            print(f'Directory "{results_fldr}" Already Exists\n')
    
    # move model to gpu
    model = model.to(device=device)
    
    # initialize best validation loss to a large number
    best_val_loss=np.float('inf')
    best_model=None
    no_improvement=0
    
    # initialize empty lists for training and validation loss
    loss_train=[]
    loss_val=[]
    
    # record the start time
    start_time = time.time()
    
    # training loop
    for e in range(epochs):
        e+=1 # start with epoch 1 not 0
        
        print(f'\nStarting Epoch: {e}')
        
        # batch loop
        for b, (x, y) in enumerate(loader_train):
            b+=1
            # set model to "train" mode
            model.train()
            
            # move data to device
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)
#             y = y[:, 0, :, :]
            
            # get model output
            scores = model(x)
#             scores = F.softmax(scores, dim=1)
            
            # get loss
            loss = F.mse_loss(scores, y)
            
            # zero gradients
            optimizer.zero_grad()
            # compute backward pass
            loss.backward()
            # take step
            optimizer.step()
            
            # print
            if b % print_every == 0:
                lap_time = (time.time() - start_time)/3600
                print(f"Epoch: {e}, Batch: {b:04}, Batch loss: {round(loss.item(), 6)}, Time: {round(lap_time, 2)} hr")
          
        # after epoch check the training loss
        l_trn = check_loss(model, loader_train, device=device, dtype=dtype)
        loss_train.append(l_trn)
        
        # check validation loss
        l_val = check_loss(model, loader_val, device=device, dtype=dtype)
        loss_val.append(l_val)
        
        # print time and loss results for the epoch
        lap_time = (time.time() - start_time)/3600
        print(f"Epoch: {e}, Training loss: {round(loss_train[-1], 6)}, Val loss: {round(loss_val[-1], 6)}, Time: {round(lap_time, 2)} hr")
        
        # display results and save model if it is the best validation loss so far
        if l_val < best_val_loss:
            best_val_loss = l_val
            no_improvement=0
            
            # save model
            if save_results:
                save_model(file_name=f"params_e{e:04}", model=model, optimizer=optimizer, results_folder=results_fldr, epoch=e, loss=l_val)
            
            if display_input == None:
                # get first image from val set
                for b, (x, y) in enumerate(loader_val):
                    # move data to device
                    x = x.to(device=device, dtype=dtype)
                    y = y.to(device=device, dtype=dtype)
                    if b==0:
                        break
            else:
                x, y = display_input
                x = x.to(device=device, dtype=dtype)
                y = y.to(device=device, dtype=dtype)

            # get prediction
            model.eval()
            with torch.no_grad():
                pred = model(x)
#                 pred = F.softmax(pred, dim=1)
#                 pred = torch.argmax(pred, dim=1)
            
            # display images
            x_middle = x.shape[1]//2
            y_middle = y.shape[1]//2
            display_images(images=[
                                x[0, x_middle, :, :].to(device='cpu'),
                                y[0, y_middle, :, :].to(device='cpu'),
                                pred.to(device='cpu')[0][0]
                                ])
            
        # if validation loss didn't improve record no_improvement
        else:
            no_improvement+=1
            
        # perform early stopping
        if early_stop:
            if no_improvement >= early_stop:
                print('EARLY STOP')
                return loss_train, loss_val
            

    return loss_train, loss_val