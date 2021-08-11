import matplotlib.pyplot as plt

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
        ax[i].imshow(img, cmap=cmap, interpolation='nearest')
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