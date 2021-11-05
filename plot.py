import numpy as np
import matplotlib.pyplot as plt
import albumentations as album
import torch
import torchvision



def plot_dataset(dataset, rows, cols, image_width, mean, std):
    """
    Plot some examples from dataset by first unnormalize with provided
    mean and std to a range in approximately [0,1].
    """
    # Initialize figure and axes
    fig, axs = plt.subplots(rows, cols, figsize=(cols*image_width, rows*image_width))
    # Iterate over rows and columns
    for row in range(rows):
        for col in range(cols):
            # Extract image and label
            img = dataset[row*cols+col][0]['image']
            lbl = dataset[row*cols+col][1]
            # Change order of dimensions and convert to numpy array
            img = img.permute(1,2,0).numpy()
            # Unnormalize the image to range [0,1] so that it can be plotted with imshow
            mean_inv = torch.div(-mean, std)
            std_inv = torch.div(1, std)
            album_inv = album.Compose([album.Normalize(mean=mean_inv, std=std_inv, max_pixel_value=1.0)])
            img = album_inv(image=img)['image']
            # Plot image
            axs[row,col].imshow(img)
            axs[row,col].axis('off')
            axs[row,col].set_title('Class '+str(lbl), fontsize=20)
    return fig, axs


def plot_image_list(images, pred_labels, labels, rows, cols, image_width, mean, std):
    """
    Plot some examples from dataset by first unnormalize with provided
    mean and std to a range in approximately [0,1].
    """
    # Initialize figure and axes
    fig, axs = plt.subplots(rows, cols, figsize=(cols*image_width, rows*image_width))
    # Iterate over rows and columns
    N = len(labels)
    plot_cont = True
    for row in range(rows):
        for col in range(cols):
            if plot_cont:
                # Extract image and label
                img = images[row*cols+col]
                pred_lbl = pred_labels[row*cols+col]
                lbl = labels[row*cols+col]
                # Change order of dimensions and convert to numpy array
                img = img.permute(1,2,0).numpy()
                # Unnormalize the image to range [0,1] so that it can be plotted with imshow
                mean_inv = torch.div(-mean, std)
                std_inv = torch.div(1, std)
                album_inv = album.Compose([album.Normalize(mean=mean_inv, std=std_inv, max_pixel_value=1.0)])
                img = album_inv(image=img)['image']
                axs[row,col].set_title('Class '+str(lbl)+' > '+str(pred_lbl), fontsize=20)
                axs[row,col].imshow(img)
            axs[row,col].axis('off')
            if row*cols+col >= N-1:
                plot_cont = False
    return fig, axs