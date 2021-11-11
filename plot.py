# Imports
import matplotlib.pyplot as plt
import torch
from data import dataloader_to_lists



def plot_dataloader(dataloader, rows, cols, width):
    """
    Helper function to plot some examples from dataloader.
    """
    # Load images and labels from dataloader into lists
    images, labels = dataloader_to_lists(dataloader, max_samples=rows*cols)
    # Use plot_images to plot images and labels
    fig, axs = plot_images(images, labels, rows, cols, width)
    # Return figure object and axes
    return fig, axs



def plot_images(images, labels, rows, cols, width,
                predicted_labels=None, targeted_labels=None):
    """
    Plot some examples from dataset by first unnormalize to range [0,1].
    """
    # Initialize figure and axes
    fig, axs = plt.subplots(rows, cols, figsize=(cols*width, rows*width))
    # Iterate over rows and columns
    num_samples = len(labels)
    # Iterate over rows and columns
    for row in range(rows):
        for col in range(cols):
            # Get current index in list
            idx = row*cols+col
            # Extract image and label
            img = images[idx]
            lbl = labels[idx]
            # Change order of dimensions and convert to numpy array
            img = img.permute(1,2,0).numpy()
            # Rescale the image values to range [0,1]
            img = (img + 1)/2
            # Print image and corresponding label
            axs[row,col].imshow(img)
            axs[row,col].set_title(r'Class {}'.format(lbl), fontsize=20)
            # Turn of axis for all axes
            axs[row,col].axis('off')
    # Return figure object and axes
    return fig, axs






            # img = img.permute(1,2,0)
            # img = (img + 1.0)/2.0




    #
    # for row in range(rows):
    #     for col in range(cols):
    #         # Extract image and label
    #         img = dataset[row*cols+col][0]['image']
    #         lbl = dataset[row*cols+col][1]
    #         # Change order of dimensions and convert to numpy array
    #         img = img.permute(1,2,0).numpy()
    #         # Unnormalize the image to range [0,1] so that it can be plotted with imshow
    #         mean_inv = torch.div(-mean, std)
    #         std_inv = torch.div(1, std)
    #         album_inv = album.Compose([album.Normalize(mean=mean_inv, std=std_inv, max_pixel_value=1.0)])
    #         img = album_inv(image=img)['image']
    #         # Plot image
    #         axs[row,col].imshow(img)
    #         axs[row,col].axis('off')
    #         axs[row,col].set_title('Class '+str(lbl), fontsize=20)
    return fig, axs