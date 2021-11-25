# Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from data import dataloader_to_lists
import albumentations as album



def plot_dataloader(dataloader, rows, cols, width, mean, std):
    """
    Helper function to plot some examples from dataloader.
    """
    # Load images and labels from dataloader into lists
    images, labels = dataloader_to_lists(dataloader, max_samples=rows*cols)
    # Use plot_images to plot images and labels
    fig, axs = plot_images(images, labels, rows, cols, width, mean, std)
    # Return figure object and axes
    return fig, axs



def convert_image(image, mean, std):
    """
    Helper function to rescale image values to the range [0,1] again and
    then to [0,255] as integers.
    """
    # # Rearange order of dimensions so that the color components are last
    image = image.permute(1,2,0).numpy()
    mean_inv = torch.div(-mean, std)
    std_inv = torch.div(1, std)
    album_inv = album.Compose([album.Normalize(mean=mean_inv, std=std_inv, max_pixel_value=1.0)])
    image = album_inv(image=image)['image']
    image = np.round(255*image).astype(int)
    return image



def plot_images(images, labels, rows, cols, width, mean, std,
                predicted_labels=None, targeted_labels=None):
    """
    Plot some examples from dataset by first unnormalize to range [0,1].
    """
    # Initialize figure and axes
    fig, axs = plt.subplots(rows, cols, figsize=(cols*width, rows*width))
    # Setup
    num_samples = len(labels)
    print_img = True
    # Iterate over rows and columns
    for row in range(rows):
        for col in range(cols):
            # Get current index in list
            idx = row*cols+col
            if print_img:
                # Extract image and label
                img = images[idx]
                lbl = labels[idx]
                # Convert image to correct range
                img = convert_image(img, mean, std)
                # Print image and corresponding label
                axs[row,col].imshow(img)
                if predicted_labels is not None:
                    pred_lbl = predicted_labels[idx]
                    axs[row,col].set_title(r'Class {} $\rightarrow$ {}'.format(lbl, pred_lbl), fontsize=20)
                else:
                    axs[row,col].set_title(r'Class {}'.format(lbl), fontsize=20)
            # Turn of axis for all axes
            axs[row,col].axis('off')
            if idx >= num_samples - 1:
                print_img = False
    # Return figure object and axes
    return fig, axs



def plot_attacks(rows, images, labels, init_labels, attacked_images, attacked_labels, gradients,
                 epsilons, steps, mean, std, targeted_labels=None):
    """
    Plot original images, the pertubation and attacked images side by side.
    """
    # Initialize figure and axes
    fig, axs = plt.subplots(rows, 3, figsize=(15,5*rows))
    # Iterate over all attacked images
    for i, (img, lbl, init_lbl, att_img, att_lbl, grad, eps, step) in enumerate(
    zip(images, labels, init_labels, attacked_images, attacked_labels, gradients, epsilons, steps)):
        # Get current axis
        ax = axs[i]
        # Convert images to range [0,1] and re-order dimensions for plotting
        img = convert_image(img, mean, std)
        grad = convert_image(grad, mean, std)
        att_img = convert_image(att_img, mean, std)
        # Plot original image
        ax[0].imshow(img)
        ax[0].set_title(r'Class {} $\rightarrow$ {}'.format(lbl, init_lbl), fontsize=20)
        # Plot gradients
        ax[1].imshow(grad)
        ax[1].set_title('Attack ($\epsilon='+str(np.around(eps.item(),5))+',T='+str(int(step))+'$)', fontsize=20)
        # Plot attacked image
        ax[2].imshow(att_img)
        ax[2].set_title(r'Class {} $\rightarrow$ {}{}'.format(lbl, att_lbl,
            ' (target '+str(int(targeted_labels[i]))+')' if targeted_labels is not None else ''), fontsize=20)
        # Turn of axis for all aexes
        for a in ax:
            a.axis('off')
        # Only plot rows images
        if i >= rows - 1:
            break
    # Return figure object and axes
    return fig, axs


