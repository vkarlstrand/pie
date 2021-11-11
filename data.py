# Imports
import os
import numpy as np
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import PIL



def partition_data(input_file, ratios, out_path, uniform=False):
    """
    Partition data into training, validation and test data.
    Saves partitioned data to three .csv files.
    """

    # Define ratios for traning, validation and test
    ratio_train = ratios[0]
    ratio_validation = ratios[1]
    ratio_test = ratios[2]

    # Import data
    data = np.genfromtxt(input_file, skip_header=1, dtype='str', delimiter=',')

    # Divide by image and label
    imgs = data[:,0]
    lbls = data[:,1]

    idxs_shuffle = np.random.permutation(len(lbls))
    imgs = imgs[idxs_shuffle]
    lbls = lbls[idxs_shuffle]

    # Print info
    print('NUMBER OF SAMPLES PER CLASS')
    samples_per_class(lbls)
    print('')

    # If uniformly many classes (do not use all data however)
    if uniform:
        # Find indices of each class
        idxs0 = np.where(lbls=='0')
        idxs1 = np.where(lbls=='1')
        idxs2 = np.where(lbls=='2')

        # Find number of samples for each class
        N0 = len(idxs0[0])
        N1 = len(idxs1[0])
        N2 = len(idxs2[0])
        N = min(N0, N1, N2)

        # Divide data by classes
        imgs0 = imgs[idxs0]
        lbls0 = lbls[idxs0]
        imgs1 = imgs[idxs1]
        lbls1 = lbls[idxs1]
        imgs2 = imgs[idxs2]
        lbls2 = lbls[idxs2]

        # Shuffle order
        idxs_shuffle0 = np.random.permutation(N0)
        idxs_shuffle1 = np.random.permutation(N1)
        idxs_shuffle2 = np.random.permutation(N2)
        imgs0 = imgs0[idxs_shuffle0]
        lbls0 = lbls0[idxs_shuffle0]
        imgs1 = imgs1[idxs_shuffle1]
        lbls1 = lbls1[idxs_shuffle1]
        imgs2 = imgs2[idxs_shuffle2]
        lbls2 = lbls2[idxs_shuffle2]

        # Take samples from each class
        imgs = []
        lbls = []
        for n in range(N):
            imgs.append(imgs0[n])
            lbls.append(lbls0[n])
            imgs.append(imgs1[n])
            lbls.append(lbls1[n])
            imgs.append(imgs2[n])
            lbls.append(lbls2[n])

    # Divide into training, validation and test data sets
    idxs_train = int(ratio_train*len(imgs))
    idxs_validation = int((ratio_train+ratio_validation)*len(imgs))
    imgs_train = imgs[0:idxs_train]
    imgs_validation = imgs[idxs_train:idxs_validation]
    imgs_test = imgs[idxs_validation:]
    lbls_train = lbls[0:idxs_train]
    lbls_validation = lbls[idxs_train:idxs_validation]
    lbls_test = lbls[idxs_validation:]

    # If path does not exist, create it
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Save to files
    np.savetxt(out_path+'data_labels_train.csv', np.vstack((imgs_train, lbls_train)).T, delimiter=',', fmt='%s')
    np.savetxt(out_path+'data_labels_validation.csv', np.vstack((imgs_validation, lbls_validation)).T, delimiter=',', fmt='%s')
    np.savetxt(out_path+'data_labels_test.csv', np.vstack((imgs_test, lbls_test)).T, delimiter=',', fmt='%s')

    # Print information
    print('PARTITIONED DATA')
    print('  Training:    ' + str(int(100*ratios[0]))+'%, ' + str(len(lbls_train)) + ' samples')
    samples_per_class(lbls_train)
    print('')
    print('  Validation:  ' + str(int(100*ratios[1]))+'%, ' + str(len(lbls_validation)) + ' samples')
    samples_per_class(lbls_validation)
    print('')
    print('  Testing:     ' + str(int(100*ratios[2]))+'%, ' + str(len(lbls_test)) + ' samples')
    samples_per_class(lbls_test)
    print('')
    print('  Total:       ' + str(len(lbls_train)+len(lbls_validation)+len(lbls_test)))



def samples_per_class(lbls):
    """
    Helper function to plot samples in each class for specific array of labels.
    """
    counter = Counter(lbls)
    keys = counter.keys()
    values = counter.values()
    pairs = sorted(zip(keys, values))
    sum = 0
    output_format = '    Class {:<1}: {:>5}   {:>3}%'
    for pair in pairs:
        sum += pair[1]
    for pair in pairs:
        print(output_format.format(pair[0], pair[1], int(100*pair[1]/sum)))
    print('    Total:   {:>5}'.format(sum))



def compute_mean_std(dataset):
    """
    Compute mean and standard deviation of dataset of images.
    """
    # Get image size
    image_size = dataset[0][0]['image'].shape[1]
    # Load data
    dataloader = DataLoader(dataset, batch_size=32)
    # Initialize sums for results
    total_sum = torch.tensor([0.0, 0.0, 0.0])
    total_sum_square = torch.tensor([0.0, 0.0, 0.0])
    # Iterate over all data and sum sums of batches
    for data in dataloader:
        img = data[0]['image']
        total_sum += img.sum(axis=[0,2,3])
        total_sum_square += (img**2).sum(axis=[0,2,3])
    # Compute total pixel count and overall mean and std
    total_pixel_count = len(dataset) * image_size * image_size
    mean = total_sum/total_pixel_count
    std = torch.sqrt((total_sum_square/total_pixel_count)-(mean**2))
    # Return results
    return mean, std



def default_image_loader(image_path):
    """
    Helper function to open image file as PIL image in RBG format.
    """
    return PIL.Image.open(image_path).convert('RGB')



class LoadDatasetFromCSV(Dataset):
    """
    Class to import dataset from .csv files. Can perform transformations defined
    by agumentations using albumentations.
    The class inhereits from from torch.utils.data's Dataset.
    """
    def __init__(self, image_root, csv_path, transforms=None, loader=default_image_loader):
        # Root directory containing images
        self.image_root = image_root
        # Read data using pandas
        self.data = pd.read_csv(csv_path, header=None)
        # Initialize list for images
        imgs = []
        # Get file names in first column as numpy array
        files_names = np.array(self.data.iloc[:,0])
        # Iterate over all files names and join the image name with the root path
        for img in files_names:
            imgs.append(os.path.join(self.image_root, str(img)))
        # Get labels in second column as numpy array
        self.labels = np.asarray(self.data.iloc[:,1])
        # Define images, now with full path, as attribute
        self.images = imgs
        # Define provided transform as attribute
        self.transforms = transforms
        # Use the provided loader to load the images
        self.loader = loader

    def __getitem__(self, index):
        # Get image and label
        img = self.images[index]
        lbl = self.labels[index]
        # Load image using the loader
        img = np.array(self.loader(img))
        # Perform transormation if it was provided
        if self.transforms is not None:
            img = self.transforms(image=img)
        return img, lbl

    def __len__(self):
        # Return number of samples in dataset
        return len(self.data.index)



def dataloader_to_lists(dataloader, max_samples):
    """
    Helper function to convert dataloader of images and labels to torch tensors
    with images and labels.
    """
    # Initialize lists for holding images and labels, as well asa counter
    images = []
    labels = []
    counter = 0
    # Iterate over all batches
    for imgs, lbls in dataloader:
        # Extract image
        imgs = imgs['image']
        # Iterate over all samples in batch
        for img, lbl in zip(imgs, lbls):
            # Append image and label to lists and increase counter
            images.append(img)
            labels.append(lbl)
            counter += 1
        # Break if maximum number of samples have been collected
        if counter >= max_samples - 1:
            break
    # Return images and labels in lists
    return images, labels


