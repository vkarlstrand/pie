import os
import numpy as np
import torch

def partition_data(file='data/data_labels.csv', ratios=[0.70, 0.20, 0.10], out_path='data/'):

    # Define ratios for traning, validation and test
    ratio_train = ratios[0]
    ratio_validation = ratios[1]
    ratio_test = ratios[2]

    # Import data
    data = np.genfromtxt(file, skip_header=1, dtype='str', delimiter=',')

    # Divide by image and label
    imgs = data[:,0]
    lbls = data[:,1]

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

    # Save to files
    np.savetxt(out_path+'data_labels_train.csv', np.vstack((imgs_train, lbls_train)).T, delimiter=',', fmt='%s')
    np.savetxt(out_path+'data_labels_validation.csv', np.vstack((imgs_validation, lbls_validation)).T, delimiter=',', fmt='%s')
    np.savetxt(out_path+'data_labels_test.csv', np.vstack((imgs_test, lbls_test)).T, delimiter=',', fmt='%s')

    # Print information
    print('PARTITIONED DATA')
    print('  Training:   ' + str(int(100*ratios[0]))+'%, ' + str(len(lbls_train)) + ' samples')
    print('  Testing:    ' + str(int(100*ratios[1]))+'%, ' + str(len(lbls_validation)) + ' samples')
    print('  Validation: ' + str(int(100*ratios[2]))+'%, ' + str(len(lbls_test)) + ' samples')



def create_save_path(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    weight_path = os.path.join(save_path, 'weights')
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    log_path = os.path.join(save_path, 'train_log')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    return log_path, weight_path



class SaveBestModel:
    def __init__(self, model, epoch, epochs, monitor_value, weight_path, best=None) -> None:
        self.epoch = epoch
        self.epochs = epochs
        self.monitor_value = monitor_value
        self.best = best
        self.weight_path = weight_path

    def run(self, model):
        # Save best model only
        if self.epoch == 0:
            # print(f'monitor value is: {self.monitor_value:.4f}')
            self.best = self.monitor_value
            # Save every epoch
            save_dir = os.path.join(self.weight_path, 'best.pt')
            torch.save(model, save_dir)
            print('Saved model.')
        elif self.best < self.monitor_value:
            # print(f'monitor value is: {self.monitor_value:.4f}')
            self.best = max(self.best, self.monitor_value)
            save_dir = os.path.join(self.weight_path, 'best.pt')
            torch.save(model, save_dir)
            print('Saved model.')
        elif (self.epoch + 1) == self.epochs:
            save_dir = os.path.join(self.weight_path, 'last.pt')
            torch.save(model, save_dir)
            print('Saved last model.')
        else:
            pass
