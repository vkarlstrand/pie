import os
import numpy as np
import torch



def create_save_path(save_path):
    """
    Creates paths to save logs and weights of model to.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    weight_path = os.path.join(save_path, 'models')
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
