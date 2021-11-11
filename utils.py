import os



def create_save_path(save_path):
    """
    Creates paths to save training logs and weights of model to, if they do
    not exist already.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    models_path = os.path.join(save_path, 'models')
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    train_log_path = os.path.join(save_path, 'train_log')
    if not os.path.exists(train_log_path):
        os.makedirs(train_log_path)
    return train_log_path, models_path


