import os
import torch
import numpy as np



def train(dataloader, model, device, loss_function, optimizer):
    """
    Train on batches in dataloader for current epoch.
    """

    # Set model in training mode
    model.train()

    # Initalize result metrics
    total_loss = 0.0
    total_accuracy = 0.0
    num_samples = 0

    # Iterate over all batches
    for batch, (images, labels) in enumerate(dataloader):

        # Forward
        images = images['image'].to(device)
        labels = labels.to(device).long()
        preds = model(images)

        # Compute loss
        loss = loss_function(preds, labels)
        total_loss += loss.item()

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

        # Predict labels and compute number of correctly predicted labels in batch
        _, pred_labels = torch.max(preds, 1)
        batch_correct = (pred_labels==labels).squeeze().sum().item()
        total_accuracy += batch_correct

        # Compute batch size and increase number of total samples
        batch_size = labels.size(0)
        num_samples += batch_size

    # Compute mean accuracy and loss
    mean_accuracy = total_accuracy/num_samples
    mean_loss = total_loss/len(dataloader)

    return mean_accuracy, mean_loss



def valid(dataloader, model, device, loss_function, optimizer):
    """
    Evaluate and validate on batches in dataloader for current epoch.
    """

    # Set model in evaluation mode
    model.eval()

    # Initalize result metrics
    total_loss = 0.0
    total_accuracy = 0.0
    num_samples = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):

            # Load images and labels and predict labels
            images = images['image'].to(device)
            labels = labels.to(device).long()
            preds = model(images)

            # Compute loss
            loss = loss_function(preds, labels)
            total_loss += loss.item()

            # Predict labels and compute number of correctly predicted labels in batch
            _, pred_labels = torch.max(preds, 1)
            batch_correct = (pred_labels==labels).squeeze().sum().item()
            total_accuracy += batch_correct

            # Compute batch size and increase number of total samples
            batch_size = labels.size(0)
            num_samples += batch_size

    # Compute mean accuracy and loss
    mean_accuracy = total_accuracy/num_samples
    mean_loss = total_loss/len(dataloader)

    return mean_accuracy, mean_loss



def test(dataloader, model, device, loss_function, optimizer):
    """
    Test on batches in dataloader and save misclassified samples.
    """

    # Set model in evaluation mode
    model.eval()

    # Initalize result metrics and lists for misclassifications
    total_loss = 0.0
    total_accuracy = 0.0
    num_samples = 0
    misclassified_images = []
    misclassified_labels = []
    correct_labels = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):

            # Load images and labels and predict labels
            images = images['image'].to(device)
            labels = labels.to(device).long()
            preds = model(images)

            # Compute loss
            loss = loss_function(preds, labels)
            total_loss += loss.item()

            # Predict labels and compute number of correctly predicted labels in batch
            _, pred_labels = torch.max(preds, 1)
            batch_correct = (pred_labels==labels).squeeze().sum().item()
            total_accuracy += batch_correct

            # Find misclassifications and save
            misclassified_idxs = np.where(pred_labels!=labels)
            correct_label = labels[misclassified_idxs]
            misclassified_label = pred_labels[misclassified_idxs]
            if misclassified_label.numel():
                for i in range(len(misclassified_label)):
                    misclassified_image = images[misclassified_idxs[0][i],:,:,:].squeeze()
                    misclassified_images.append(misclassified_image)
                    misclassified_labels.append(int(misclassified_label[i]))
                    correct_labels.append(int(correct_label[i]))

            # Compute batch size and increase number of total samples
            batch_size = labels.size(0)
            num_samples += batch_size

    # Compute mean accuracy and loss
    mean_accuracy = total_accuracy/num_samples
    mean_loss = total_loss/len(dataloader)

    return mean_accuracy, mean_loss, misclassified_images, misclassified_labels, correct_labels



def save_model(model, epoch, best_monitor_value, monitor_value, epochs, models_path):
    """
    Function to save current best model during training and last model weights.
    Needs the best_monitor_value to be initialized to 0 outside this function.
    """
    if best_monitor_value < monitor_value:
        best_monitor_value = max(best_monitor_value, monitor_value)
        save_dir = os.path.join(models_path, 'best.pt')
        torch.save(model, save_dir)
        print('Saved current model as best model.')
        return monitor_value
    elif (epoch + 1) == epochs:
        save_dir = os.path.join(models_path, 'last.pt')
        torch.save(model, save_dir)
        print('Saved last model.')
        return best_monitor_value
    else:
        return best_monitor_value


