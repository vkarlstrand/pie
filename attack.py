import numpy as np
import torch
from torchvision.utils import save_image
import os
from plot import convert_image


class Attack:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.targeted = False
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.loss_function = torch.nn.CrossEntropyLoss()

    def attack(self, *input):
        """
        Function to be implemented for each attack type.
        """
        raise NotImplementedError

    def clamp(self, images):
        """
        Clamp image to min and max values based on mean and standard devation
        for each individual RGB value from training data set.
        """
        images = torch.clamp(input=images, min=-1, max=1)
        return images

    def get_targeted_labels(self, labels):
        """
        Draw targeted label uniformly from other labels than the actual one.
        """
        # Initialize targeted labels by cloning labels
        targeted_labels = labels.clone()
        # Iterate over all labels
        for i, label in enumerate(labels):
            # Initialize all possible targets
            targets = [0, 1, 2]
            # Remove the current label as a possible target
            targets.remove(int(label))
            # Draw a target uniformly and append to targeted_labels
            targeted_labels[i] = np.random.choice(targets)
        # Return targeted labels in a tensor
        return targeted_labels

    def similarity(self, images, attacked_images):
        """
        Function to compute similarity between original images and attacked
        images using normalized summed square difference.
        """
        similarities = torch.zeros(size=(images.shape[0],))
        for i, (image, attacked_image) in enumerate(zip(images, attacked_images)):
            sum_square_difference = torch.sum(torch.pow(image.ravel() - attacked_image.ravel(), 2))
            normalizing_factor = torch.sqrt(torch.sum(torch.pow(images, 2))*torch.sum(torch.pow(attacked_images, 2)))
            similarity = sum_square_difference/normalizing_factor
            similarities[i] = similarity
        return similarities


class FGSM(Attack):
    def __init__(self, model, epsilon=0.007, targeted=False):
        super().__init__('FGSM', model)
        self.epsilon = epsilon
        self.targeted = targeted

    def attack(self, images, labels):
        """
        FGSM attack algorithm.
        Based on https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = self.loss_function.to(self.device)

        if self.targeted:
            targeted_labels = self.get_targeted_labels(labels)
        else:
            targeted_labels = None

        images.requires_grad = True

        outputs = self.model(images)

        if self.targeted:
            cost = -loss(outputs, targeted_labels)
        else:
            cost = loss(outputs, labels)

        gradients = torch.autograd.grad(cost, images)[0]
        gradients_sign = gradients.sign()
        attacked_images = images + self.epsilon*gradients_sign
        attacked_images = self.clamp(attacked_images).detach()
        gradients_sign = self.clamp(gradients_sign)
        similarities = self.similarity(images, attacked_images)

        return attacked_images, gradients_sign, targeted_labels, similarities


class IFGSM(Attack):
    def __init__(self, model, epsilon=0.007, steps=1, targeted=False):
        super().__init__('FGSM', model)
        self.epsilon = epsilon
        self.steps = steps
        self.targeted = targeted

    def attack(self, images, labels):
        """
        IFGSM attack algorithm.
        Based on https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = self.loss_function.to(self.device)

        if self.targeted:
            targeted_labels = self.get_targeted_labels(labels)
        else:
            targeted_labels = None

        original_images = images.clone().detach()
        gradients_sum = 0

        for step in range(self.steps):
            images.requires_grad = True

            outputs = self.model(images)

            if self.targeted:
                cost = -loss(outputs, targeted_labels)
            else:
                cost = loss(outputs, labels)

            gradients = torch.autograd.grad(cost, images)[0]
            gradients_sign = gradients.sign()
            attacked_images = images + (self.epsilon/self.steps)*gradients_sign
            attacked_images = self.clamp(attacked_images).detach()
            images = attacked_images

            gradients_sum = gradients_sum + gradients_sign

        gradients_sign = self.clamp(gradients_sum/self.steps)
        similarities = self.similarity(original_images, attacked_images)

        return attacked_images, gradients_sign, targeted_labels, similarities



def evaluate_attacks(model, labels, attacked_images, attacked_labels, targeted_labels=None):
    """
    Print statistics and information regarding the attacks.
    """
    num_samples = labels.shape[0]
    out = '  {:<25} {}'
    print('RESULTS OF ATTACKS')
    print(out.format('True labels:',
                     ' '.join([str(int(lbl)) for lbl in labels])))
    print(out.format('Labels after attack:',
                     ' '.join([str(int(lbl)) for lbl in attacked_labels])))
    print(out.format('Wrong prediction:',
                     ' '.join(['x' if labels[i]!=attacked_labels[i] else ' ' for i in range(len(labels))])))
    print(out.format('Accuracy:',
                     str(np.around(np.count_nonzero(attacked_labels==labels)*100/num_samples, 2))+str('%')))
    if targeted_labels is not None:
        print('')
        print(out.format('True labels:',
                 ' '.join([str(int(lbl)) for lbl in labels])))
        print(out.format('Labels after attack:',
                         ' '.join([str(int(lbl)) for lbl in attacked_labels])))
        print(out.format('Targeted labels:',
                         ' '.join([str(int(lbl)) for lbl in targeted_labels])))
        print(out.format('Correct target:',
                         ' '.join(['o' if targeted_labels[i]==attacked_labels[i] else ' ' for i in range(len(labels))])))
        print(out.format('Target accuracy:',
                         str(np.around(np.count_nonzero(attacked_labels==targeted_labels)*100/num_samples, 2))+str('%')))



def save_images(images, attacked_images, gradients, labels, attacked_labels):
    save_dir = './attacked_images'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Saving images and attacked images to', save_dir)
    csv_images = ''
    csv_attacked_images = ''
    csv_gradients = ''
    for i, (img, att_img, grad, lbl, att_lbl) in enumerate(
    zip(images, attacked_images, gradients, labels, attacked_labels)):
        img = (img + 1)/2
        att_img = (att_img + 1)/2
        grad = (grad + 1)/2
        img_str = str(i)+'_img.png'
        att_str = str(i)+'_att.png'
        grad_str = str(i)+'_grad.png'
        save_image(img, save_dir+'/'+img_str)
        save_image(att_img, save_dir+'/'+att_str)
        save_image(grad, save_dir+'/'+grad_str)
        csv_images += (img_str+','+str(lbl.item())+'\n')
        csv_attacked_images += (att_str+','+str(att_lbl.item())+'\n')
        csv_gradients += (grad_str+','+str(att_lbl.item())+'\n')
    with open(save_dir+'/'+'original_images.csv', 'w') as file:
        file.write(csv_images[0:-1])
    with open(save_dir+'/'+'attacked_images.csv', 'w') as file:
        file.write(csv_attacked_images[0:-1])
    with open(save_dir+'/'+'gradients.csv', 'w') as file:
        file.write(csv_gradients[0:-1])
    print('Done.')

