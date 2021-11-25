import numpy as np
import torch
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
import os
from plot import convert_image


class Attack:
    def __init__(self, name, model, mean, std):
        self.name = name
        self.model = model
        self.mean = mean
        self.std = std
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
        data_min = torch.div(-self.mean, self.std)
        data_max = torch.div(1.0 - self.mean, self.std)
        for i, (dmin, dmax) in enumerate(zip(data_min, data_max)):
            images[:,i,:,:] = torch.clamp(input=images[:,i,:,:], min=dmin, max=dmax)
        return images

    def quantize(self, images, steps=256):
        """
        Quantize image to specified number of steps.
        """
        data_min = torch.div(-self.mean, self.std)
        data_max = torch.div(1.0 - self.mean, self.std)
        for i, (dmin, dmax) in enumerate(zip(data_min, data_max)):
            step_size = (dmax - dmin)/steps
            images[:,i,:,:] = step_size*torch.floor(images[:,i,:,:]/step_size + 1/2)
        return images

    def rescale_gradients(self, gradients):
        """
        Scale gradients so that when epsilon is 1 and a pixel is in one edge of the range, it goes all the way
        out to the other edge, like in the article 'Explaining and harnessing adversarial examples'.
        """
        data_min = torch.div(-self.mean, self.std)
        data_max = torch.div(1.0 - self.mean, self.std)
        for i, (dmin, dmax) in enumerate(zip(data_min, data_max)):
            gradients[:,i,:,:] = gradients[:,i,:,:]*(dmax - dmin)/2
        return gradients

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

    def ssim(self, images, attacked_images):
        """
        Function to compute structural similarity (SSIM) of images.
        """
        similarities = torch.zeros(size=(images.shape[0],))
        data_min = torch.div(-self.mean, self.std)
        data_max = torch.div(1.0 - self.mean, self.std)
        data_range = (data_max.numpy() - data_min.numpy()).max()
        for i, (image, attacked_image) in enumerate(zip(images, attacked_images)):
            similarities[i] = ssim(im1=attacked_image.permute(1,2,0).numpy(),
                                   im2=image.permute(1,2,0).numpy(),
                                   multichannel=True,
                                   data_range=data_range)
        return similarities



class FGSM(Attack):
    def __init__(self, model, mean, std, epsilon=0.007, targeted=False):
        super().__init__('FGSM', model, mean, std)
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
        gradients_sign = self.rescale_gradients(gradients.sign())
        attacked_images = images + self.epsilon*gradients_sign
        attacked_images = self.clamp(attacked_images).detach()
        gradients_sign = self.clamp(gradients_sign)
        similarities = self.ssim(images, attacked_images)

        return attacked_images, gradients_sign, targeted_labels, similarities


class IFGSM(Attack):
    def __init__(self, model, mean, std, epsilon=0.007, steps=1, targeted=False, threshold=None, until_attacked=False):
        super().__init__('FGSM', model, mean, std)
        self.epsilon = epsilon
        self.steps = steps
        self.targeted = targeted
        self.threshold = threshold
        self.until_attacked = until_attacked

    def attack(self, images, labels):
        """
        IFGSM attack algorithm.
        Based on https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html.
        """
        # Get data
        num_samples = images.shape[0]
        true_labels = labels.clone()
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        original_images = images.clone().detach()
        loss = self.loss_function.to(self.device)

        # Get targeted labels
        if self.targeted:
            targeted_labels = self.get_targeted_labels(true_labels)
        else:
            targeted_labels = None

        # Initalize result arrays
        steps = torch.zeros(size=(num_samples,))
        epsilons = torch.zeros(size=(num_samples,))
        gradients_sum = torch.zeros(size=images.shape)
        attacked_images = torch.zeros(size=images.shape)

        # Initalize list of not done samples
        samples_not_done = torch.arange(start=0, end=num_samples)

        # Set model to evaluate
        self.model.eval()

        # Iterate for all steps
        for step in range(self.steps):

            # Predict label of current image
            images.requires_grad = True
            outputs = self.model(images)

            # Get cost for current preducted outputs and targeted labels or true labels
            if self.targeted:
                cost = -loss(outputs, targeted_labels)
            else:
                cost = loss(outputs, labels)

            # Compute gradient, take sign and rescale to mean and std of data
            gradients = torch.autograd.grad(cost, images)[0]
            gradients_sign = self.rescale_gradients(gradients.sign())

            # Attack image and clamp it
            attacked_images[samples_not_done,:,:,:] = images[samples_not_done,:,:,:] + (self.epsilon/self.steps)*gradients_sign[samples_not_done,:,:,:]
            attacked_images = attacked_images.detach()
            attacked_images = self.clamp(attacked_images)
            attacked_images_quantized = self.quantize(attacked_images.clone(), steps=256)

            # Save results of epsilon and number of steps
            epsilons[samples_not_done] += (self.epsilon/self.steps)
            steps[samples_not_done] += 1
            gradients_sum[samples_not_done,:,:,:] = gradients_sum[samples_not_done,:,:,:] + gradients_sign[samples_not_done,:,:,:]

            # Compute similarity
            similarities = self.ssim(original_images, attacked_images_quantized)

            # Update images with attacked images
            images = attacked_images.detach()

            # Predict labels of attacked quantized images
            outputs_attacked = self.model(attacked_images_quantized)
            attacked_labels = outputs_attacked.max(1, keepdim=False)[1]

            # See if similarity is below threshold
            if self.threshold is not None:
                samples_not_done = torch.where(similarities>self.threshold)[0]
            # See if attack succeded to misclassify the samples
            if self.until_attacked:
                samples_not_done = torch.where(attacked_labels==true_labels)[0]
            # Break of no samples left to attack
            if samples_not_done.shape[0] <= 0:
                break

        #  Normalize gradients and clamp them
        for i, step in enumerate(steps):
            gradients_sum[i] = gradients_sum[i]/step
        gradients_sign = self.clamp(gradients_sum)

        attacked_images = self.quantize(attacked_images.clone(), steps=256)
        attacked_images = self.clamp(attacked_images)

        # Return results
        return attacked_images, attacked_labels, gradients_sign, targeted_labels, similarities, epsilons, steps



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



def save_images(images, attacked_images, gradients, labels, attacked_labels, targeted_labels, init_labels, mean, std):
    save_dir = './attacked_images'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Saving images and attacked images to', save_dir)
    csv_images = ''
    csv_attacked_images = ''
    csv_targeted_images = ''
    csv_gradients = ''
    for i, (img, att_img, grad, lbl, att_lbl, targeted_lbl, init_lbl) in enumerate(
    zip(images, attacked_images, gradients, labels, attacked_labels, targeted_labels, init_labels)):
        img = torch.tensor(convert_image(img, mean, std)).permute(2,0,1)
        att_img = torch.tensor(convert_image(att_img, mean, std)).permute(2,0,1)
        grad = torch.tensor(convert_image(grad, mean, std)).permute(2,0,1)
        img_str = str(i)+'_img.png'
        att_str = str(i)+'_att.png'
        grad_str = str(i)+'_grad.png'
        save_image(img, save_dir+'/'+img_str)
        save_image(att_img, save_dir+'/'+att_str)
        save_image(grad, save_dir+'/'+grad_str)
        csv_images += (img_str+','+str(lbl.item())+'\n')
        csv_attacked_images += (att_str+','+str(att_lbl.item())+'\n')
        csv_targeted_images += (att_str+','+str(targeted_lbl.item())+'\n')
        csv_gradients += (grad_str+','+str(lbl.item())+','+str(init_lbl.item())+','+str(att_lbl.item())+','+str(targeted_lbl.item())+'\n')
    with open(save_dir+'/'+'original_images.csv', 'w') as file:
        file.write(csv_images[0:-1])
    with open(save_dir+'/'+'attacked_images.csv', 'w') as file:
        file.write(csv_attacked_images[0:-1])
    with open(save_dir+'/'+'targeted_images.csv', 'w') as file:
        file.write(csv_targeted_images[0:-1])
    with open(save_dir+'/'+'gradients.csv', 'w') as file:
        file.write(csv_gradients[0:-1])
    print('Done.')

