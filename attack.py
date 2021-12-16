import numpy as np
import torch
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
import os
from plot import convert_image
from plot import plot_attacks
from data import LoadDatasetFromCSV, compute_mean_std
from torch.utils.data import DataLoader
import gc
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
import copy
from PIL import Image as im

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

    def quantize(self, images, max_val=255):
        """
        Quantize image to specified number of steps.
        """
        data_min = torch.div(-self.mean, self.std)
        data_max = torch.div(1.0 - self.mean, self.std)
        for i, (dmin, dmax) in enumerate(zip(data_min, data_max)):
            # Convert to range [0,1]
            images[:,i,:,:] = (images[:,i,:,:] - dmin)/(dmax - dmin)
            # Scale to {0,255} and round, then divide by 255 again
            images[:,i,:,:] = torch.round(max_val*images[:,i,:,:])/max_val
            # Return to range [dmin,dmax]
            images[:,i,:,:] = images[:,i,:,:]*(dmax - dmin) + dmin
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

        if images.shape != attacked_images.shape:
            attacked_images = attacked_images[None, :, :, :]
        similarities = torch.zeros(size=(images.shape[0],))
        data_min = torch.div(-self.mean, self.std)
        data_max = torch.div(1.0 - self.mean, self.std)
        data_range = (data_max.numpy() - data_min.numpy()).max()
        try:
            for i, (image, attacked_image) in enumerate(zip(images, attacked_images)):
                similarities[i] = ssim(im1=attacked_image.permute(1,2,0).numpy(),
                                       im2=image.permute(1,2,0).numpy(),
                                       multichannel=True,
                                       data_range=data_range)
        except:
            images = images.cpu()
            attacked_images = attacked_images.cpu()
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
    def __init__(self, model, mean, std, epsilon=0.007, steps=1, targeted=False, threshold=None, until_attacked=False, compare_orig = False):
        super().__init__('FGSM', model, mean, std)
        self.epsilon = epsilon
        self.steps = steps
        self.targeted = targeted
        self.threshold = threshold
        self.until_attacked = until_attacked
        self.compare_orig = compare_orig

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


        # Initalize result arrays
        steps = torch.zeros(size=(num_samples,))
        epsilons = torch.zeros(size=(num_samples,))
        gradients_sum = torch.zeros(size=images.shape)
        attacked_images = torch.zeros(size=images.shape)

        # Initalize list of not done samples
        samples_not_done = torch.arange(start=0, end=num_samples)

        # Set model to evaluate
        self.model.eval()
        
        orig_label_tensor =  self.model(images)
        orig_label = orig_label_tensor.cpu().max(1, keepdim=False)[1]

        # Get targeted labels
        if self.targeted and self.compare_orig == False:
            targeted_labels = self.get_targeted_labels(true_labels)
        elif self.targeted and self.compare_orig:
            targeted_labels = self.get_targeted_labels(orig_label)
        else:
            targeted_labels = None
        # print("targeted label is ",targeted_labels)
        # print("true label is ", true_labels)
        # print("original label is", orig_label)
        # print(" orig_label_tensor", orig_label_tensor)
        # print(int(orig_label_tensor.max(1, keepdim=False)[1][0]))
              
        # Iterate for all steps
        for step in range(self.steps):
            # Predict label of current image
            images.requires_grad = True
            outputs = self.model(images)

            # Get cost for current preducted outputs and targeted labels or true labels
            try:
                if self.targeted:
                    cost = -loss(outputs, targeted_labels)
                else:
                    cost = loss(outputs, labels)
            except:
                if self.targeted:
                    cost = -loss(outputs.cuda(), targeted_labels.cuda())
                else:
                    cost = loss(outputs, labels)
            # Compute gradient, take sign and rescale to mean and std of data
            gradients = torch.autograd.grad(cost, images)[0]
            gradients_sign = self.rescale_gradients(gradients.sign())

            # Attack image and clamp it
            try:
                attacked_images[samples_not_done,:,:,:] = images[samples_not_done,:,:,:] + (self.epsilon/self.steps)*gradients_sign[samples_not_done,:,:,:]
            except:
                attacked_images[samples_not_done,:,:,:] = images[samples_not_done,:,:,:].cpu() + (self.epsilon/self.steps)*gradients_sign[samples_not_done,:,:,:].cpu()
            attacked_images = attacked_images.detach()
            attacked_images = self.clamp(attacked_images)
            attacked_images_quantized = self.quantize(attacked_images.clone())

            # Save results of epsilon and number of steps
            epsilons[samples_not_done] += (self.epsilon/self.steps)
            steps[samples_not_done] += 1
            try:
                gradients_sum[samples_not_done,:,:,:] = gradients_sum[samples_not_done,:,:,:] + gradients_sign[samples_not_done,:,:,:]
            except:
                gradients_sum[samples_not_done,:,:,:] = gradients_sum[samples_not_done,:,:,:].cpu() + gradients_sign[samples_not_done,:,:,:].cpu()

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
                try:
                    if self.compare_orig:
                        samples_not_done = torch.where(attacked_labels==orig_label)[0]
                    elif self.compare_orig == False:
                        samples_not_done = torch.where(attacked_labels==true_labels)[0]
                except:
                    if self.compare_orig:
                        samples_not_done = torch.where(attacked_labels.cpu()==orig_label.cpu())[0]
                    elif self.compare_orig == False:
                        samples_not_done = torch.where(attacked_labels.cpu()==true_labels.cpu())[0]
            # Break of no samples left to attack
            if samples_not_done.shape[0] <= 0:
                break

        #  Normalize gradients and clamp them
        for i, step in enumerate(steps):
            gradients_sum[i] = gradients_sum[i]/step
        gradients_sign = self.clamp(gradients_sum)

        # Return results
        return attacked_images_quantized, attacked_labels, gradients_sign, targeted_labels, similarities, epsilons, steps
    
def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)


class DeepFool(Attack):
    def __init__(self, model, mean, std, max_iter = 10, overshoot=0.02,num_classes = 3, min_ssim = 0.995, restrict_iter = False, restrict_ssim = True ):
        super().__init__('FGSM', model, mean, std)
        self.max_iter = max_iter
        self.overshoot = overshoot
        self.num_classes = num_classes
        self.min_ssim = min_ssim
        self.restrict_iter = restrict_iter
        self.restrict_ssim = restrict_ssim
    def attack(self, images, labels):
        #intialize
        original_images = images.clone().detach()
        label_arr = []
        k_i_arr = []
        ssim_arr = []
        i_arr = []
        count = 0
        fool_count = 0
        attacked_arr = []
        ori_arr = []
        #load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.model
        net = model.eval()
        # output transform
        in_transform = transforms.Compose([
                        transforms.Scale(400),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = self.mean, std = self.std)
                        ])
        clip = lambda x: clip_tensor(x, 0, 255)
        out_transform = transforms.Compose([
                        transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, self.std))),
                        transforms.Normalize(mean=list(map(lambda x: -x, self.mean)), std=[1, 1, 1]),
                        transforms.Lambda(clip),
                        transforms.ToPILImage(),
                        ])
        #handle restrictions
        if self.restrict_iter == False: self.max_iter = np.inf
        if self.restrict_ssim == False: self.min_ssim = 0
        # attack
        
        # batch: int
        # labels: tensor([int])
        #image = images['image'] # image: tensor([[[...]]]) shape: 1, 3, 400, 400
        image = torch.squeeze(images) # squeeze to 3*400*400
        
        images = images.clone().detach().to(self.device)
        original_images = images.clone().detach()
        
        ori_arr.append(image)
        img_rt = out_transform(image) #a copy of unattacked image
        try:
            f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.numpy().flatten()
        except:
            f_image = net.forward(Variable(image[None, :, :, :].cpu(), requires_grad=True)).cpu().data.numpy().flatten()
        I = (np.array(f_image)).flatten().argsort()[::-1]
        I = I[0:self.num_classes]
        label = I[0]
        input_shape = image.numpy()[0].shape
        pert_image = copy.deepcopy(image)
        w = np.zeros(input_shape)
        r_tot = np.zeros(input_shape)
        loop_i = 0
        x = Variable(pert_image[None, :], requires_grad=True)
        fs = net.forward(x)
        fs_list = [fs[0,I[k]] for k in range(self.num_classes)]
        k_i = label
        ssim_value = 1
        while k_i == label and loop_i <self.max_iter:
            pert = np.inf
            fs[0, I[0]].backward(retain_graph=True)
            grad_orig = x.grad.data.numpy().copy()
            for k in range(1, self.num_classes):
                zero_gradients(x)
                fs[0, I[k]].backward(retain_graph=True)
                cur_grad = x.grad.data.numpy().copy()
                # set new w_k and new f_k
                w_k = cur_grad - grad_orig
                try:
                    f_k = (fs[0, I[k]] - fs[0, I[0]]).data.numpy()
                except:
                    f_k = (fs[0, I[k]] - fs[0, I[0]]).cpu().data.numpy()
                pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())
                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k
                    w = w_k
            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i =  (pert+1e-4) * w / np.linalg.norm(w)
            r_tot_tmp = np.float32(r_tot + r_i)
            # compute ssim
            pert_image_tmp = image + (1+self.overshoot)*torch.from_numpy(r_tot_tmp)
            pert_image_tmp = out_transform(pert_image_tmp[0])
            pert_image_tmp = im.fromarray((np.array(pert_image_tmp)), 'RGB')
            #changing CHECK!!!
            #pert_image_tmp.save('tmp.png')
            #pert_image_quantized = in_transform(im.open('tmp.png'))
            #NEW ONE
            pert_image_quantized = in_transform(pert_image_tmp)            
            
            #pert_image_quantized_rt = out_transform(pert_image_quantized)
            pert_image_quantized_rt = pert_image_tmp
            ssim_tmp = ssim(np.array(img_rt), np.array(pert_image_quantized_rt), data_range = np.array(img_rt).max() - np.array(img_rt).min(), multichannel=True)
            if ssim_tmp > self.min_ssim: 
                ssim_value = ssim_tmp
                r_tot = np.float32(r_tot + r_i)
                pert_image = image + (1+self.overshoot)*torch.from_numpy(r_tot)
                x = Variable(pert_image_quantized[None, :, :, :], requires_grad=True)
                #x = Variable(pert_image, requires_grad=True)
                fs = net.forward(x)
                try:
                    k_i = np.argmax(fs.data.numpy().flatten())
                except:
                    k_i = np.argmax(fs.cpu().data.numpy().flatten())
                loop_i += 1
            else: 
                #print(ssim_tmp)
                r_tot = np.float32(r_tot + r_i * 0)
                #pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)
                pert_image_tmp = image + (1+self.overshoot)*torch.from_numpy(r_tot)
                pert_image_tmp = out_transform(pert_image_tmp[0])
                pert_image_tmp = im.fromarray((np.array(pert_image_tmp)), 'RGB')
                #CHANGE!
                #pert_image_tmp.save('tmp.png')
                #pert_image_quantized = in_transform(im.open('tmp.png'))
                pert_image_quantized = in_transform(pert_image_tmp)                
                #pert_image_quantized_rt = out_transform(pert_image_quantized)
                pert_image_quantized_rt = pert_image_tmp
                break
        r_tot = (1+self.overshoot)*r_tot
        # process output
        #pert_image_rt = np.array(out_transform(pert_image[0]))
        attacked_arr.append(pert_image_quantized)#[0])
        r_rt = np.array(pert_image_quantized_rt) - np.array(img_rt)
        # return loop_i, label, k_i, pert_image_rt, img_rt, r_rt, ssim
        # save outputs 
        label_arr.append(label)
        k_i_arr.append(k_i)
        similarities = self.ssim(original_images, pert_image_quantized)
        # print labels
        # print('###')

        # print('original label: ', label)
        # print('new label: ', k_i)
        # print('ssim: ', ssim_value)
        # print('loop num: ', loop_i)
        ###
        #x = Variable(pert_image_quantized[None, :, :, :], requires_grad=True)
        #fs = net.forward(x)
        #k_i = np.argmax(fs.data.numpy().flatten())
        #print('test: ', k_i)
        #img = pert_image_quantized.unsqueeze(0)
        #net.to(device)
        #img = img.to(device)
        #out = net(img)
        #preds = F.softmax(out, dim=1)
        #prod, index = torch.max(preds, 1)
        #print('test: ', index)

        # save images
        # pert_file = save_path + 'pert_' + str(batch) + '.png'
        # Image.fromarray((np.array(pert_image_quantized_rt)), 'RGB').save(pert_file)
        # ori_file = save_path + 'ori_' + str(batch) + '.png'
        # Image.fromarray((np.array(img_rt)), 'RGB').save(ori_file)
        # r_file = save_path + 'r_' + str(batch) + '.png'
        # Image.fromarray((np.array(r_rt)), 'RGB').save(r_file)
        # if label != k_i: fool_count = fool_count + 1
        # count = count + 1
        # ssim_arr.append(ssim)
        # i_arr.append(loop_i)
        #break
        # print('###')
        # print('total number of attacked images: ', count)
        # print("total number of fooled classifications: ", fool_count)
        # print('fool ratio: ', fool_count/count)
        # print('mean ssim: ', np.mean(np.array(ssim_arr)))
        # print('std ssim: ', np.std(np.array(ssim_arr)))
        # print('avg loop num: ', np.mean(np.array(i_arr)))
        return pert_image_quantized, k_i, similarities
        
        



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
    try:
        print(out.format('Accuracy:',
                     str(np.around(np.count_nonzero(attacked_labels==labels)*100/num_samples, 2))+str('%')))
    except:
        print(out.format('Accuracy:',
                     str(np.around(np.count_nonzero(attacked_labels.cpu()==labels.cpu())*100/num_samples, 2))+str('%')))   
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
        try:
            print(out.format('Target accuracy:',
                             str(np.around(np.count_nonzero(attacked_labels==targeted_labels)*100/num_samples, 2))+str('%')))
        except:
            print(out.format('Target accuracy:',
                             str(np.around(np.count_nonzero(attacked_labels.cpu()==targeted_labels.cpu())*100/num_samples, 2))+str('%')))    



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
        img = torch.tensor(convert_image(img, mean, std)).permute(2,0,1).type(torch.float)/255
        att_img = torch.tensor(convert_image(att_img, mean, std)).permute(2,0,1).type(torch.float)/255
        grad = torch.tensor(convert_image(grad, mean, std)).permute(2,0,1).type(torch.float)/255
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

def attack_experiment_ssim(rows, LOAD_PATH, IMAGE_ROOT, PARTITION_PATH_ROOT, album_compose, BATCH_SIZE, SHUFFLE,ATTACK_METHOD,  data_mean, data_std, 
                      EPSILON, TARGETED, STEPS, MAX_BATCHES,COMPARE_ORIG, MAX_ITER = 10, THRESHOLD = None,MIN_SSIM = None, UNTIL_ATTACKED = True, p_str = None, p_end = None, OVERSHOOT = 0.02, 
                      RESTRICT_ITER = False, RESTRICT_SSIM = True ):
    # INPUTS:
    # rows: number of rows to show attacked images
    # LOAD_PATH: model path to be loaded
    # IMAGE_ROOT, PARTITION_PATH_ROOT, album_compose : required for loading dataset from csv
    # BATCH_SIZE, SHUFFLE: required for dataloader function.
    # data_mean, data_std, EPSILON, TARGETED, STEPS : parameters for attack algorithms. STEPS is only used in I-FGSM
    # MAX_BATCHES: maximum number of batches to be used as limit
    # THRESHOLD: Set to None if no threshold
    # UNTIL_ATTACKED:If to attack until misclassified
    # p_str = The image position that starts plotting Default is None. If filled, plot_end must be filled too.
    # p_end = The image position that ends plotting. Default is None. If filled, plot_start must be filled too.
    
    # Load trained model
    model = torch.load(LOAD_PATH)

    if torch.cuda.is_available():
        model.cuda()

    # Load test data to attack
    dataset_attack = LoadDatasetFromCSV(image_root=IMAGE_ROOT,
                                        csv_path=PARTITION_PATH_ROOT+'data_labels_test.csv',
                                        transforms=album_compose)

    # Load data into loaders
    dataloader_attack = DataLoader(dataset=dataset_attack, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    # Initialize and choose attacker
    if ATTACK_METHOD == 'FGSM':
        attack_method = FGSM(model=model, mean=data_mean, std=data_std,
                                    epsilon=EPSILON, targeted=TARGETED)

    elif ATTACK_METHOD == 'IFGSM':
        attack_method = IFGSM(model= model, mean=data_mean, std=data_std,
                                     epsilon=EPSILON, steps=STEPS, targeted=TARGETED,
                                     threshold=THRESHOLD, until_attacked=UNTIL_ATTACKED, compare_orig = COMPARE_ORIG)
    elif ATTACK_METHOD == "DeepFool":
        attack_method = DeepFool(model = model, mean = data_mean, std = data_std, max_iter = MAX_ITER, min_ssim = MIN_SSIM,
                                 overshoot = OVERSHOOT, restrict_iter = RESTRICT_ITER, 
                                 restrict_ssim = RESTRICT_SSIM)

    if ATTACK_METHOD == 'FGSM' or ATTACK_METHOD == 'IFGSM':
        
        # Intialize reults arrays
        all_images = torch.Tensor()
        all_labels = torch.Tensor().type(dtype=torch.uint8)
        all_init_labels = torch.Tensor().type(dtype=torch.uint8)
        all_attacked_images = torch.Tensor()
        all_attacked_labels = torch.Tensor().type(dtype=torch.uint8)
        all_gradients = torch.Tensor()
        all_epsilons = torch.Tensor()
        all_steps = torch.Tensor()
    
        if TARGETED:
            all_targeted_labels = torch.Tensor().type(dtype=torch.uint8)
        else:
            all_targeted_labels = None
        all_similarities = torch.Tensor()
    
        
        # Iterate over each batch
        for batch, (images, labels) in enumerate(dataloader_attack):
    
            images = images['image']
            init_labels = model(images).max(1, keepdim=False)[1]
    
            # Attack images
            attacked_images, attacked_labels, gradients, targeted_labels, similarities, epsilons, steps = \
            attack_method.attack(images, labels)
    
    
    
            # Concatenate results for each batch
            all_images = torch.cat((all_images.cuda(), images.cuda()), dim=0)
            all_labels = torch.cat((all_labels.cuda(), labels.cuda()), dim=0)
            all_init_labels = torch.cat((all_init_labels.cuda(), labels.cuda()), dim=0)
            all_attacked_images = torch.cat((all_attacked_images.cuda(), attacked_images.cuda()), dim=0)
            all_attacked_labels = torch.cat((all_attacked_labels.cuda(), attacked_labels.cuda()), dim=0)
            all_gradients = torch.cat((all_gradients.cuda(), gradients.cuda()), dim=0)
            all_epsilons = torch.cat((all_epsilons.cuda(), epsilons.cuda()), dim=0)
            all_steps = torch.cat((all_steps.cuda(), steps.cuda()), dim=0)
            if TARGETED:
                all_targeted_labels = torch.cat((all_targeted_labels.cuda(), targeted_labels.cuda()), dim=0)
            all_similarities = torch.cat((all_similarities, similarities), dim=0)
    
            # Stop if above max amount of batches to test for
            if MAX_BATCHES is None:
                MAX_BATCHES = len(dataloader_attack)
            if batch >= MAX_BATCHES - 1:
                break
        print('Experiment is done with epsilon = ' + str(EPSILON) + " and steps = " + str(STEPS))
    
    
    
        # Evaluate attacks
        num_samples = all_labels.shape[0]
        Accuracy = np.around(np.count_nonzero(all_attacked_labels.cpu()==all_labels.cpu())*100/num_samples, 4)
        Target_Accuracy = np.around(np.count_nonzero(all_attacked_labels.cpu()==all_targeted_labels.cpu())*100/num_samples, 4)
        mean_similarity = torch.mean(all_similarities).item()
        std_similarity = torch.std(all_similarities).item()
        mean_steps = torch.mean(all_steps).item()
        mean_epsilons = torch.mean(all_epsilons).item()
        
        #checks if a specific interval for plotting has been chosen. If it is, only plot these images in the interval.
        # if p_str != None and p_end != None:
        #     plot_attacks(rows, all_images[p_str:p_end], all_labels[p_str:p_end], all_init_labels[p_str:p_end], 
        #                  all_attacked_images[p_str:p_end], all_attacked_labels[p_str:p_end], all_gradients[p_str:p_end],
        #         all_epsilons[p_str:p_end], all_steps[p_str:p_end], mean=data_mean, std=data_std, targeted_labels=all_targeted_labels[p_str:p_end])
        # #plot all images of amount defined as rows    
        # else:
        #     plot_attacks(rows, all_images, all_labels, all_init_labels, all_attacked_images, all_attacked_labels, all_gradients,
        #              all_epsilons, all_steps, mean=data_mean, std=data_std, targeted_labels=all_targeted_labels)
        # plt.show()
        #deletes all the objects to create memory on next run
        del all_images,all_labels,all_init_labels,all_attacked_images, all_attacked_labels,all_gradients, \
        all_epsilons,all_steps, all_targeted_labels, all_similarities
        
        gc.collect()
        
        return Accuracy, Target_Accuracy, mean_similarity, std_similarity, mean_steps, mean_epsilons
    elif ATTACK_METHOD == "DeepFool":
        # Intialize reults arrays
        all_images = torch.Tensor()
        all_labels = torch.Tensor().type(dtype=torch.uint8)
        all_init_labels = torch.Tensor().type(dtype=torch.uint8)
        all_attacked_images = torch.Tensor()
        all_attacked_labels = torch.Tensor().type(dtype=torch.uint8)
        #all_gradients = torch.Tensor()
        #all_epsilons = torch.Tensor()
        #all_steps = torch.Tensor()
    
        # if TARGETED:
        #     all_targeted_labels = torch.Tensor().type(dtype=torch.uint8)
        # else:
        #     all_targeted_labels = None
        all_similarities = torch.Tensor()
    
        
        # Iterate over each batch
        for batch, (images, labels) in enumerate(dataloader_attack):
    
            images = images['image']
            init_labels = model(images).max(1, keepdim=False)[1]
    
            # Attack images
            attacked_images, attacked_labels,similarities= attack_method.attack(images, labels)
    
    
            # Concatenate results for each batch
            all_images = torch.cat((all_images.cuda(), images.cuda()), dim=0)
            all_labels = torch.cat((all_labels.cuda(), labels.cuda()), dim=0)
            all_init_labels = torch.cat((all_init_labels.cuda(), labels.cuda()), dim=0)
            all_attacked_images = torch.cat((all_attacked_images.cuda(), attacked_images.cuda()), dim=0)
            all_attacked_labels = torch.cat((all_attacked_labels.cuda(), torch.tensor([attacked_labels]).cuda()), dim=0)
            # all_gradients = torch.cat((all_gradients.cuda(), gradients.cuda()), dim=0)
            # all_epsilons = torch.cat((all_epsilons.cuda(), epsilons.cuda()), dim=0)
            # all_steps = torch.cat((all_steps.cuda(), steps.cuda()), dim=0)
            # if TARGETED:
            #     all_targeted_labels = torch.cat((all_targeted_labels.cuda(), targeted_labels.cuda()), dim=0)
            all_similarities = torch.cat((all_similarities, similarities), dim=0)
    
            # Stop if above max amount of batches to test for
            if MAX_BATCHES is None:
                MAX_BATCHES = len(dataloader_attack)
            if batch >= MAX_BATCHES - 1:
                break
        print('Experiment is done with Overshoot = ', str(OVERSHOOT) )
    
    
    
        # Evaluate attacks
        num_samples = all_labels.shape[0]
        Accuracy = np.around(np.count_nonzero(all_attacked_labels.cpu()==all_labels.cpu())*100/num_samples, 4)
        #Target_Accuracy = np.around(np.count_nonzero(all_attacked_labels.cpu()==all_targeted_labels.cpu())*100/num_samples, 4)
        mean_similarity = torch.mean(all_similarities).item()
        std_similarity = torch.std(all_similarities).item()
        # mean_steps = torch.mean(all_steps).item()
        # mean_epsilons = torch.mean(all_epsilons).item()
        
        #checks if a specific interval for plotting has been chosen. If it is, only plot these images in the interval.
        # if p_str != None and p_end != None:
        #     plot_attacks(rows, all_images[p_str:p_end], all_labels[p_str:p_end], all_init_labels[p_str:p_end], 
        #                  all_attacked_images[p_str:p_end], all_attacked_labels[p_str:p_end], all_gradients[p_str:p_end],
        #         all_epsilons[p_str:p_end], all_steps[p_str:p_end], mean=data_mean, std=data_std, targeted_labels=all_targeted_labels[p_str:p_end])
        # #plot all images of amount defined as rows    
        # else:
        #     plot_attacks(rows, all_images, all_labels, all_init_labels, all_attacked_images, all_attacked_labels, all_gradients,
        #              all_epsilons, all_steps, mean=data_mean, std=data_std, targeted_labels=all_targeted_labels)
        # plt.show()
        #deletes all the objects to create memory on next run
        del all_images,all_labels,all_init_labels,all_attacked_images, all_attacked_labels, all_similarities
        
        gc.collect()
        
        return Accuracy,mean_similarity, std_similarity