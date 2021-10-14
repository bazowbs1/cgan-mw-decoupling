# https://github.com/eriklindernoren/Keras-GAN#pix2pix

from glob import glob  # path name
import numpy as np
import skimage.transform
import imageio
import re


class DataLoader:
    def __init__(self, dataset_name, 
                 input_name, scaleB, maxB,
                 output_name, scaleA, maxA, 
                 numTrain, img_res=(128, 128)):
        self.dataset_name = dataset_name
        
        self.input_name = input_name        # B
        self.scaleB = scaleB
        self.maxB = maxB
        
        self.output_name = output_name      # A
        self.scaleA = scaleA
        self.maxA = maxA
        
        self.numTrain = numTrain            # total number of images for training

        self.img_res = img_res

        self.path_A_total = sorted(glob('./%s/%s_*.*' % (self.dataset_name, self.output_name)),
                                key=lambda f: int(re.search(self.output_name+'_(.*).tif', f).group(1)))

        self.path_B_total = sorted(glob('./%s/%s_*.*' % (self.dataset_name, self.input_name)),
                                key=lambda f: int(re.search(self.input_name+'_(.*).tif', f).group(1)))

    def load_data(self, batch_size=1, is_testing=False):
        
        if not is_testing:  # training
            path_A = self.path_A_total[0:self.numTrain]
            path_B = self.path_B_total[0:self.numTrain]
        else:  # testing
            path_A = self.path_A_total[self.numTrain:]
            path_B = self.path_B_total[self.numTrain:]
        
        imgs_A = []  # output
        imgs_B = []  # input
        
        for i in range(len(path_A)):
            
            img_A = self.imread(path_A[i])
            img_A = self.processImage(img_A, self.scaleA, self.maxA)
            
            img_B = self.imread(path_B[i])
            img_B = self.processImage(img_B, self.scaleB, self.maxB)
            
            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)
                
            imgs_A.append(img_A)
            imgs_B.append(img_B)
        
        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False):

        if not is_testing:  # training
            path_A = self.path_A_total[0:self.numTrain]
            path_B = self.path_B_total[0:self.numTrain]
        else:  # testing
            path_A = self.path_A_total[self.numTrain:]
            path_B = self.path_B_total[self.numTrain:]
        
        self.n_batches = int(len(path_A)/batch_size)  # len(path_A) = len(path_B)
        
        for i in range(self.n_batches):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            
            for j in range(batch_size):
                
                img_A = self.imread(batch_A[j])
                img_A = self.processImage(img_A, self.scaleA, self.maxA)
                
                img_B = self.imread(batch_B[j])
                img_B = self.processImage(img_B, self.scaleB, self.maxB)
                
                if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)
                
                imgs_A.append(img_A)
                imgs_B.append(img_B)
            
            yield imgs_A, imgs_B
            
            # return: send a specified value back to its caller
            # yeild: produce a sequence of values (iterate over a sequence but dont
            #        want to store the entire sequence in memory)

    def imread(self, path):
        return imageio.imread(path).astype(np.float)  # 32-bit float
    
    def processImage(self, image, scale, maxPixel):
        image = skimage.transform.resize(image, self.img_res)
        image = np.array(image)/scale
        maxPixel = maxPixel/2
        image = np.array(image)/maxPixel - 1.
        return image

# 3 different sets of data
# training data:     is used to optimize the model parameters
# validation data:   is used to make choice about the meta-parameters, e.g. the number of epochs
# testing data:      is used to get a fair estimate of the model performance

# do not need validation data in pix2pix network
# https://nchlis.github.io/2019_11_22/page.html
