# https://github.com/eriklindernoren/Keras-GAN#pix2pix
# https://keras.io/api/layers/regularization_layers/dropout/
# https://stackoverflow.com/questions/51632716/keras-concatenate-layers-difference-between-different-types-of-concatenate-func

# unwrapped phase 1 -> unwrapped phase 2
# unwrapped phase 1 -> index
# wrapped phase 1 -> wrapped phase 2
# wrapped phase 1 -> index
# hologram 1 + hologram 2 -> height + index

# may change optimizer, LeakyReLU, BatchNormalization, activation

from data_loader import DataLoader
from keras.layers import Input, Dropout, Concatenate
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam
# this is for tensorflow 2.x. Delete the prefix tensorflow if using tensorflow 1.x

import datetime
import numpy as np
from skimage import io

import os

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' #show GPUs in computer
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' #specify which GPUs to be used


class Pix2Pix:
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'decoupled_images_512'
        
        self.input_name = 'phantom_phase_488nm'         # B
        self.scaleB = 10**6
        self.maxB = 12
        
        self.output_name = 'phantom_phase_505nm'        # A
        self.scaleA = 10**6
        self.maxA = 14

        self.numTotal = 512                 # total number of images
        self.numTrain = 412                 # total number of images for training
        # 100 images for testing
        
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                input_name=self.input_name, scaleB=self.scaleB, maxB=self.maxB,
                                output_name=self.output_name, scaleA=self.scaleA, maxA=self.maxA, 
                                numTrain=self.numTrain,
                                img_res=(self.img_rows, self.img_cols))
        
        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)
        # our discriminator uses a convolutional PatchGAN classifier, which only penalizes structure at the scale of image patches

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)  # learning_rate, beta_1

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)
        # https://ai.stackexchange.com/questions/25822/in-this-implementation-of-pix2pix-why-are-the-weights-for-the-discriminator-and/26021
        # using L1 distance rather than L2 as L1 encourage less blurring

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)  # default = 0.3
            if bn:
                d = BatchNormalization(momentum=0.8)(d)  # default = 0.99
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, use_dropout=False):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if use_dropout:
                u = Dropout(0.5)(u)  # dropout rate = 50%
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)

        # Upsampling
        u1 = deconv2d(d6, d5, self.gf*8)
        u2 = deconv2d(u1, d4, self.gf*8)
        u3 = deconv2d(u2, d3, self.gf*4)
        u4 = deconv2d(u3, d2, self.gf*2)
        u5 = deconv2d(u4, d1, self.gf)

        u6 = UpSampling2D(size=2)(u5)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u6)
        
        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)  # activation = 'sigmoid'


        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):

            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size=batch_size)):

                imgs_A = np.reshape(imgs_A, np.shape(imgs_A)+(1,)) if (self.channels == 1) else np.array(imgs_A)
                imgs_B = np.reshape(imgs_B, np.shape(imgs_B)+(1,)) if (self.channels == 1) else np.array(imgs_B)
                # img shape = (1,target_size,1)
                
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)

                # Train the discriminators (original images = real / generated = fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)  # (d_loss_real+d_loss_fake)/2

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch+1, epochs,
                                                                        batch_i+1, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))

            print("--------------")
            if (epoch+1) % sample_interval == 0:
                self.test(name=str(epoch+1))

# Keras Deep Learning library includes 3 separate functions that can be used to train the models
# .fit: -> .predict
    # entire training dataset can fit into RAM
    # there is no data augmentation going on -> no need for Keras generators
    # tensorflow 2.2.0 supports data augmentation
# .fit_generator: -> .predict_generator
    # real-word datasets are often too large to fit into memory
    # require us to perform data augmentation to avoid overfitting and increase the ability of the model to generalize
# .train_on_batch:
    # accept a single batch of data, perform backpropagation, and then update the model parameters
# https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/

    def test(self, name=''):
        ROOT_PATH = os.getcwd()

        folder_name = 'predict_' + name + 'epochs'
        num_test_img = self.numTotal - self.numTrain        # total number of images for testing

        if not os.path.exists(folder_name):
            path = os.path.join(ROOT_PATH, folder_name)
            os.mkdir(path)

        imgs_A, imgs_B = self.data_loader.load_data(is_testing=True)
        
        for i, item in enumerate(imgs_B):
            img_B = np.reshape(item, np.shape(item)+(1,)) if (self.channels == 1) else item
            img_B = np.reshape(img_B, (1,)+img_B.shape)  # img shape = (1,target_size,1)
            fake_A = self.generator.predict(img_B)

            fake_A = np.array(fake_A) + 1.
            max_A = self.maxA / 2
            fake_A = np.array(fake_A)*max_A

            io.imsave(os.path.join(folder_name, '%d_predict_%s.tif' % (self.numTotal-num_test_img+1+i, self.output_name)),
                      fake_A,
                      check_contrast=False)


if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train(epochs=200, batch_size=1, sample_interval=50)  # set sample_interval = epochs to run one-time testing
    