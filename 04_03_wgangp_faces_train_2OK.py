#!/usr/bin/env python
# coding: utf-8

# # WGAN-GP Training

# ## imports

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')

import os, shutil
import matplotlib.pyplot as plt
import numpy as np

from models.WGANGP import WGANGP
from utils.loaders_a import load_celeb
from utils.custom_utils import timer

import pickle


# In[2]:


# run params
SECTION = 'gan'
RUN_ID = '0003'
DATA_NAME = 'celeba_200k'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

    
DS_path = 's:\_ML\Datasets'    

mode =  'load' #'build' #'load' #


# ## data

# In[3]:


BATCH_SIZE = 64
IMAGE_SIZE = 64


# In[4]:


x_train = load_celeb(DATA_NAME, IMAGE_SIZE, BATCH_SIZE, DS_path)


# In[5]:


x_train[0][0][0]


# In[6]:


plt.imshow((x_train[0][0][0]+1)/2)


# ## architecture

# In[7]:


gan = WGANGP(input_dim = (IMAGE_SIZE,IMAGE_SIZE,3)
        , critic_conv_filters = [64,128,256,512]
        , critic_conv_kernel_size = [5,5,5,5]
        , critic_conv_strides = [2,2,2,2]
        , critic_batch_norm_momentum = None
        , critic_activation = 'leaky_relu'
        , critic_dropout_rate = None
        , critic_learning_rate = 0.0002
        , generator_initial_dense_layer_size = (4, 4, 512)
        , generator_upsample = [1,1,1,1]
        , generator_conv_filters = [256,128,64,3]
        , generator_conv_kernel_size = [5,5,5,5]
        , generator_conv_strides = [2,2,2,2]
        , generator_batch_norm_momentum = 0.9
        , generator_activation = 'leaky_relu'
        , generator_dropout_rate = None
        , generator_learning_rate = 0.0002
        , optimiser = 'adam'
        , grad_weight = 10
        , z_dim = 100
        , batch_size = BATCH_SIZE
        )

if mode == 'build':
    gan.save(RUN_FOLDER)

else:
    #gan.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))
    gan.load_weights(os.path.join(RUN_FOLDER, 'weights/weights_.h5'))
    print("wights loaded: ", os.path.join(RUN_FOLDER, 'weights/weights_'))


# In[8]:


gan.critic.summary()


# In[9]:


gan.generator.summary()


# ## training

# In[10]:


EPOCHS = 10 #6000
PRINT_EVERY_N_BATCHES = 5
N_CRITIC = 5
BATCH_SIZE = 64


# In[11]:


gan.train = timer(gan.train)  # timer decorator start


gan.train(     
    x_train
    , batch_size = BATCH_SIZE
    , epochs = EPOCHS
    , run_folder = RUN_FOLDER
    , print_every_n_batches = PRINT_EVERY_N_BATCHES
    , n_critic = N_CRITIC
    , using_generator = True
)


# In[12]:


fig = plt.figure()
plt.plot([x[0] for x in gan.d_losses], color='black', linewidth=0.25)

plt.plot([x[1] for x in gan.d_losses], color='green', linewidth=0.25)
plt.plot([x[2] for x in gan.d_losses], color='red', linewidth=0.25)
plt.plot(gan.g_losses, color='orange', linewidth=0.25)

plt.xlabel('batch', fontsize=18)
plt.ylabel('loss', fontsize=16)

plt.xlim(0, 2000)
# plt.ylim(0, 2)

plt.show()


# In[13]:


# Copy weights
shutil.copy(os.path.join(RUN_FOLDER, 'weights\\weights.h5'), os.path.join(RUN_FOLDER, 'weights\\weights_.h5'))


# In[14]:


# import tensorflow as tf
# tf.keras.backend.clear_session()


# # GAN images

# In[21]:

'''
def sample_gan_imgs(range=5, name='custom'):
    """save sampled gan images
    (range=5, name='custom')
    """
    for id in range(range):
        noise = np.random.normal(0, 1, (1, gan.z_dim))

        gen_imgs = gan.generator.predict(noise)
        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(np.squeeze(gen_imgs[0, :,:,:]), cmap = 'gray_r')
        plt.savefig(os.path.join(RUN_FOLDER, "images/" + name + "000%s.png" % id))

'''