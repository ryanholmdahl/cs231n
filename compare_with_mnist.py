from tensorflow.examples.tutorials.mnist import input_data
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt
import numpy as np
import os

target_dir = "recon_dual_weakim_weakgauss_outputs"
target_iter = "340"

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_images = mnist.train.images.reshape((-1, 28, 28))
dir_path = os.path.join(target_dir, target_iter)
for filename in os.listdir(dir_path):
    im = imread(os.path.join(dir_path, filename))
    min_loss = float('inf')
    best_im = None
    for i in range(len(train_images)):
        l2 = np.sum((im - train_images[i, :, :]*256)**2)
        if l2 < min_loss:
            min_loss = l2
            best_im = train_images[i,:,:]
    imsave(os.path.join(dir_path,(filename.split(".")[0])+"_closest.png"), best_im)
