from lfw.dataset_builder import Dataset
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt
import numpy as np
import os

target_dir = "outputs\\100style_0image_5gauss_5train_gaussianlabels_decoder_5gauss_1image_5dropout_lfw_tanhnoscale_inconv_1000_1005"
target_iter = "4065"

d = Dataset((32, 32, 1))
d.read_samples('lfw/lfw_data')
train_images = np.concatenate((d.train_examples[0], d.dev_examples[0], d.test_examples[0]), axis=0)
dir_path = os.path.join(target_dir, target_iter)
for filename in os.listdir(dir_path):
    if "_closest" in filename:
        continue
    im = imread(os.path.join(dir_path, filename))/255.0
    min_loss = float('inf')
    best_im = None
    for i in range(len(train_images)):
        l2 = np.sum((im - train_images[i, :, :, 0])**2)
        if l2 < min_loss:
            min_loss = l2
            best_im = train_images[i, :, :, 0]
    print(min_loss)
    imsave(os.path.join(dir_path,(filename.split(".")[0])+"_closest.png"), best_im)
