from lfw.dataset_builder import Dataset
from scipy.misc import imread, imsave, imresize, toimage
import matplotlib.pyplot as plt
import numpy as np
import os

target_dir = "outputs\\100style_0image_5gauss_5train_gaussianlabels_decoder_5gauss_1image_5dropout_lfw_tanhnoscale_inconv_1000_1005_slow"
target_iter = "2300"

d = Dataset((32, 32, 1), split=[1, 0, 0])
d.read_samples('lfw/lfw_data')
#train_images = np.concatenate((d.train_examples[0], d.dev_examples[0], d.test_examples[0]), axis=0)
train_images = d.train_examples[0]
print(train_images.shape)
dir_path = os.path.join(target_dir, target_iter)
for filename in os.listdir(dir_path):
    if "_closest" in filename:
        continue
    im = imresize(imread(os.path.join(dir_path, filename), mode='L'), d.dims)/255.0
    min_loss = float('inf')
    best_im = None
    for i in range(train_images.shape[0]):
        l2 = np.sum((im - np.squeeze(train_images[i, :, :, :]))**2)
        if l2 < min_loss:
            min_loss = l2
            best_im = train_images[i, :, :, 0]
    print(min_loss)
    im = toimage(best_im, cmin=0, cmax=1)
    im.save(os.path.join(dir_path,(filename.split(".")[0])+"_closest.png"))
