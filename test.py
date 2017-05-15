import tensorflow as tf
import numpy as np
from util import show_image_example
from baseline_models import Conv2ConvModel, Affine2AffineModel
import pickle

params = {}
params['lr'] = 1e-3
params['lr_decay'] = .96
params['lr_decay_steps'] = 40
params['im_width'] = 16
params['im_height'] = 16
params['im_channels'] = 3
params['in_conv_layers'] = 1
params['in_conv_filters'] = [1]
params['in_conv_dim'] = [5]
params['in_conv_stride'] = [2]
params['fc_layers'] = 2
params['fc_dim'] = [256, 256]
params['embed_channels'] = 100
params['out_conv_layers'] = 2
params['out_conv_filters'] = [8, 16]
params['out_conv_dim'] = [5, 5]
params['out_conv_stride'] = [1, 1]
params['output_conv_dim'] = 5
params['output_conv_stride'] = 1
params["model_name"] = "test"
params["ckpt_path"] = "ckpt"
params["log_path"] = "log"
params["n_epochs"] = 10000
params["n_eval_batches"] = 10
params["batch_size"] = 4

class TestGenerator:
    def __init__(self, w, h, c, b):
        self.w = w
        self.h = h
        self.c = c
        self.b = b

    def next(self):
        x = []
        y = []
        for _ in range(self.b):
            x_sample = np.zeros((self.h, self.w, self.c))
            y_sample = x_sample+1
            x.append(x_sample)
            y.append(y_sample)
            if self.b == 1:
                return x_sample, y_sample
        return x, y


train_tuples, valid_tuples, test_tuples = pickle.load(open("data/data (1).p", "rb"), encoding='latin1')
train_examples = [np.array([np.array(tup[0]) for tup in train_tuples]), np.array([np.array(tup[1]) for tup in train_tuples])]
valid_examples = [np.array([np.array(tup[0]) for tup in valid_tuples]), np.array([np.array(tup[1]) for tup in valid_tuples])]


m = Affine2AffineModel(params)
m.build()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    m.fit(sess, None, train_examples, valid_examples)
    show_image_example(sess, m, train_tuples[0][0], train_tuples[0][1])

