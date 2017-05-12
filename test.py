import tensorflow as tf
import numpy as np
from baseline_model import BaselineModel

params = {}
params['lr'] = 1e-4
params['lr_decay'] = 1.
params['decay_steps'] = 1000
params['width'] = 16
params['height'] = 16
params['channels'] = 3
params['in_conv_layers'] = 1
params['in_conv_filters'] = [1]
params['in_conv_dim'] = [5]
params['in_conv_stride'] = [2]
params['fc_layers'] = 1
params['fc_dim'] = [128]
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
        return x, y

m = BaselineModel(params)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    m.train(10000, TestGenerator(16, 16, 3, 16), TestGenerator(16, 16, 3, 16))

