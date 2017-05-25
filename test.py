import pickle

import numpy as np
import tensorflow as tf

from model_builder import ModularModel

params = {}
params['lr'] = 1e-3
params['lr_decay'] = 0.96
params['lr_decay_steps'] = 100
params['im_width'] = 32
params['im_height'] = 32
params['im_channels'] = 3
params['in_conv_layers'] = 1
params['in_conv_filters'] = [10]
params['in_conv_dim'] = [5]
params['in_conv_stride'] = [2]
params['fc_layers'] = 2
params['fc_dim'] = [256, 256]
params['embed_channels'] = 100
params['out_conv_layers'] = 2
params['out_conv_filters'] = [8, 3]
params['out_conv_dim'] = [5, 5]
params['out_conv_stride'] = [1, 1]
params["model_name"] = "test"
params["ckpt_path"] = "ckpt"
params["log_path"] = "log"
params["n_epochs"] = 10
params["n_eval_batches"] = 1
params["batch_size"] = 1
params["use_transpose"] = False

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

train_examples = [np.array([np.zeros((32,32,3)) for _ in range(10)]), np.array([np.zeros((32,32,3)) for _ in range(10)])]
valid_examples = train_examples

m = ModularModel(params)
m.build()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    m.fit(sess, None, train_examples, valid_examples)
    # saver = tf.train.Saver()
    # saver.save(sess, "ckpt/overfit")
    # for i in range(len(train_tuples)):
    #     show_image_example(sess, m, train_tuples[i][0], train_tuples[i][1], "figs/figt{}.png".format(i))
    # for i in range(len(valid_tuples)):
    #     show_image_example(sess, m, valid_tuples[i][0], valid_tuples[i][1], "figs/figv{}.png".format(i))

