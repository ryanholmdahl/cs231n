import os

import numpy as np
import tensorflow as tf

from layers import unpool


class BaselineModel:
    def __init__(self, params):
        self.model_name = params['model_name']
        self.ckpt_path = params['ckpt_path']
        self.log_path = params['log_path']

        input_dims = (None, params['height'], params['width'], params['channels'])
        self.image_in = tf.placeholder(tf.float32, shape=input_dims)
        prev_output = self.image_in
        for i in range(params['in_conv_layers']):
            prev_output = tf.layers.conv2d(prev_output, params['in_conv_filters'][i], params['in_conv_dim'][i],
                                           strides=params['in_conv_stride'][i], activation=tf.nn.relu,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           name='inconv{}'.format(i))
        prev_output = tf.contrib.layers.flatten(prev_output)
        for i in range(params['fc_layers']):
            prev_output = tf.layers.dense(prev_output, params['fc_dim'][i], activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='fc{}'.format(i))
        embed_width = int(params['width'] / 2 ** (params['out_conv_layers'] + 1))
        embed_height = int(params['height'] / 2 ** (params['out_conv_layers'] + 1))
        embed_dim = embed_width * embed_height * params['embed_channels']
        self.image_embed = tf.layers.dense(prev_output, embed_dim, activation=tf.nn.relu,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(), name='embed')
        embed_out_dim = (-1, embed_height, embed_width, params['embed_channels'])
        prev_output = tf.reshape(self.image_embed, embed_out_dim)
        for i in range(params['out_conv_layers']):
            prev_unpooled = unpool(prev_output)
            prev_padded = tf.pad(prev_unpooled, [[0, 0], [2, 2], [2, 2], [0, 0]])
            prev_output = tf.layers.conv2d(prev_padded, params['out_conv_filters'][i],
                                           params['out_conv_dim'][i],
                                           strides=params['out_conv_stride'][i], activation=tf.nn.relu,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           name='outconv{}'.format(i))
        prev_unpooled = unpool(prev_output)
        prev_padded = tf.pad(prev_unpooled, [[0, 0], [2, 2], [2, 2], [0, 0]])
        self.image_out = tf.layers.conv2d(prev_padded, params['channels'],
                                          params['output_conv_dim'],
                                          strides=params['output_conv_stride'],
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name='out')
        self.truth_in = tf.placeholder(tf.float32, input_dims)
        self.loss = tf.reduce_mean((self.image_out - self.truth_in)**2/2)
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(params['lr'], self.global_step, params['decay_steps'],
                                                        params['lr_decay'], staircase=True)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

    def train(self, iters, train_gen, valid_gen, print_iters=100, save_iters=1000, sess=None):
        saver = tf.train.Saver()

        logfile = open(os.path.join(self.log_path, self.model_name), "a")

        if not sess:
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

        print("Training started")
        for i in range(iters):
            try:
                self.train_batch(sess, train_gen)
                if i % save_iters == 0:
                    print("Saving...")
                    saver.save(sess, os.path.join(self.ckpt_path, self.model_name))
                if i % print_iters == 0:
                    print("\nIteration {}:".format(i))
                    print("Getting validation loss...")
                    val_loss = self.eval_batches(sess, valid_gen, 10)
                    print("Getting training loss...")
                    train_loss = self.eval_batches(sess, train_gen, 10)
                    print("Valid Loss: {0:.6f}".format(val_loss))
                    print("Train Loss: {0:.6f}".format(train_loss))
                    logfile.write("\n{:d} {:.6f} {:.6f}".format(i, train_loss, val_loss))
            except KeyboardInterrupt:
                print("Interrupted by user at iteration {}".format(i))
                logfile.close()
                return sess
        logfile.close()

    def train_batch(self, sess, train_gen):
        batch_x, batch_y = train_gen.next()
        sess.run(self.train_op, feed_dict={self.image_in: batch_x, self.truth_in: batch_y})

    def eval_batches(self, sess, eval_gen, num_batches):
        losses = []
        for i in range(num_batches):
            batch_x, batch_y = eval_gen.next()
            loss_v = sess.run(self.loss, feed_dict={self.image_in: batch_x, self.truth_in: batch_y})
            losses.append(loss_v)
        return np.mean(losses)

    def restore_last_session(self):
        saver = tf.train.Saver()
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        return sess
