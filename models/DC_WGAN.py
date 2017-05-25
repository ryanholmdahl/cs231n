#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
CS 231N 2016-2017
DC_WGAN.py: Implement DC_GAN with Improved WGAN Loss Function
Sahil Chopra <schopra8@cs.stanford.edu>
Ryan Holmdahl <ryanlh@stanford.edu>
"""

import tensorflow as tf
import numpy as np
import os

from model_builder import ModularModel
from utils.activation_funcs import leaky_relu
from utils.util import minibatches, Progbar


class DC_WGAN():
    """ DCGAN w/ Improved WGAN Loss
    """
    def __init__(self,):
        """ Initialize the DC_WGAN.
        """
        params = {}
        # Learning Parameters
        params['lr'] = 5e-5
        params['lr_decay'] = 0.98
        params['lr_decay_steps'] = 100
        params["n_eval_batches"] = 10
        params["batch_size"] = 32
        params["beta1"] = 0.5
        params["beta2"] = 0.9

        # Input Image Parameters
        params['im_width'] = 32
        params['im_height'] = 32
        params['im_channels'] = 1

        # Logging Params
        params["ckpt_path"] = "../ckpt"
        params["log_path"] = "../log"

        # Create Two Neural Networks
        self.generator = Generator(params=params)
        self.discriminator = Discriminator(params=params)

        self.generator_epochs = 200000
        self.discr_epochs = 5

    def train_on_batch(self, sess, inputs_batch, outputs_batch, get_loss=False):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: tf.Session()
            inputs_batch: batch of image inputs
            outputs_batch: batch of output (rotated) images
            get_loss: whether to calculate the batch loss
        Returns:
            loss: loss over the batch (a scalar) or zero if not requested
        """
        feed = self.create_feed_dict(inputs_batch, outputs_batch)
        if get_loss:
            _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
            return loss
        else:
            sess.run(self.train_op, feed_dict=feed)
            return 0

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            inputs_batch
        Returns:
            predictions: np.ndarray of shape (n_samples, img_width, img_height, img_channels)
        """
        feed = self.create_feed_dict(inputs_batch)
        preds = sess.run(self.pred, feed_dict=feed)
        return preds

    def eval_on_batch(self, sess, inputs_batch, outputs_batch):
        """Evaluate the loss on a given batch

        Args:
            sess: tf.Session()
            inputs_batch
            outputs_batch
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(inputs_batch, outputs_batch)
        loss = sess.run(self.loss, feed_dict=feed)
        return loss

    def eval_batches(self, sess, eval_set, num_batches):
        """Evaluate the loss on a number of given minibatches of a dataset.

        Args:
            sess: tf.Session()
            eval_set: full dataset, as passed to run_epoch
            num_batches: number of batches to evaluate
        Returns:
            loss: loss over the batches (a scalar)
        """
        losses = []
        for i, (inputs_batch, outputs_batch) in enumerate(minibatches(eval_set, self.config.batch_size)):
            if i >= num_batches:
                break
            loss = self.eval_on_batch(sess, inputs_batch, outputs_batch)
            losses.append(loss)
        return np.mean(losses)

    def run_epoch(self, sess, train_examples, dev_set, logfile=None):
        prog = Progbar(target=1 + train_examples[0].shape[0] / self.config.batch_size)
        for i, (inputs_batch, outputs_batch) in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, inputs_batch, outputs_batch, get_loss=True)
            prog.update(i + 1, [("train loss", loss)])
        print("")
        print("Evaluating on train set...")
        train_loss = self.eval_batches(sess, train_examples, self.config.n_eval_batches)
        print("Train Loss: {0:.6f}".format(train_loss))
        print("Evaluating on dev set...")
        dev_loss = self.eval_batches(sess, dev_set, self.config.n_eval_batches)
        print("Dev Loss: {0:.6f}".format(dev_loss))
        logfile.write(",{0:.5f},{1:.5f}\n".format(float(train_loss), float(dev_loss)))
        return dev_loss

    def fit(self, sess, saver, train_examples, dev_set):
        with open(os.path.join(self.params['ckpt_path'], self.params['model_name']), "w") as logfile:
            best_gen_dev_loss = float('inf')
            best_discr_dev_loss = float('inf')
            for gen_epoch in range(self.generator_epochs):
                print("Generator Epoch {:} out of {:}".format(gen_epoch + 1, self.generator_epochs))
                logfile.write(str(gen_epoch+1))
                if gen_epoch > 0:
                    gen_dev_loss = self.generator.run_epoch(sess, train_examples, dev_set, logfile=logfile)
                    if gen_dev_loss < best_gen_dev_loss:
                        best_gen_dev_loss = gen_dev_loss
                        save_path = os.path.join(self.params['ckpt_path'], self.generator.model_name)
                        print("New best dev for generator! Saving model in {}".format(save_path))
                        saver.save(sess, save_path)
                for discr_epoch in range(self.discr_epochs):
                    print("Gen Epoch {} - Discriminator Epoch {:} out of {:}".format(gen_epoch + 1,
                                                                                     discr_epoch + 1,
                                                                                     self.discr_epochs))
                    discr_dev_loss = self.discriminator.run_epoch(sess, train_examples, dev_set, logfile=logfile)
                    if discr_dev_loss < best_discr_dev_loss:
                        best_discr_dev_loss = discr_dev_loss
                        save_path = os.path.join(self.params['ckpt_path'], self.discriminator.model_name)
                        print("New best dev for discriminator! Saving model in {}".format(save_path))
                        saver.save(sess, save_path)

class Generator(ModularModel):
    """
    Input: N x 32 x 32 x 1
    Output: N x 32 x 32 x 1

    Generator Network Architecture:

    (FC Layer (1024 Hidden Units) + Leaky ReLU Activation Function) x 5
    FC Layer (128 Hidden Units) + Leaky ReLU Activation Function)
    Batch Norm
    FC Layer + Leaky ReLU -> N x 4 x 4 x 256
    Deconv Layer + Leaky ReLU -> N x 8 x 8 x 128 + Leaky
    Deconv Layer + Leaky ReLU -> N x 16 x 16 x 64
    Deconv Layer + Tanh -> N x 32 x 32 x 1
    """

    def __init__(self, params):

        params["gans_reconstruction_lambda"] = 0.8

        # Regularization
        params["fc_dropout"] = 0

        # Input Convolution Layers
        params['in_conv_layers'] = 0

        # Input FC Layers
        params['fc_layers'] = 6
        params['fc_dim'] = [1024, 1024, 1024, 1024, 1024, 128]
        params['fc_activation_funcs'] = [tf.nn.relu] * params['fc_layers']

        # Normalize Input Vector?
        params['normalize_input'] = True

        # Embedding Layer (FC -> Conv Intermediary Layer)
        params['dim'] = 64
        params['embed_channels'] = params['dim'] * 4
        params['embed_activation_func'] = tf.nn.relu

        # Output Deconvolution (Transpose Convolution) or Unconvolution Layers
        params["use_transpose"] = True
        params['out_conv_layers'] = 3
        params['out_conv_filters'] = [params['dim'] * 2, params['dim'], 1]
        params['out_conv_dim'] = [5, 5, 5]
        params['out_conv_stride'] = [2, 2, 2]
        params['out_conv_activation_func'] = [tf.nn.relu, tf.nn.relu, tf.nn.tanh]

        # Model Info Params
        params["model_name"] = "generator"

        # Initialize the Model
        super().__init__(params)

    def add_placeholders(self):
        input_dims = (None, self.config.im_height, self.config.im_width, self.config.im_channels)
        self.input_imgs = tf.placeholder(tf.float32, shape=input_dims)
        self.rotated_imgs = tf.placeholder(tf.float32, shape=input_dims)
        self.d_logits = tf.placeholder(tf.float32, shape=[None, 1])
        self.global_step = tf.Variable(0, trainable=False)

    def add_loss_op(self, **kwargs):
        gen_cost = -tf.reduce_mean(kwargs['d_logits'])
        reconstruction_cost = -tf.reduce_mean((kwargs['rotated_imgs'] - kwargs['generated_imgs']) **2 / 2)
        cost = self.config.gans_reconstruction_lambda * gen_cost + \
               (1 - self.config.gans_reconstruction_lambda) * reconstruction_cost
        return cost

    def add_training_op(self, loss):
        learning_rate = tf.train.exponential_decay(self.config.lr, self.global_step, self.config.lr_decay_steps,
                                                   self.config.lr_decay, staircase=True)
        train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=self.config.beta1,
            beta2=self.config.beta2,
        ).minimize(loss)
        return train_op

    def build(self):
        self.add_placeholders()
        self.real_imgs_preds = self.add_prediction_op(self.real_imgs, data_type='real_imgs')
        self.fake_imgs_preds = self.add_prediction_op(self.fake_imgs, data_type='fake_imgs')
        self.loss = self.add_loss_op(d_logits=self.d_logits, rotated_imgs=self.rotated_imgs,
                                     generated_imgs=self.real_imgs_preds)
        self.train_op = self.add_training_op(self.loss)

    def train_on_batch(self, sess, inputs_batch, outputs_batch, get_loss=False):
        pass

    def predict_on_batch(self, sess, inputs_batch):
        pass

    def eval_on_batch(self, sess, inputs_batch, outputs_batch):
        pass

    def eval_batches(self, sess, eval_set, num_batches):
        pass

    def run_epoch(self, sess, train_examples, dev_set, logfile=None):
        pass

    def fit(self, sess, saver, train_examples, dev_set):
        pass

class Discriminator(ModularModel):
    """
    Input: N x 32 x 32 x 1
    Output: N x 1

    Discriminator Network Architecture:

    Conv2 Layer + Leaky ReLU -> N x 16 x 16 x 64
    Conv2 Layer + Leaky ReLU -> N x 8 x 8 x 128
    Conv2 Layer + Leaky ReLU -> N x 4 x 4 x 256
    FC Layer -> N x 1
    """

    def __init__(self, params):
        # Learning Parameters
        params["lambda_cost"] = 10

        # Regularization
        params["fc_dropout"] = 0

        # Input Convolution Layers
        params['dim'] = 64
        params['in_conv_layers'] = 3
        params['in_conv_filters'] = [params['dim'], params['dim'] * 2, params['dim'] * 4]
        params['in_conv_dim'] = [5, 5, 5]
        params['in_conv_stride'] = [2, 2, 2]
        params['out_conv_activation_func'] = [leaky_relu, leaky_relu, leaky_relu]

        # Model Info Params
        params["model_name"] = "generator"

        # Initialize the Model
        super().__init__(params)

    def add_placeholders(self):
        input_dims = (None, self.config.im_height, self.config.im_width, self.config.im_channels)
        self.real_imgs = tf.placeholder(tf.float32, shape=input_dims)
        self.fake_imgs = tf.placeholder(tf.float32, shape=input_dims)
        self.global_step = tf.Variable(0, trainable=False)

    def create_feed_dict(self, inputs_batch, outputs_batch=None, **kwargs):
        feed_dict = {
            self.real_imgs: inputs_batch,
            self.fake_imgs: kwargs['fake_imgs'],
        }
        return feed_dict

    def add_prediction_op(self, input_logits=None, **kwargs):
        unsquashed_output = super(ModularModel, self).add_prediction_op(input_logits=input_logits)
        layer_name = '{}.discriminator_output.{}'.format(self.config.model_name, kwargs['data_type'])
        preds = tf.layers.dense(unsquashed_output, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name=layer_name)
        return preds

    def add_loss_op(self, loss_params=None):
        cost = tf.reduce_mean(self.real_imgs_preds) - tf.reduce_mean(self.fake_imgs_preds)
        alpha = tf.random_uniform(
            shape=[self.config.batch_size, 1],
            minval=0.,
            maxval=1.
        )
        differences = self.fake_imgs - self.real_imgs
        interpolates = self.real_imgs + (alpha * differences)
        gradients = tf.gradients(self.add_prediction_op(input_logits=interpolates, data_type='interpolates'),
                                 [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        cost += self.config.lambda_cost * gradient_penalty
        return cost

    def add_training_op(self, loss):
        learning_rate = tf.train.exponential_decay(self.config.lr, self.global_step, self.config.lr_decay_steps,
                                                   self.config.lr_decay, staircase=True)
        train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=self.config.beta1,
            beta2=self.config.beta2,
        ).minimize(loss)
        return train_op

    def build(self):
        self.add_placeholders()
        self.real_imgs_preds = self.add_prediction_op(self.real_imgs, data_type='real_imgs')
        self.fake_imgs_preds = self.add_prediction_op(self.fake_imgs, data_type='fake_imgs')
        self.loss = self.add_loss_op()
        self.train_op = self.add_training_op(self.loss)

    def train_on_batch(self, sess, inputs_batch, outputs_batch, get_loss=False):
        pass

    def predict_on_batch(self, sess, inputs_batch):
        pass

    def eval_on_batch(self, sess, inputs_batch, outputs_batch):
        pass

    def eval_batches(self, sess, eval_set, num_batches):
        pass

    def run_epoch(self, sess, train_examples, dev_set, logfile=None):
        pass

    def fit(self, sess, saver, train_examples, dev_set):
        pass

if __name__ == '__main__':
    dc_wgan = DC_WGAN()
