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
from util import show_image_example
from model_builder import ModularModel
from activation_funcs import leaky_relu
from data.dataset_builder import Dataset


class DC_WGAN():
    """ DCGAN w/ Improved WGAN Loss
    """
    def __init__(self,):
        """ Initialize the DC_WGAN.
        """
        generator = Generator(params={})
        # discriminator = Discriminator(params={})

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
        # Learning Parameters
        params['lr'] = 5e-5
        params['lr_decay'] = 0.98
        params['lr_decay_steps'] = 100
        params["n_epochs"] = 10
        params["n_eval_batches"] = 10
        params["batch_size"] = 32
        params["beta1"] = 0.5
        params["beta2"] = 0.9
        params["gans_reconstruction_lambda"] = 0.8

        # Regularization
        params["fc_dropout"] = 0

        # Input Image Parameters
        params['im_width'] = 32
        params['im_height'] = 32
        params['im_channels'] = 1

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
        params["ckpt_path"] = "../ckpt"
        params["log_path"] = "../log"

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
        params['lr'] = 5e-5
        params['lr_decay'] = 0.98
        params['lr_decay_steps'] = 100
        params["n_epochs"] = 10
        params["n_eval_batches"] = 10
        params["batch_size"] = 32
        params["lambda_cost"] = 10
        params["beta1"] = 0.5
        params["beta2"] = 0.9

        # Regularization
        params["fc_dropout"] = 0

        # Input Image Parameters
        params['im_width'] = 32
        params['im_height'] = 32
        params['im_channels'] = 1

        # Input Convolution Layers
        params['dim'] = 64
        params['in_conv_layers'] = 3
        params['in_conv_filters'] = [params['dim'], params['dim'] * 2, params['dim'] * 4]
        params['in_conv_dim'] = [5, 5, 5]
        params['in_conv_stride'] = [2, 2, 2]
        params['out_conv_activation_func'] = [leaky_relu, leaky_relu, leaky_relu]

        # Model Info Params
        params["model_name"] = "generator"
        params["ckpt_path"] = "../ckpt"
        params["log_path"] = "../log"

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

if __name__ == '__main__':
    dc_wgan = DC_WGAN()