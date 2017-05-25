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
    """
    """
    def __init__(self, params):
        """ Initialize the DC_WGAN.

        :param params: directory {"gen": gen_params, "critic": critic_params, "learning": learning_params}
                        Where gen_params and critic_params are dictionaries that will be provided to initialize
                        neural network architectures via ModelBuilder. and learning_params will contain
                        parameters as to the learning rates for both models.
        """
        generator = Generator(params)
        critic = Discriminator(params)


class Generator(ModularModel):
    """
    Input: N x 32 x 32 x 1
    Output: N x 32 x 32 x 1

    Generator Network Architecture:

    (FC Layer (1024 Hidden Units) + Leaky ReLU Activation Function) x 5
    FC Layer (128 Hidden Units) + Leaky ReLU Activation Function)
    FC Layer + Leaky ReLU -> N x 4 x 4 x 256
    Deconv Layer + Leaky ReLU -> N x 8 x 8 x 128 + Leaky
    Deconv Layer + Leaky ReLU -> N x 16 x 16 x 64
    Deconv Layer + Tanh -> N x 32 x 32 x 1
    """

    def __init__(self, params):
        params = {}

        # Learning Parameters
        params['lr'] = 5e-5
        params['lr_decay'] = 0.98
        params['lr_decay_steps'] = 100
        params["n_epochs"] = 10
        params["n_eval_batches"] = 10
        params["batch_size"] = 32

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
        params['normalize_input'] = False

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

    def add_loss_op(self, final_layer):
        """

        :param final_layer: Final Layer
        :return:
        """
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
        params = {}

        # Learning Parameters
        params['lr'] = 5e-5
        params['lr_decay'] = 0.98
        params['lr_decay_steps'] = 100
        params["n_epochs"] = 10
        params["n_eval_batches"] = 10
        params["batch_size"] = 32

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

    def add_loss_op(self, final_layer):
        pass











