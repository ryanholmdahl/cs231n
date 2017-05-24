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
                        Where gen_paras and critic_params are dictionaries that will be provided to initialize
                        neural network architectures via ModelBuilder. and learning_params will contain
                        parameters as to the learning rates for both models.
        """
        generator = Generator(params)
        critic = Critic(params)


class Generator(ModularModel):
    # Generator Network Architecture:
    # (Dense Layer + Batch Norm + ReLU) x N
    # Deconvolution + Batch Norm + Activation x N
    # Deconvolution

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
        params['fc_dim'] = [1024, 1024, 1024, 1024, 1024, 1024]
        params['fc_activation_funcs'] = [leaky_relu] * params['fc_layers']

        # Embedding Layer (FC -> Conv Intermediary Layer)
        params['embed_channels'] = 100
        params['embed_activation_func'] = leaky_relu

        # Output Deconvolution (Transpose Convolution) or Unconvolution Layers
        params["use_transpose"] = True
        params['out_conv_layers'] = 3
        params['out_conv_filters'] = [5, 5]
        params['out_conv_dim'] = [5, 5]
        params['out_conv_stride'] = [2, 2, 2]
        params['out_conv_activation_func'] = [leaky_relu, leaky_relu, tf.nn.tanh]

        # Model Info Params
        params["model_name"] = "generator"
        params["ckpt_path"] = "../ckpt"
        params["log_path"] = "../log"

        super().__init__(params)


    def add_loss_op(self, final_layer):
        pass


class Critic(ModularModel):
    # Discriminator Network Architecture (Input Image):
    # 2D Convolution with stride 2
    # Apply Leaky ReLU
    # (2D Convolution with stride 2 + Batch Normalization + Leaky ReLU) x N
    # Dense Layers + Batch Normalization + Leaky ReLU
    # Dense Layer -> 0/1
    def add_loss_op(self, final_layer):
        pass











