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

from adversarial_autoencoder import ModularGenerator
from model_builder import ModularDiscriminator
from utils.activation_funcs import leaky_relu
from utils.util import minibatches, Progbar
from data.dataset_builder import Dataset
from tensorflow.examples.tutorials.mnist import input_data

class DC_WGAN():
    """ DCGAN w/ Improved WGAN Loss
    """

    def __init__(self):
        """ Initialize the DC_WGAN.
        """
        # Learning Parameters
        self.generator_epochs = 200000
        self.discr_epochs = 5
        self.lr = 1e-5
        self.lr_decay = 1
        self.lr_decay_steps = 100
        self.n_eval_batches = 10
        self.batch_size = 32
        self.beta1 = 0.5
        self.beta2 = 0.9
        self.lambda_cost = 10
        self.gans_image_lambda = 1
        self.gans_gaussian_lambda = 1
        self.gans_reconstruction_lambda = 1

        # Logging Params
        self.ckpt_path = "ckpt"
        self.log_path = "log"

        # Model Parameters
        self.im_width = 48
        self.im_height = 48
        self.im_channels = 1
        self.style_dim = 128
        self.num_emotions = 10
        params = {}
        params['im_width'] = self.im_width
        params['im_height'] = self.im_height
        params['im_channels'] = self.im_channels
        params['style_dim'] = self.style_dim

        # Create Three Neural Networks
        self.generator = Generator(params=params)
        self.gaussian_discriminator = GaussianDiscriminator(params=params)
        self.image_discriminator = ImageDiscriminator(params=params)

    def add_place_holders(self):
        input_dims = (None, self.im_height, self.im_width, self.im_channels)
        image_in = tf.placeholder(tf.float32, shape=input_dims)  # Input Images
        emotion_label = tf.placeholder(tf.float32, shape=(None, self.num_emotions))  # Emotion One-hot Encoding
        gaussian_in = tf.placeholder(tf.float32, shape=(None, self.style_dim))
        global_step = tf.Variable(0, trainable=False)
        return image_in, emotion_label, gaussian_in, global_step

    def get_solvers(self):
        dg_solver = tf.train.AdamOptimizer(
            learning_rate=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
        )
        di_solver = tf.train.AdamOptimizer(
            learning_rate=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
        )
        g_solver = tf.train.AdamOptimizer(
            learning_rate=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
        )
        return dg_solver, di_solver, g_solver

    def loss(self, image_logits_real, image_logits_fake, gaussian_logits_real, gaussian_logits_fake, real_imgs,
             generated_imgs, real_gaussians, fake_gaussians):
        # Generator Cost
        gen_image_cost = -tf.reduce_mean(image_logits_fake)
        gen_gaussian_cost = -tf.reduce_mean(gaussian_logits_fake)
        gen_reconstruction_cost = tf.reduce_mean(real_imgs - generated_imgs) ** 2 / 2
        g_cost = (
            self.gans_image_lambda * gen_image_cost +
            self.gans_gaussian_lambda * gen_gaussian_cost +
            self.gans_reconstruction_lambda * gen_reconstruction_cost
        )

        # Discriminator Cost
        discr_image_cost = tf.reduce_mean(image_logits_fake) - tf.reduce_mean(image_logits_real)
        discr_gaussian_cost = tf.reduce_mean(gaussian_logits_fake) - tf.reduce_mean(gaussian_logits_real)

        image_alpha = tf.random_uniform(
            shape=tf.shape(real_imgs),
            minval=0.,
            maxval=1.
        )
        image_differences = generated_imgs - real_imgs
        image_interpolates = real_imgs + (image_alpha * image_differences)
        print(image_interpolates.get_shape())
        interpolate_imgs = self.image_discriminator.add_prediction_op(input_logits=image_interpolates,
                                                                      data_type='interpolates', reuse=True)
        image_gradients = tf.gradients(interpolate_imgs, [image_interpolates])[0]
        image_slopes = tf.sqrt(tf.reduce_sum(tf.square(image_gradients), reduction_indices=[1]))
        image_gradient_penalty = tf.reduce_mean((image_slopes - 1.) ** 2)
        discr_image_cost += self.lambda_cost * image_gradient_penalty

        gaussian_alpha = tf.random_uniform(
            shape=[tf.shape(image_logits_real)[0], self.style_dim],
            minval=0.,
            maxval=1.
        )
        gaussian_differences = fake_gaussians - real_gaussians
        gaussian_interpolates = real_gaussians + (gaussian_alpha * gaussian_differences)
        interpolate_gauss = self.gaussian_discriminator.add_prediction_op(input_logits=gaussian_interpolates,
                                                                          data_type='interpolates', reuse=True)
        gaussian_gradients = tf.gradients(interpolate_gauss, [gaussian_interpolates])[0]
        gaussian_slopes = tf.sqrt(tf.reduce_sum(tf.square(gaussian_gradients), reduction_indices=[1]))
        gaussian_gradient_penalty = tf.reduce_mean((gaussian_slopes - 1.) ** 2)
        discr_gaussian_cost += self.lambda_cost * gaussian_gradient_penalty
        return discr_gaussian_cost, discr_image_cost, g_cost, gen_reconstruction_cost

    def build(self):
        tf.reset_default_graph()

        self.image_in, self.emotion_label, self.gaussian_in, self.global_step = self.add_place_holders()
        self.gen_images = tf.reshape(
            self.generator.add_prediction_op(preprocess_imgs(self.image_in), self.emotion_label),
            shape=[-1, self.im_height, self.im_width, self.im_channels])
        self.gen_styles = tf.reshape(self.generator.image_style, shape=[-1, self.style_dim])

        with tf.variable_scope("") as scope:
            # scale images to be -1 to 1
            self.image_logits_real = self.image_discriminator.add_prediction_op(preprocess_imgs(self.image_in))

            # Re-use discriminator weights on new inputs
            scope.reuse_variables()
            self.image_logits_fake = self.image_discriminator.add_prediction_op(self.gen_images)

        with tf.variable_scope("") as scope:
            # scale images to be -1 to 1
            self.gaussian_logits_real = self.gaussian_discriminator.add_prediction_op(self.gaussian_in)

            # Re-use discriminator weights on new inputs
            scope.reuse_variables()
            self.gaussian_logits_fake = self.gaussian_discriminator.add_prediction_op(self.gen_styles)

        # Get the list of variables for the discriminator and generator
        self.dg_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         self.gaussian_discriminator.config.model_name)
        self.di_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.image_discriminator.config.model_name)
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.generator.config.model_name)

        # get our solvers
        self.dg_solver, self.di_solver, self.g_solver = self.get_solvers()

        # get our loss
        self.dg_loss, self.di_loss, self.g_loss, self.reconstruction_loss = self.loss(self.image_logits_real,
                                                                                      self.image_logits_fake,
                                                                                      self.gaussian_logits_real,
                                                                                      self.gaussian_logits_fake,
                                                                                      self.image_in, self.gen_images,
                                                                                      self.gaussian_in, self.gen_styles)

        # setup training steps
        self.g_train_step = self.g_solver.minimize(self.g_loss, var_list=self.g_vars)
        self.di_train_step = self.di_solver.minimize(self.di_loss, var_list=self.di_vars)
        self.dg_train_step = self.dg_solver.minimize(self.dg_loss, var_list=self.dg_vars)
        self.dg_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.gaussian_discriminator.config.model_name)
        self.di_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.image_discriminator.config.model_name)
        self.g_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.generator.config.model_name)

    def fit(self, sess, saver, train_examples, dev_set):
        with open(os.path.join(self.ckpt_path, "DC_WGAN"), "w") as logfile:
            best_gen_dev_loss = float('inf')
            best_discr_dev_loss = float('inf')

            for gen_epoch in range(self.generator_epochs):
                print("Generator Epoch {:} out of {:}".format(gen_epoch + 1, self.generator_epochs))
                logfile.write(str(gen_epoch + 1))

                if gen_epoch > 0:
                    gen_tf_ops = self.g_train_step
                    gen_dev_loss = self.run_epoch(gen_tf_ops, [(self.reconstruction_loss, "reconstruction"),
                                                               (self.g_loss, "generator")], sess, train_examples,
                                                  dev_set, self.batch_size, logfile)
                    # if gen_dev_loss < best_gen_dev_loss:
                    #     best_gen_dev_loss = gen_dev_loss
                    #     save_path = os.path.join(self.ckpt_path, self.generator.config.model_name)
                    #     print("New best dev for generator! Saving model in {}".format(save_path))
                    #     saver.save(sess, save_path)

                for discr_epoch in range(self.discr_epochs):
                    print("Gen Epoch {} - Discriminator Epoch {:} out of {:}".format(gen_epoch + 1,
                                                                                     discr_epoch + 1,
                                                                                     self.discr_epochs))
                    discr_tf_ops = [self.dg_train_step, self.di_train_step]
                    discr_dev_loss = self.run_epoch(discr_tf_ops,
                                                    [(self.dg_loss, "Gaussian"), (self.di_loss, "image")],
                                                    sess,
                                                    train_examples, dev_set, self.batch_size, logfile)

    def run_epoch(self, tf_ops, loss_fns, sess, train_examples, dev_set, batch_size, logfile=None):
        # prog = Progbar(target=1 + train_examples[0].shape[0] / batch_size)
        for i, (inputs_batch, outputs_batch) in enumerate(minibatches(train_examples, batch_size)):
            feed = {
                self.image_in: inputs_batch,
                self.emotion_label: outputs_batch,
                self.gaussian_in: np.random.normal(size=(len(inputs_batch), self.style_dim))
            }
            self.train_on_batch(tf_ops, feed, sess)
            # prog.update(i + 1, [("train loss", loss)])
        dev_loss_sum = 0
        for (loss_fn, loss_name) in loss_fns:
            train_loss = self.eval_batches(loss_fn, sess, train_examples, self.n_eval_batches)
            print("Train {} loss: {:.6f}".format(loss_name, train_loss))
            dev_loss = self.eval_batches(loss_fn, sess, dev_set, self.n_eval_batches)
            print("Dev {} loss: {:.6f}".format(loss_name, dev_loss))
            dev_loss_sum += dev_loss
        # logfile.write(",{0:.5f},{1:.5f}\n".format(float(train_loss), float(dev_loss)))
        print("")
        return dev_loss_sum

    def train_on_batch(self, tf_ops, feed, sess):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            tf_ops: list of tf ops to compute
            feed: feed dict
            sess: tf.Session()
            get_loss: whether to calculate the batch loss
        Returns:
            loss: loss over the batch (a scalar) or zero if not requested
        """
        # if get_loss:
        #     if discriminator:
        #         _, _, dg_loss, di_loss = sess.run(tf_ops, feed_dict=feed)
        #         return dg_loss, di_loss
        #     else:
        #         _, loss = sess.run(tf_ops, feed_dict=feed)
        #         return loss
        # else:
        sess.run(tf_ops, feed_dict=feed)

    def eval_batches(self, loss_fn, sess, eval_set, num_batches):
        """Evaluate the loss on a number of given minibatches of a dataset.

        Args:
            loss_fn: loss function
            sess: tf.Session()
            eval_set: full dataset, as passed to run_epoch
            num_batches: number of batches to evaluate
        Returns:
            loss: loss over the batches (a scalar)
        """
        losses = []
        for i, (inputs_batch, outputs_batch) in enumerate(minibatches(eval_set, self.batch_size)):
            if i >= num_batches:
                break
            feed = {
                self.image_in: inputs_batch,
                self.emotion_label: outputs_batch,
                self.gaussian_in: np.random.normal(size=(len(inputs_batch), self.style_dim))
            }
            loss = self.eval_on_batch(loss_fn, feed, sess)
            losses.append(loss)
        return np.mean(losses)

    def eval_on_batch(self, loss_fn, feed, sess):
        """Evaluate the loss on a given batch

        Args:
            loss_fn: loss function
            feed: feed dict
            sess: tf.Session()
        Returns:
            loss: loss over the batch (a scalar)
        """
        loss = sess.run(loss_fn, feed_dict=feed)
        return loss

    def restore_from_checkpoint(self, sess, saver):
        save_path = os.path.join(self.ckpt_path, self.generator.config.model_name)
        saver.restore(sess, save_path)


class Generator(ModularGenerator):
    """
    Input: N x 32 x 32 x 1
    Output: N x 32 x 32 x 1

    Generator Network Architecture:

    (FC Layer (1024 Hidden Units) + Leaky ReLU Activation Function) x 5
    FC Layer (128 Hidden Units) + Leaky ReLU Activation Function)
    Batch Norm
    FC Layer + Leaky ReLU -> N x 4 x 4 x 256
    Deconv Layer + Leaky ReLU -> N x 8 x 8 x 128
    Deconv Layer + Leaky ReLU -> N x 16 x 16 x 64
    Deconv Layer + Tanh -> N x 32 x 32 x 1
    """

    def __init__(self, params):
        # Regularization
        params["fc_dropout"] = 0

        # Input Convolution Layers
        params['in_conv_layers'] = 0

        # Input FC Layers
        params['fc_layers'] = 6
        params['fc_dim'] = [1024, 1024, 1024, 1024, 1024, 128]
        params['fc_activation_funcs'] = [tf.nn.relu] * params['fc_layers']

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


class GaussianDiscriminator(ModularDiscriminator):
    """
    Input: N x 32 x 32 x 1
    Output: N x 1

    Discriminator Network Architecture:

    a bunch of fc
    """

    def __init__(self, params):
        # Regularization
        params["fc_dropout"] = 0

        # Input Convolution Layers
        params["fc_layers"] = 3
        params["fc_dim"] = [1024, 1024, 1]
        params["fc_activation_funcs"] = [leaky_relu, leaky_relu, None]

        # Model Info Params
        params["model_name"] = "gaussian_discriminator"

        # Initialize the Model
        super().__init__(params)

    def add_placeholders(self):
        pass

    def create_feed_dict(self, inputs_batch, outputs_batch=None, **kwargs):
        pass

    def add_prediction_op(self, input_logits=None, reuse=None, **kwargs):
        with tf.variable_scope(self.config.model_name, reuse=reuse):
            prev_output = self.add_in_fc(input_logits)
            return prev_output

    def add_loss_op(self, loss_params=None):
        pass

    def add_training_op(self, loss):
        pass

    def build(self):
        pass

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


class ImageDiscriminator(ModularDiscriminator):
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
        # Regularization
        params["fc_dropout"] = 0

        # Input Convolution Layers
        params['dim'] = 64
        params['in_conv_layers'] = 0
        params['in_conv_filters'] = [params['dim'], params['dim'] * 2, params['dim'] * 4]
        params['in_conv_dim'] = [5, 5, 5]
        params['in_conv_stride'] = [2, 2, 2]
        params['in_conv_activation_func'] = [leaky_relu, leaky_relu, leaky_relu]
        params["fc_layers"] = 3
        params["fc_dim"] = [1024, 1024, 1]
        params["fc_activation_funcs"] = [leaky_relu, leaky_relu, None]

        # Model Info Params
        params["model_name"] = "image_discriminator"

        # Initialize the Model
        super().__init__(params)

    def add_prediction_op(self, input_logits=None, reuse=None, **kwargs):
        with tf.variable_scope(self.config.model_name, reuse=reuse):
            # prev_output = self.add_in_convolution(input_logits, maxpooling=False)
            prev_output = self.add_in_fc(tf.contrib.layers.flatten(input_logits))
            return prev_output

    def add_placeholders(self):
        pass

    def add_loss_op(self, **kwargs):
        pass

    def add_training_op(self, loss):
        pass

    def create_feed_dict(self, inputs_batch, outputs_batch=None, **kwargs):
        pass

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

    def build(self):
        pass


def preprocess_imgs(imgs):
    return imgs

if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    m = DC_WGAN()
    m.build()

    train_examples = [
        np.pad(mnist.train.images.reshape((-1, 28, 28, 1)), ((0, 0), (10, 10), (10, 10), (0, 0)), 'constant')[0:1000],
        mnist.train.labels]
    dev_examples = [
        np.pad(mnist.validation.images.reshape((-1, 28, 28, 1)), ((0, 0), (10, 10), (10, 10), (0, 0)), 'constant')[
        0:1000],
        mnist.validation.labels]
    print(train_examples[0].shape)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        m.fit(sess, saver, train_examples, dev_examples)
        # m.restore_from_checkpoint(sess, saver)
        # m.generator.demo(sess, dataset.train_examples, dataset.dev_examples, 10)
