#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
CS 231N 2016-2017
model.py: Model class that abstracts Tensorflow graph for learning tasks.
Sahil Chopra <schopra8@cs.stanford.edu>
Ryan Holmdahl <ryanlh@stanford.edu>
"""

import numpy as np
import tensorflow as tf
import os
from util import minibatches, Progbar, vectorize_stances


class Config:
    def __init__(self, params):
        for key in params:
            setattr(self, key, params[key])


class Model(object):
    """Abstracts a Tensorflow graph for a learning task.

    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs.
    """
    def __init__(self, params):
        self.config = Config(params)

    def add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph.

        Tensorflow uses placeholder variables to represent locations in a
        computational graph where data is inserted.  These placeholders are used as
        inputs by the rest of the model building and will be fed data during
        training.

        See for more information:
        https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def create_feed_dict(self, inputs_batch, outputs_batch=None):
        """Creates the feed_dict for one step of training.

        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        If labels_batch is None, then no labels are added to feed_dict.

        Args:
            inputs_batch = image inputs
            outputs_batch = output inputs
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input data into predictions.

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, final_layer):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        sess.run() to train the model.

        Args:
            loss: Loss tensor (a scalar).
        Returns:
            train_op: The Op for training.
        """

        raise NotImplementedError("Each Model must re-implement this method.")

    def demo(self, sess, train_data, dev_data, demo_count):
        """Demonstrates the current model.

        Args:
            sess: Current session.
            train_data: training data to demo.
            dev_data: development data to demo.
            demo_count: Number of demo samples.
        """

        raise NotImplementedError("Each Model must re-implement this method.")

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
        with open(os.path.join(self.config.log_path, self.config.model_name), "w") as logfile:
            best_dev_loss = float('inf')
            for epoch in range(self.config.n_epochs):
                print("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
                logfile.write(str(epoch+1))
                dev_loss = self.run_epoch(sess, train_examples, dev_set, logfile)
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    if saver:
                        save_path = os.path.join(self.config.ckpt_path, self.config.model_name)
                        print("New best dev! Saving model in {}".format(save_path))
                        saver.save(sess, save_path)

    def restore_from_checkpoint(self, sess, saver):
        save_path = os.path.join(self.config.ckpt_path, self.config.model_name)
        saver.restore(sess, save_path)

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

