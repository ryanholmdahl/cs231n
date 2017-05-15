import tensorflow as tf

from layers import unpool
from model import Model


class Conv2ConvModel(Model):
    def add_placeholders(self):
        input_dims = (None, self.config.im_height, self.config.im_width, self.config.im_channels)
        self.image_in = tf.placeholder(tf.float32, shape=input_dims)
        self.truth_in = tf.placeholder(tf.float32, input_dims)
        self.global_step = tf.Variable(0, trainable=False)

    def add_prediction_op(self):
        prev_output = self.image_in
        for i in range(self.config.in_conv_layers):
            prev_output = tf.layers.conv2d(prev_output, self.config.in_conv_filters[i], self.config.in_conv_dim[i],
                                           strides=self.config.in_conv_stride[i], activation=tf.nn.relu,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           name='inconv{}'.format(i))
            prev_output = tf.layers.max_pooling2d(prev_output, 2, strides=2)
        prev_output = tf.contrib.layers.flatten(prev_output)
        for i in range(self.config.fc_layers):
            prev_output = tf.layers.dense(prev_output, self.config.fc_dim[i], activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='fc{}'.format(i))
        embed_width = int(self.config.im_width / 2 ** (self.config.out_conv_layers + 1))
        embed_height = int(self.config.im_height / 2 ** (self.config.out_conv_layers + 1))
        embed_dim = embed_width * embed_height * self.config.embed_channels
        image_embed = tf.layers.dense(prev_output, embed_dim, activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(), name='embed')
        embed_out_dim = (-1, embed_height, embed_width, self.config.embed_channels)
        prev_output = tf.reshape(image_embed, embed_out_dim)
        for i in range(self.config.out_conv_layers):
            prev_unpooled = unpool(prev_output)
            pad_val = int((self.config.out_conv_dim[i] - 1) / 2)
            prev_padded = tf.pad(prev_unpooled, [[0, 0], [pad_val, pad_val], [pad_val, pad_val], [0, 0]])
            prev_output = tf.layers.conv2d(prev_padded, self.config.out_conv_filters[i],
                                           self.config.out_conv_dim[i], activation=tf.nn.relu,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           name='outconv{}'.format(i))
        prev_unpooled = unpool(prev_output)
        pad_val = int((self.config.output_conv_dim - 1) / 2)
        prev_padded = tf.pad(prev_unpooled, [[0, 0], [pad_val, pad_val], [pad_val, pad_val], [0, 0]])
        return tf.layers.conv2d(prev_padded, self.config.im_channels,
                                self.config.output_conv_dim,
                                strides=self.config.output_conv_stride,
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                name='out')

    def add_loss_op(self, final_layer):
        return tf.reduce_mean((final_layer - self.truth_in) ** 2 / 2)

    def add_training_op(self, loss):
        learning_rate = tf.train.exponential_decay(self.config.lr, self.global_step, self.config.lr_decay_steps,
                                                   self.config.lr_decay, staircase=True)
        return tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=self.global_step)

    def create_feed_dict(self, inputs_batch, outputs_batch=None):
        feed_dict = {self.image_in: inputs_batch}
        if outputs_batch is not None:
            feed_dict[self.truth_in] = outputs_batch
        return feed_dict

class Affine2AffineModel(Model):
    def add_placeholders(self):
        input_dims = (None, self.config.im_height, self.config.im_width, self.config.im_channels)
        self.image_in = tf.placeholder(tf.float32, shape=input_dims)
        self.truth_in = tf.placeholder(tf.float32, input_dims)
        self.global_step = tf.Variable(0, trainable=False)

    def add_prediction_op(self):
        prev_output = self.image_in
        prev_output = tf.contrib.layers.flatten(prev_output)
        for i in range(self.config.fc_layers):
            prev_output = tf.layers.dense(prev_output, self.config.fc_dim[i], activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='fc{}'.format(i))
        prev_output = tf.layers.dense(prev_output, self.config.im_height * self.config.im_width * self.config.im_channels, kernel_initializer=tf.contrib.layers.xavier_initializer(),name='fc_out')
        return tf.reshape(prev_output, (-1, self.config.im_height, self.config.im_width, self.config.im_channels))

    def add_loss_op(self, final_layer):
        return tf.reduce_mean((final_layer - self.truth_in) ** 2 / 2)

    def add_training_op(self, loss):
        learning_rate = tf.train.exponential_decay(self.config.lr, self.global_step, self.config.lr_decay_steps,
                                                   self.config.lr_decay, staircase=True)
        return tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=self.global_step)

    def create_feed_dict(self, inputs_batch, outputs_batch=None):
        feed_dict = {self.image_in: inputs_batch}
        if outputs_batch is not None:
            feed_dict[self.truth_in] = outputs_batch
        return feed_dict