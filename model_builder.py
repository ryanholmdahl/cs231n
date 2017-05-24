import tensorflow as tf

from layers import unpool
from model import Model


class ModularModel(Model):
    def add_placeholders(self):
        input_dims = (None, self.config.im_height, self.config.im_width, self.config.im_channels)
        self.image_in = tf.placeholder(tf.float32, shape=input_dims)
        self.truth_in = tf.placeholder(tf.float32, input_dims)
        self.global_step = tf.Variable(0, trainable=False)
        self.is_train = tf.placeholder(tf.bool, shape=())

    def add_in_convolution(self, prev_output):
        for i in range(self.config.in_conv_layers):
            prev_output = tf.layers.conv2d(prev_output, self.config.in_conv_filters[i], self.config.in_conv_dim[i],
                                           strides=self.config.in_conv_stride[i], activation=tf.nn.relu,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           name='inconv{}'.format(i))
            prev_output = tf.layers.max_pooling2d(prev_output, 2, strides=2)
        return prev_output

    def add_in_fc(self, prev_output):
        for i in range(self.config.fc_layers):
            prev_output = tf.layers.dense(prev_output, self.config.fc_dim[i], activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='fc{}'.format(i))
        return prev_output

    def add_fixed_size_embed(self, prev_output):
        scaling_factor = 1
        for stride in self.config.out_conv_stride:
            scaling_factor *= stride
        embed_width = int(self.config.im_width / scaling_factor)
        embed_height = int(self.config.im_height / scaling_factor)
        embed_dim = embed_width * embed_height * self.config.embed_channels
        image_embed = tf.layers.dense(prev_output, embed_dim, activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(), name='embed')
        embed_out_dim = (-1, embed_height, embed_width, self.config.embed_channels)
        return tf.reshape(image_embed, embed_out_dim)

    def add_out_unconvolution(self, prev_output):
        for i in range(self.config.out_conv_layers):
            prev_unpooled = unpool(prev_output, self.config.out_conv_stride[i])
            prev_output = tf.layers.conv2d(prev_unpooled, self.config.out_conv_filters[i],
                                           self.config.out_conv_dim[i], activation=tf.nn.relu, padding='SAME',
                                           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           name='outconv{}'.format(i))
        return prev_output

    def add_out_deconvolution(self, prev_output):
        prev_dyn_shape = tf.shape(prev_output)
        batch_size = prev_dyn_shape[0]
        prev_shape = prev_output.get_shape().as_list()
        for i in range(self.config.out_conv_layers):
            in_filters = self.config.out_conv_filters[i - 1] if i > 0 else prev_output.get_shape()[3]
            W = tf.get_variable("deconvW{}".format(i), shape=(
                self.config.out_conv_dim[i], self.config.out_conv_dim[i], self.config.out_conv_filters[i], in_filters),
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            out_shape = tf.stack([batch_size, prev_shape[1] * self.config.out_conv_stride[i],
                         prev_shape[2] * self.config.out_conv_stride[i], self.config.out_conv_filters[i]])
            prev_shape = (None, prev_shape[1] * self.config.out_conv_stride[i],
                         prev_shape[2] * self.config.out_conv_stride[i], self.config.out_conv_filters[i])
            prev_output = tf.nn.conv2d_transpose(prev_output, W, out_shape,
                                                 [1, self.config.out_conv_stride[i], self.config.out_conv_stride[i], 1],
                                                 padding='SAME')
        return prev_output

    def add_out_fc(self, prev_output):
        prev_output = tf.layers.dense(prev_output,
                                      self.config.im_height * self.config.im_width * self.config.im_channels,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(), name='fc_out')
        prev_output = tf.layers.dropout(prev_output, rate=self.config.fc_dropout, training=self.is_train)
        return tf.reshape(prev_output, (-1, self.config.im_height, self.config.im_width, self.config.im_channels))

    def add_prediction_op(self):
        prev_output = self.add_in_convolution(self.image_in)
        prev_output = self.add_in_fc(tf.contrib.layers.flatten(prev_output))
        if self.config.out_conv_layers > 0:
            prev_output = self.add_fixed_size_embed(prev_output)
            if self.config.use_transpose:
                return self.add_out_deconvolution(prev_output)
            else:
                return self.add_out_unconvolution(prev_output)
        else:
            return self.add_out_fc(prev_output)

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
            feed_dict[self.is_train] = True
        else:
            feed_dict[self.is_train] = False
        return feed_dict
