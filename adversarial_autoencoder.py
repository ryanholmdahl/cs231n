import tensorflow as tf

from model import Model
from utils.layers import unpool
from utils.util import show_image_example


class ModularGenerator(Model):
    def add_placeholders(self):
        input_dims = (None, self.config.im_height, self.config.im_width, self.config.im_channels)
        self.image_in = tf.placeholder(tf.float32, shape=input_dims)
        self.truth_in = tf.placeholder(tf.float32, input_dims)
        self.global_step = tf.Variable(0, trainable=False)
        self.is_train = tf.placeholder(tf.bool, shape=())
        self.image_style = None

    def add_in_convolution(self, prev_output, maxpooling=True):
        for i in range(self.config.in_conv_layers):
            layer_name = 'inconv.{}'.format(i + 1)
            if self.config.model_name != '':
                layer_name = '{}.{}'.format(self.config.model_name, layer_name)
            prev_output = tf.layers.conv2d(prev_output, self.config.in_conv_filters[i], self.config.in_conv_dim[i],
                                           strides=self.config.in_conv_stride[i], activation=tf.nn.relu,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           name=layer_name)
            if maxpooling:
                prev_output = tf.layers.max_pooling2d(prev_output, 2, strides=2)
        return prev_output

    def add_in_fc(self, prev_output):
        for i in range(self.config.fc_layers):
            layer_name = 'fc.{}'.format(i + 1)
            if self.config.model_name != '':
                layer_name = '{}.{}'.format(self.config.model_name, layer_name)
            try:
                activation_func = self.config.fc_activation_funcs[i]
            except AttributeError:
                activation_func = tf.nn.relu
            prev_output = tf.layers.dense(prev_output, self.config.fc_dim[i], activation=activation_func,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name=layer_name)
        try:
            if self.config.normalize_input:
                prev_output = tf.layers.batch_normalization(prev_output, training=True)
        except AttributeError:
            pass
        return prev_output

    def add_fixed_size_embed(self, prev_output):
        layer_name = 'embed'
        if self.config.model_name != '':
            layer_name = '{}.{}'.format(self.config.model_name, layer_name)
        scaling_factor = 1
        for stride in self.config.out_conv_stride:
            scaling_factor *= stride
        embed_width = int(self.config.im_width / scaling_factor)
        embed_height = int(self.config.im_height / scaling_factor)
        embed_dim = embed_width * embed_height * self.config.embed_channels
        try:
            activation_func = self.config.embed_activation_func
        except AttributeError:
            activation_func = tf.nn.relu
        image_embed = tf.layers.dense(prev_output, embed_dim, activation=activation_func,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(), name=layer_name)
        embed_out_dim = (-1, embed_height, embed_width, self.config.embed_channels)
        return tf.reshape(image_embed, embed_out_dim)

    def add_out_unconvolution(self, prev_output):
        for i in range(self.config.out_conv_layers):
            layer_name = 'outconv.{}'.format(i + 1)
            if self.config.model_name != '':
                layer_name = '{}.{}'.format(self.config.model_name, layer_name)
            prev_unpooled = unpool(prev_output, self.config.out_conv_stride[i])
            prev_output = tf.layers.conv2d(prev_unpooled, self.config.out_conv_filters[i],
                                           self.config.out_conv_dim[i], activation=tf.nn.relu, padding='SAME',
                                           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           name=layer_name)
        return prev_output

    def add_out_deconvolution(self, prev_output):
        prev_dyn_shape = tf.shape(prev_output)
        batch_size = prev_dyn_shape[0]
        prev_shape = prev_output.get_shape().as_list()
        for i in range(self.config.out_conv_layers):
            try:
                activation_func = self.config.out_conv_activation_func[i]
            except AttributeError:
                activation_func = tf.nn.relu
            if i > 0:
                prev_output = activation_func(prev_output)
            filter_name = 'deconvW.{}'.format(i + 1)
            if self.config.model_name != '':
                filter_name = '{}.{}'.format(self.config.model_name, filter_name)
            in_filters = self.config.out_conv_filters[i - 1] if i > 0 else prev_output.get_shape()[3]
            W = tf.get_variable(filter_name, shape=(
                self.config.out_conv_dim[i], self.config.out_conv_dim[i], self.config.out_conv_filters[i], in_filters),
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            out_shape = tf.stack([batch_size, prev_shape[1] * self.config.out_conv_stride[i],
                                  prev_shape[2] * self.config.out_conv_stride[i], self.config.out_conv_filters[i]])
            prev_shape = [None, prev_shape[1] * self.config.out_conv_stride[i],
                          prev_shape[2] * self.config.out_conv_stride[i], self.config.out_conv_filters[i]]
            prev_output = tf.nn.conv2d_transpose(prev_output, W, out_shape,
                                                 [1, self.config.out_conv_stride[i], self.config.out_conv_stride[i], 1],
                                                 padding='SAME')
        return prev_output

    def add_out_fc(self, prev_output, ):
        layer_name = 'fc_out'
        if self.config.model_name != '':
            layer_name = '{}.{}'.format(self.config.model_name, layer_name)
        prev_output = tf.layers.dense(prev_output,
                                      self.config.im_height * self.config.im_width * self.config.im_channels,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(), name=layer_name)
        prev_output = tf.layers.dropout(prev_output, rate=self.config.fc_dropout, training=self.is_train)
        return tf.reshape(prev_output, (-1, self.config.im_height, self.config.im_width, self.config.im_channels))

    def add_prediction_op(self, input_logits=None, style_concat_input=None, style_input=None, **kwargs):
        with tf.variable_scope(self.config.model_name) as scope:
            if style_input is None:
                prev_output = self.add_in_convolution(input_logits)
                prev_output = self.add_in_fc(tf.contrib.layers.flatten(prev_output))
                self.image_style = tf.layers.dense(prev_output, self.config.style_dim,
                                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                   name="style")
            else:
                self.image_style = style_input
            if style_concat_input is not None:
                prev_output = tf.concat((self.image_style, style_concat_input), axis=1)
            else:
                prev_output = self.image_style
            if self.config.out_conv_layers > 0:
                prev_output = self.add_fixed_size_embed(prev_output)
                if self.config.use_transpose:
                    result = self.add_out_deconvolution(prev_output)
                else:
                    result = self.add_out_unconvolution(prev_output)
            else:
                result = self.add_out_fc(prev_output)
            # except ValueError:
            #     scope.reuse_variables()
            #     if style_input is None:
            #         prev_output = self.add_in_convolution(input_logits)
            #         prev_output = self.add_in_fc(tf.contrib.layers.flatten(prev_output))
            #         self.image_style = tf.layers.dense(prev_output, self.config.style_dim,
            #                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                                            name="style")
            #     else:
            #         self.image_style = style_input
            #     if style_concat_input is not None:
            #         prev_output = tf.concat((self.image_style, style_concat_input), axis=1)
            #     else:
            #         prev_output = self.image_style
            #     if self.config.out_conv_layers > 0:
            #         prev_output = self.add_fixed_size_embed(prev_output)
            #         if self.config.use_transpose:
            #             result = self.add_out_deconvolution(prev_output)
            #         else:
            #             result = self.add_out_unconvolution(prev_output)
            #     else:
            #         result = self.add_out_fc(prev_output)
            return result

    def add_loss_op(self, **kwargs):
        return tf.reduce_mean(kwargs['preds'] - kwargs['rotated_imgs']) ** 2 / 2

    def add_training_op(self, loss):
        learning_rate = tf.train.exponential_decay(self.config.lr, self.global_step, self.config.lr_decay_steps,
                                                   self.config.lr_decay, staircase=True)
        return tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=self.global_step)

    def create_feed_dict(self, inputs_batch, outputs_batch=None, **kwargs):
        feed_dict = {self.image_in: inputs_batch}
        if outputs_batch is not None:
            feed_dict[self.truth_in] = outputs_batch
            feed_dict[self.is_train] = True
        else:
            feed_dict[self.is_train] = False
        return feed_dict

    def demo(self, sess, train_data, dev_data, demo_count):
        for i in range(demo_count):
            show_image_example(sess, self, train_data[0][i], train_data[1][i], name='train/fig_{}.png'.format(i + 1))
            show_image_example(sess, self, dev_data[0][i], dev_data[1][i], name='dev/fig_{}.png'.format(i + 1))

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(preds=self.pred, rotated_imgs=self.truth_in)
        self.train_op = self.add_training_op(self.loss)
