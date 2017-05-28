import tensorflow as tf

from model import Model
from utils.layers import unpool
from utils.util import show_image_example


class ModularGenerator(Model):
    def add_in_convolution(self, prev_output, maxpooling=True):
        for i in range(self.config.in_conv_layers):
            # Create Layer Name
            layer_name = 'inconv.{}'.format(i + 1)
            if self.config.model_name != '':
                layer_name = '{}.{}'.format(self.config.model_name, layer_name)

            # Determine Activation Function
            try:
                activation_func = self.config.in_conv_activation_func[i]
            except AttributeError:
                activation_func = tf.nn.relu

            # Apply 2D Convolution + maxpooling
            prev_output = tf.layers.conv2d(prev_output, self.config.in_conv_filters[i], self.config.in_conv_dim[i],
                                           strides=self.config.in_conv_stride[i], activation=activation_func,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           name=layer_name)
            if maxpooling:
                prev_output = tf.layers.max_pooling2d(prev_output, 2, strides=2)
        return prev_output

    def add_in_fc(self, prev_output):
        for i in range(self.config.fc_layers):
            # Create Layer Name
            layer_name = 'fc.{}'.format(i + 1)
            if self.config.model_name != '':
                layer_name = '{}.{}'.format(self.config.model_name, layer_name)

            # Determine Activation Function
            try:
                activation_func = self.config.fc_activation_funcs[i]
            except AttributeError:
                activation_func = tf.nn.relu

            # Construct Dense Fully Connected Layer
            prev_output = tf.layers.dense(prev_output, self.config.fc_dim[i], activation=activation_func,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name=layer_name)
        return prev_output

    def add_fixed_size_embed(self, prev_output):
        # Create Layer Name
        layer_name = 'embed'
        if self.config.model_name != '':
            layer_name = '{}.{}'.format(self.config.model_name, layer_name)

        # Determine embedding dimensions, necessary to produce the desired output after the ensuing Conv2D Layers
        scaling_factor = 1
        for stride in self.config.out_conv_stride:
            scaling_factor *= stride
        embed_width = int(self.config.im_width / scaling_factor)
        embed_height = int(self.config.im_height / scaling_factor)
        embed_dim = embed_width * embed_height * self.config.embed_channels

        # Determine Activation Function
        try:
            activation_func = self.config.embed_activation_func
        except AttributeError:
            activation_func = tf.nn.relu

        # Construct Image Embedding via Fully Connected Layer
        image_embed = tf.layers.dense(prev_output, embed_dim, activation=activation_func,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(), name=layer_name)
        embed_out_dim = (-1, embed_height, embed_width, self.config.embed_channels)
        return tf.reshape(image_embed, embed_out_dim)

    def add_out_unconvolution(self, prev_output):
        for i in range(self.config.out_conv_layers):
            # Create Layer Name
            layer_name = 'outconv.{}'.format(i + 1)
            if self.config.model_name != '':
                layer_name = '{}.{}'.format(self.config.model_name, layer_name)

            # Apply Unpooling Layer
            prev_unpooled = unpool(prev_output, self.config.out_conv_stride[i])
            prev_output = tf.layers.conv2d(prev_unpooled, self.config.out_conv_filters[i],
                                           self.config.out_conv_dim[i], activation=tf.nn.relu, padding='SAME',
                                           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           name=layer_name)
        return prev_output

    def add_out_deconvolution(self, prev_output):
        # Determine shape of input tensor
        prev_dyn_shape = tf.shape(prev_output)
        batch_size = prev_dyn_shape[0]
        prev_shape = prev_output.get_shape().as_list()

        for i in range(self.config.out_conv_layers):
            # Determine Filter Name
            filter_name = 'deconvW.{}'.format(i + 1)
            if self.config.model_name != '':
                filter_name = '{}.{}'.format(self.config.model_name, filter_name)

            # Determine Activation Function
            try:
                activation_func = self.config.out_conv_activation_func[i]
            except AttributeError:
                activation_func = tf.nn.relu

            # Retrieve Filers and Apply Conv 2D Transpose (Deconvolution)
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
            prev_output = activation_func(prev_output)
        return prev_output

    def add_out_fc(self, prev_output, ):
        # Determine Layer Name
        layer_name = 'fc_out'
        if self.config.model_name != '':
            layer_name = '{}.{}'.format(self.config.model_name, layer_name)

        # Apply Dense (Fully Connected Layer)
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
                prev_output = tf.layers.dense(prev_output, self.config.style_dim,
                                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                   name="style")
                self.image_style = prev_output
            else:
                prev_output = style_input
            if style_concat_input is not None:
                prev_output = tf.concat((prev_output, style_concat_input), axis=1)
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
            return result

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

    def demo(self, sess, train_data, dev_data, demo_count):
        for i in range(demo_count):
            show_image_example(sess, self, train_data[0][i], train_data[1][i], name='train/fig_{}.png'.format(i + 1))
            show_image_example(sess, self, dev_data[0][i], dev_data[1][i], name='dev/fig_{}.png'.format(i + 1))

