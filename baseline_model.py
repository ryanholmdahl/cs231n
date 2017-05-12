import tensorflow as tf


class BaselineModel:
    def __init__(self, params):
        self.initial_lr = params['lr']
        self.lr_decay = params['lr_decay']

        input_dims = (None, params['width'], params['height'], params['channels'])
        self.image_in = tf.placeholder(tf.float32, shape=input_dims)
        prev_output = self.image_in
        for i in range(params['in_conv_layers']):
            prev_output = tf.layers.conv2d(prev_output, params['in_conv_filters'][i], params['in_conv_dim'][i],
                                           strides=params['in_conv_stride'][i],
                                           activation=tf.nn.relu,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
        for i in range(params['fc_layers']):
            prev_output = tf.layers.dense(prev_output, params['fc_dim'][i], activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.image_embed = tf.layers.dense(prev_output, params['embed_dim'], activation=tf.nn.relu,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
        prev_output = self.image_embed
        for i in range(params['out_conv_layers']):
            prev_output = tf.layers.conv2d_transpose(prev_output, params['out_conv_filters'][i],
                                                     params['out_conv_dim'][i],
                                                     strides=params['out_conv_stride'][i], activation=tf.nn.relu,
                                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.image_out = tf.layers.conv2d_transpose(prev_output, params['channels'],
                                                    (params['width'], params['height']),
                                                    kernel_initializer=tf.contrib.xavier_initializer())
