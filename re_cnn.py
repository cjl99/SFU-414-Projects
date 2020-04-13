from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
if float(tf.__version__[0:3]) >= 2:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()


class Model():
    def __init__(self, train, common_params, net_params):
        self.train = train

        self.batch_size = common_params['batch_size']
        self.weight_decay = 0.0  # ?
        self.weight_decay = float(net_params['weight_decay'])  # ?

        # ?
        self.CLASS_MAP_R = tf.constant(np.asarray([32 * i + 16 for i in range(8)] * 64), dtype=tf.float32)
        self.CLASS_MAP_G = tf.constant(np.asarray([32 * int(i / 8) + 16 for i in range(64)] * 8), dtype=tf.float32)
        self.CLASS_MAP_B = tf.constant(np.asarray([32 * int(i / 64) + 16 for i in range(512)]), dtype=tf.float32)
        self.input_shape = 256

        # Create the neural network

    def conv_net(self, data_l):
        # Define a scope for reusing the variables
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            x = data_l
            x = tf.reshape(x, [-1, self.input_shape, self.input_shape, 1])
            # -----------------------conv1--------------------------
            # Convolution Layer with 64 filters and a kernel size of
            conv1_1 = self.conv2d('conv1_1', x, [3, 3, 1, 64], stride=1, wd=self.weight_decay)
            # Convolution Layer with 64 filters and a kernel size of
            conv1_2 = self.conv2d("conv1_2", conv1_1, [3, 3, 64, 64], stride=2, wd=self.weight_decay)
            conv1 = tf.contrib.layers.batch_norm(conv1_2, center=True, scale=True, updates_collections=None,
                                                 is_training=self.train, trainable=True, scope="conv1_3")

            # -----------------------conv2----------------------------
            # Convolution Layer with 128 filters and a kernel size of 3
            conv2_1 = self.conv2d('conv2_1', conv1, [3, 3, 64, 128], stride=1, wd=self.weight_decay)
            conv2_2 = self.conv2d("conv2_2", conv2_1, [3, 3, 128, 128], stride=2, wd=self.weight_decay)
            conv2 = tf.contrib.layers.batch_norm(conv2_2, center=True, scale=True, updates_collections=None,
                                                 is_training=self.train, trainable=True, scope="conv2")

            # -----------------------conv3----------------------------
            # Convolution Layer with 256 filters and a kernel size of 3
            conv3_1 = self.conv2d("conv3_1", conv2, [3, 3, 128, 256], stride=1, dilation=1, relu=True, wd=self.weight_decay)
            conv3_2 = self.conv2d("conv3_2", conv3_1, [3, 3, 256, 256], stride=1, dilation=1, relu=True,
                                  wd=self.weight_decay)
            conv3_3 = self.conv2d("conv3_3", conv3_2, [3, 3, 256, 256], stride=2, dilation=1, relu=True,
                                  wd=self.weight_decay)
            conv3 = tf.contrib.layers.batch_norm(conv3_3, center=True, scale=True, updates_collections=None,
                                                 is_training=self.train, trainable=True, scope="conv3")

            # ----------------------conv4-----------------------------
            # Convolution Layer with 512 filters and a kernel size of 3
            conv4_1 = self.conv2d("conv4_1", conv3, [3, 3, 256, 512], stride=1, dilation=1, relu=True, wd=self.weight_decay)
            conv4_2 = self.conv2d("conv4_2", conv4_1, [3, 3, 512, 512], stride=1, dilation=1, relu=True,
                                  wd=self.weight_decay)
            conv4_3 = self.conv2d("conv4_3", conv4_2, [3, 3, 512, 512], stride=1, dilation=1, relu=True,
                                  wd=self.weight_decay)
            conv4 = tf.contrib.layers.batch_norm(conv4_3, center=True, scale=True, updates_collections=None,
                                                 is_training=self.train, trainable=True, scope="conv4")

            # ----------------------conv5-----------------------------

            conv5_1 = self.conv2d("conv5_1", conv4, [3, 3, 512, 512], stride=1, dilation=2, relu=True, wd=self.weight_decay)
            conv5_2 = self.conv2d("conv5_2", conv5_1, [3, 3, 512, 512], stride=1, dilation=2, relu=True,
                                  wd=self.weight_decay)
            conv5_3 = self.conv2d("conv5_3", conv5_2, [3, 3, 512, 512], stride=1, dilation=2, relu=True,
                                  wd=self.weight_decay)
            conv5 = tf.contrib.layers.batch_norm(conv5_3, center=True, scale=True, updates_collections=None,
                                                 is_training=self.train, trainable=True, scope="conv5")

            # ----------------------conv6-------------------------------
            conv6_1 = self.conv2d("conv6_1", conv5, [3, 3, 512, 512], stride=1, dilation=2, relu=True, wd=self.weight_decay)
            conv6_2 = self.conv2d("conv6_2", conv6_1, [3, 3, 512, 512], stride=1, dilation=2, relu=True,
                                  wd=self.weight_decay)
            conv6_3 = self.conv2d("conv6_3", conv6_2, [3, 3, 512, 512], stride=1, dilation=2, relu=True,
                                  wd=self.weight_decay)
            conv6 = tf.contrib.layers.batch_norm(conv6_3, center=True, scale=True, updates_collections=None,
                                                 is_training=self.train, trainable=True, scope="conv6")

            # ---------------------conv7---------------------------------
            conv7_1 = self.conv2d("conv7_1", conv6, [3, 3, 512, 512], stride=1, dilation=1, relu=True, wd=self.weight_decay)
            conv7_2 = self.conv2d("conv7_2", conv7_1, [3, 3, 512, 512], stride=1, dilation=1, relu=True,
                                  wd=self.weight_decay)
            conv7_3 = self.conv2d("conv7_3", conv7_2, [3, 3, 512, 512], stride=1, dilation=1, relu=True,
                                  wd=self.weight_decay)
            conv7 = tf.contrib.layers.batch_norm(conv7_3, center=True, scale=True, updates_collections=None,
                                                 is_training=self.train, trainable=True, scope="conv7")
            # --------------------conv8----------------------------------
            conv8_1 = self.reverse_conv2d("conv8_1", conv7, [4, 4, 512, 256], stride=2,
                                          wd=self.weight_decay)
            conv8_2 = self.conv2d("conv8_2", conv8_1, [3, 3, 256, 256], stride=1, dilation=1, relu=True,
                                  wd=self.weight_decay)
            conv8_3 = self.conv2d("conv8_3", conv8_2, [3, 3, 256, 256], stride=1, dilation=1, relu=True,
                                  wd=self.weight_decay)

            # ------------------pro_pred----------------------------------
            self.conv8_313 = self.conv2d("conv8_313", conv8_3, [1, 1, 256, 512], stride=1, dilation=1, wd=self.weight_decay)
            # ------------------upsampling--------------------------------
            self.upSample = self.reverse_conv2d("upSample", self.conv8_313, [4, 4, 512, 512], stride=4,
                                         wd=self.weight_decay)
            # self.output = tf.nn.softmax(self.upSample)
            output = tf.nn.softmax(self.upSample)

        return output

    def conv2d(self, scope, input, kernel_size,  stride=1, dilation=1, relu=True, wd=0):
        with tf.variable_scope(scope) as scope:
            kernel = tf.get_variable(name='weights',
                                     shape=kernel_size,
                                     initializer=tf.truncated_normal_initializer(stddev=5e-2,  dtype=tf.float32),
                                     dtype=tf.float32)
            if wd is not None:  # ?
                weight_decay = tf.multiply(tf.nn.l2_loss(kernel), wd, name='weight_loss')
                tf.add_to_collection('losses', weight_decay)
            if dilation == 1:
                conv = tf.nn.conv2d(input, kernel, [1, stride, stride, 1], padding='SAME')
            else:
                conv = tf.nn.atrous_conv2d(input, kernel, dilation, padding='SAME')

            biases = tf.get_variable('biases', kernel_size[3:], initializer=tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            if relu:
                conv1 = tf.nn.relu(bias)
            else:
                conv1 = bias
        return conv1
    # ?
    def reverse_conv2d(self, scope, input, kernel_size, stride=1, wd=0):
        pad_size = int((kernel_size[0] - 1) / 2)
        # input = tf.pad(input, [[0,0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], "CONSTANT")
        batch_size, height, width, in_channel = [int(i) for i in input.get_shape()]
        out_channel = kernel_size[3]
        kernel_size = [kernel_size[0], kernel_size[1], kernel_size[3], kernel_size[2]]
        output_shape = [batch_size, height * stride, width * stride, out_channel]
        with tf.variable_scope(scope) as scope:
            kernel = tf.get_variable(name='weights',
                                     shape=kernel_size,
                                     initializer=tf.truncated_normal_initializer(stddev=5e-2,  dtype=tf.float32),
                                     dtype=tf.float32)
            if wd is not None:  # ?
                weight_decay = tf.multiply(tf.nn.l2_loss(kernel), wd, name='weight_loss')
                tf.add_to_collection('losses', weight_decay)
            deconv = tf.nn.conv2d_transpose(input, kernel, output_shape, [1, stride, stride, 1], padding='SAME')

            biases = tf.get_variable('biases', (out_channel), initializer=tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(deconv, biases)
            deconv1 = tf.nn.relu(bias)

        return deconv1

    def loss_update(self, upSample, gt_ab_313): #, prior_boost_nongray=None):
        flat_conv8_313 = tf.reshape(upSample, [-1, 512])
        flat_gt_ab_313 = tf.reshape(gt_ab_313, [-1, 512])
        g_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=flat_conv8_313, labels=flat_gt_ab_313)) / (int(self.batch_size))

        # tf.summary.scalar('weight_loss', tf.add_n(tf.get_collection('losses', scope=scope)))
        dl2c = tf.gradients(g_loss, upSample)
        dl2c = tf.stop_gradient(dl2c)

        # new_loss = tf.reduce_sum(dl2c * upSample * prior_boost_nongray)
        new_loss = tf.reduce_sum(dl2c * upSample)
        # self.update = tf.train.AdamOptimizer(self.lr).minimize(self.g_loss)
        return new_loss, g_loss

    # ----------------------generate the img from probability---------------------
    def probability2img(self, probTensor, color_space='rgb'):
        eps = 1e-4
        output_dim = 256
        TEMPERATURE = 0.38
        batch_sz = tf.shape(probTensor)[0]

        unnormalized = tf.exp((tf.log(probTensor)) / TEMPERATURE)
        probabilities = unnormalized / tf.reduce_sum(unnormalized, axis=2, keep_dims=True)

        if color_space == 'rgb':
            out_img = tf.stack((tf.reduce_sum(self.CLASS_MAP_R * probabilities, axis=2),
                                tf.reduce_sum(self.CLASS_MAP_G * probabilities, axis=2),
                                tf.reduce_sum(self.CLASS_MAP_B * probabilities, axis=2)), axis=2)

            out_img = tf.reshape(out_img, shape=[batch_sz, output_dim, output_dim, 3])
        return out_img


