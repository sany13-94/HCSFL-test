# models/resnet_pathmnist_tf.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from models.cnn_abstract import ModelCNNAbstract

def conv_bn_relu(x, filters, k, s):
    x = tf.layers.conv2d(x, filters, k, strides=s, padding='same',
                         use_bias=False, kernel_initializer=tf.initializers.variance_scaling())
    x = tf.layers.batch_normalization(x, momentum=0.9, epsilon=1e-5, training=True)
    x = tf.nn.relu(x)
    return x

def residual_block(x, filters, stride):
    in_channels = x.get_shape().as_list()[-1]
    y = tf.layers.conv2d(x, filters, 3, strides=stride, padding='same',
                         use_bias=False, kernel_initializer=tf.initializers.variance_scaling())
    y = tf.layers.batch_normalization(y, momentum=0.9, epsilon=1e-5, training=True)
    y = tf.nn.relu(y)
    y = tf.layers.conv2d(y, filters, 3, strides=1, padding='same',
                         use_bias=False, kernel_initializer=tf.initializers.variance_scaling())
    y = tf.layers.batch_normalization(y, momentum=0.9, epsilon=1e-5, training=True)

    if (stride != 1) or (in_channels != filters):
        x = tf.layers.conv2d(x, filters, 1, strides=stride, padding='same',
                             use_bias=False, kernel_initializer=tf.initializers.variance_scaling())
        x = tf.layers.batch_normalization(x, momentum=0.9, epsilon=1e-5, training=True)

    out = tf.nn.relu(x + y)
    return out

def layer_group(x, filters, blocks, first_stride):
    x = residual_block(x, filters, first_stride)
    for _ in range(blocks - 1):
        x = residual_block(x, filters, 1)
    return x

class ModelResNetPathMNIST(ModelCNNAbstract):
    """ResNet-18 style for 28x28x3, 9 classes"""
    def __init__(self):
        super().__init__()

    def create_graph(self, learning_rate=None):
        self.x = tf.placeholder(tf.float32, shape=[None, 28*28*3])   # flattened input
        self.y_ = tf.placeholder(tf.float32, shape=[None, 9])

        x_img = tf.reshape(self.x, [-1, 28, 28, 3])

        # Stem adapted for small images (3x3, stride1)
        x = conv_bn_relu(x_img, 64, 3, 1)

        # ResNet-18 groups: [2,2,2,2] with strides [1,2,2,2]
        x = layer_group(x,  64, 2, 1)
        x = layer_group(x, 128, 2, 2)
        x = layer_group(x, 256, 2, 2)
        x = layer_group(x, 512, 2, 2)

        x = tf.layers.average_pooling2d(x, pool_size=4, strides=1, padding='valid')
        x = tf.layers.flatten(x)
        logits = tf.layers.dense(x, 9, kernel_initializer=tf.initializers.variance_scaling())

        self.y = tf.nn.softmax(logits)
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=logits)
        )
        self.optimizer_op = tf.train.AdamOptimizer(learning_rate=learning_rate if learning_rate else 1e-3) \
                                .minimize(self.loss)

        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self._collect_tf_weights_for_abstract()
        self._session_init()
        self.graph_created = True