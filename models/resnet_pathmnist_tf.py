# models/resnet_pathmnist_tf.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # enables TF1 graph + placeholders in TF2
from models.cnn_abstract import ModelCNNAbstract

# Use Keras layers from TF1-compat
from tensorflow.compat.v1.keras import layers as KL
from tensorflow.compat.v1.keras import initializers as KI

def conv_bn_relu(x, filters, k, s, training=True):
    x = KL.Conv2D(
        filters, k, strides=s, padding='same', use_bias=False,
        kernel_initializer=KI.VarianceScaling()
    )(x)
    x = KL.BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=training)
    x = tf.nn.relu(x)
    return x

def residual_block(x, filters, stride, training=True):
    in_channels = x.get_shape().as_list()[-1]

    y = KL.Conv2D(
        filters, 3, strides=stride, padding='same', use_bias=False,
        kernel_initializer=KI.VarianceScaling()
    )(x)
    y = KL.BatchNormalization(momentum=0.9, epsilon=1e-5)(y, training=training)
    y = tf.nn.relu(y)
    y = KL.Conv2D(
        filters, 3, strides=1, padding='same', use_bias=False,
        kernel_initializer=KI.VarianceScaling()
    )(y)
    y = KL.BatchNormalization(momentum=0.9, epsilon=1e-5)(y, training=training)

    if (stride != 1) or (in_channels != filters):
        x = KL.Conv2D(
            filters, 1, strides=stride, padding='same', use_bias=False,
            kernel_initializer=KI.VarianceScaling()
        )(x)
        x = KL.BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=training)

    out = tf.nn.relu(x + y)
    return out

def layer_group(x, filters, blocks, first_stride, training=True):
    x = residual_block(x, filters, first_stride, training=training)
    for _ in range(blocks - 1):
        x = residual_block(x, filters, 1, training=training)
    return x

class ModelResNetPathMNIST(ModelCNNAbstract):
    """ResNet-18 style for 28x28x3, 9 classes (TF1-compat, Keras layers)"""
    def __init__(self):
        super().__init__()

    def create_graph(self, learning_rate=None):
        self.x  = tf.placeholder(tf.float32, shape=[None, 28*28*3])  # flattened input
        self.y_ = tf.placeholder(tf.float32, shape=[None, 9])

        x_img = tf.reshape(self.x, [-1, 28, 28, 3])
        training = True  # this model is only called in training mode for grads; eval uses same graph

        # Small-image stem
        x = conv_bn_relu(x_img, 64, 3, 1, training=training)

        # ResNet-18 blocks: [2,2,2,2] with strides [1,2,2,2]
        x = layer_group(x,  64, 2, 1, training=training)
        x = layer_group(x, 128, 2, 2, training=training)
        x = layer_group(x, 256, 2, 2, training=training)
        x = layer_group(x, 512, 2, 2, training=training)

        # Global avg pool + head
        x = KL.AveragePooling2D(pool_size=4, strides=1, padding='valid')(x)
        x = KL.Flatten()(x)
        logits = KL.Dense(9, kernel_initializer=KI.VarianceScaling())(x)

        self.y = tf.nn.softmax(logits)
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=logits)
        )
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate if learning_rate else 1e-3)
        self.optimizer_op = opt.minimize(self.loss)

        correct = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.acc = tf.reduce_mean(tf.cast(correct, tf.float32))

        self._collect_tf_weights_for_abstract()
        self._session_init()
        self.graph_created = True