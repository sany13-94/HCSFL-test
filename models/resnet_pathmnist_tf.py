# models/resnet_pathmnist_tf.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

# TF1-compat in TF2+ envs
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from models.cnn_abstract import ModelCNNAbstract

# Use tf-keras (2.20) rather than tensorflow.keras
from tf_keras import layers as KL
from tf_keras import initializers as KI



def conv_bn_relu(x, filters, k, s, training=True):
    x = KL.Conv2D(filters, k, strides=s, padding='same', use_bias=False,
                  kernel_initializer=KI.VarianceScaling())(x)
    x = KL.BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=training)
    return tf.nn.relu(x)

def residual_block(x, filters, stride, training=True):
    in_channels = x.get_shape().as_list()[-1]

    y = KL.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False,
                  kernel_initializer=KI.VarianceScaling())(x)
    y = KL.BatchNormalization(momentum=0.9, epsilon=1e-5)(y, training=training)
    y = tf.nn.relu(y)
    y = KL.Conv2D(filters, 3, strides=1, padding='same', use_bias=False,
                  kernel_initializer=KI.VarianceScaling())(y)
    y = KL.BatchNormalization(momentum=0.9, epsilon=1e-5)(y, training=training)

    if (stride != 1) or (in_channels != filters):
        x = KL.Conv2D(filters, 1, strides=stride, padding='same', use_bias=False,
                      kernel_initializer=KI.VarianceScaling())(x)
        x = KL.BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=training)

    return tf.nn.relu(x + y)

def layer_group(x, filters, blocks, first_stride, training=True):
    x = residual_block(x, filters, first_stride, training=training)
    for _ in range(blocks - 1):
        x = residual_block(x, filters, 1, training=training)
    return x


class ModelResNetPathMNIST(ModelCNNAbstract):
    """ResNet-18 style for 28x28x3, 9 classes, TF1-compat + tf_keras. Provides flat-weight ops and gradient API."""

    def __init__(self):
        super().__init__()
        self.graph_created = False
        self._last_loss_val = None

    def create_graph(self, learning_rate=None):
        # ---- placeholders
        self.x  = tf.placeholder(tf.float32, shape=[None, 28*28*3])   # flattened input
        self.y_ = tf.placeholder(tf.float32, shape=[None, 9])         # one-hot labels

        x_img = tf.reshape(self.x, [-1, 28, 28, 3])
        training = True

        # ---- network
        x = conv_bn_relu(x_img, 64, 3, 1, training=training)
        x = layer_group(x,  64, 2, 1, training=training)
        x = layer_group(x, 128, 2, 2, training=training)
        x = layer_group(x, 256, 2, 2, training=training)
        x = layer_group(x, 512, 2, 2, training=training)
      
        x = KL.GlobalAveragePooling2D()(x)

        logits = KL.Dense(9, kernel_initializer=KI.VarianceScaling())(x)

        # ---- losses/metrics
        self.probs = tf.nn.softmax(logits)
        self.loss_tensor = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=logits)
        )

        lr = learning_rate if learning_rate else 1e-3
        self.optimizer_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss_tensor)

        correct = tf.equal(tf.argmax(self.probs, 1), tf.argmax(self.y_, 1))
        self.acc_tensor = tf.reduce_mean(tf.cast(correct, tf.float32))

        # ---- flat weights + assign ops (to interop with server/client using flat w vectors)
        self.trainable_vars = tf.trainable_variables()
        self.var_shapes = [v.shape.as_list() for v in self.trainable_vars]
        self.var_sizes  = [int(np.prod(s)) for s in self.var_shapes]
        self.total_params = int(np.sum(self.var_sizes))

        self.flat_weights = tf.concat([tf.reshape(v, [-1]) for v in self.trainable_vars], axis=0)

        self._assign_phs = [tf.placeholder(v.dtype, shape=s) for v, s in zip(self.trainable_vars, self.var_shapes)]
        self._assign_ops = [tf.assign(v, p) for v, p in zip(self.trainable_vars, self._assign_phs)]

        # ---- flat gradient op (w.r.t. trainable vars)
        grads = tf.gradients(self.loss_tensor, self.trainable_vars)
        # Some grads can be None (e.g., disconnected); replace with zeros
        grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, self.trainable_vars)]
        self.flat_grad = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)

        # ---- session init (GPU friendly)
        cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        cfg.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cfg)
        self.sess.run(tf.global_variables_initializer())

        self.graph_created = True

    # ------------- helpers for flat weight assignment -------------
    def _split_flat(self, w_flat):
        """Split a 1D flat vector into a list of arrays matching var shapes."""
        out = []
        offset = 0
        for size, shape in zip(self.var_sizes, self.var_shapes):
            chunk = w_flat[offset:offset+size]
            out.append(chunk.reshape(shape))
            offset += size
        return out

    def _assign_from_flat(self, w_flat):
        chunks = self._split_flat(np.asarray(w_flat, dtype=np.float32))
        feed = {ph: arr for ph, arr in zip(self._assign_phs, chunks)}
        self.sess.run(self._assign_ops, feed_dict=feed)

    # ------------- API used by client.py -------------
    def gradient(self, train_image, train_label, w, train_indices):
        """Return flat gradient vector for the given minibatch and weights w."""
        if train_indices is None:
            x = train_image
            y = train_label
        else:
            x = train_image[train_indices]
            y = train_label[train_indices]

        # assign incoming weights
        self._assign_from_flat(w)

        # compute gradient and cache current loss
        grad_val, loss_val = self.sess.run(
            [self.flat_grad, self.loss_tensor],
            feed_dict={self.x: x, self.y_: y}
        )
        self._last_loss_val = float(loss_val)
        return grad_val

    def loss_from_prev_gradient_computation(self):
        """Client code calls this right after gradient(); return cached loss."""
        return self._last_loss_val

    def loss(self, train_image, train_label, w, train_indices):
        """Compute scalar loss for given weights and data (no gradient)."""
        if train_indices is None:
            x = train_image
            y = train_label
        else:
            x = train_image[train_indices]
            y = train_label[train_indices]
        self._assign_from_flat(w)
        return float(self.sess.run(self.loss_tensor, feed_dict={self.x: x, self.y_: y}))

    # (optional convenience)
    def get_weights_flat(self):
        return self.sess.run(self.flat_weights)