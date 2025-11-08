# models/resnet_pathmnist_tf2.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

# TF2 runtime, TF1 graph mode (HCSFL expects sessions/feeds)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from models.cnn_abstract import ModelCNNAbstract

# Use tf.keras from Keras 3 stack
from tensorflow import keras
KL = keras.layers
KI = keras.initializers


def conv3x3(x, filters, stride=1, training=True):
    x = KL.Conv2D(
        filters, kernel_size=3, strides=stride, padding='same', use_bias=False,
        kernel_initializer=KI.VarianceScaling(mode='fan_out', distribution='truncated_normal')
    )(x)
    x = KL.BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=training)
    return tf.nn.relu(x)


def conv1x1(x, filters, stride=1, training=True):
    x = KL.Conv2D(
        filters, kernel_size=1, strides=stride, padding='same', use_bias=False,
        kernel_initializer=KI.VarianceScaling(mode='fan_out', distribution='truncated_normal')
    )(x)
    x = KL.BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=training)
    return x


def basic_block(inp, filters, stride, training=True):
    """Pytorch BasicBlock analog (expansion=1)"""
    identity = inp

    out = KL.Conv2D(
        filters, 3, strides=stride, padding='same', use_bias=False,
        kernel_initializer=KI.VarianceScaling(mode='fan_out', distribution='truncated_normal')
    )(inp)
    out = KL.BatchNormalization(momentum=0.9, epsilon=1e-5)(out, training=training)
    out = tf.nn.relu(out)

    out = KL.Conv2D(
        filters, 3, strides=1, padding='same', use_bias=False,
        kernel_initializer=KI.VarianceScaling(mode='fan_out', distribution='truncated_normal')
    )(out)
    out = KL.BatchNormalization(momentum=0.9, epsilon=1e-5)(out, training=training)

    # downsample if shape mismatch (stride != 1 or channels differ)
    in_ch = inp.shape.as_list()[-1]
    if (stride != 1) or (in_ch != filters):
        identity = KL.Conv2D(
            filters, 1, strides=stride, padding='same', use_bias=False,
            kernel_initializer=KI.VarianceScaling(mode='fan_out', distribution='truncated_normal')
        )(inp)
        identity = KL.BatchNormalization(momentum=0.9, epsilon=1e-5)(identity, training=training)

    out = tf.nn.relu(out + identity)
    return out


def make_layer(x, filters, blocks, stride, training=True):
    x = basic_block(x, filters, stride, training=training)
    for _ in range(1, blocks):
        x = basic_block(x, filters, 1, training=training)
    return x


class ModelResNetPathMNISTTF(ModelCNNAbstract):
    """
    ResNet-18 style backbone adapted for PathMNIST (28x28x3) with 9 classes.
    Implements HCSFL flat-weight & flat-gradient API in TF1-graph mode,
    but builds layers with tf.keras (Keras 3 on TF2.20).
    """

    def __init__(self, n_classes=9):
        super().__init__()
        self.n_classes = n_classes
        self.graph_created = False
        self._last_loss_val = None

    def create_graph(self, learning_rate=None):
        # ---- placeholders
        self.x  = tf.placeholder(tf.float32, shape=[None, 28*28*3])   # flattened inputs
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.n_classes])

        x_img = tf.reshape(self.x, [-1, 28, 28, 3])
        training = True  # HCSFL uses this only for training passes

        # ---- first conv (3 -> 32)
        x = KL.Conv2D(
            32, kernel_size=3, strides=1, padding='same', use_bias=False,
            kernel_initializer=KI.VarianceScaling(mode='fan_out', distribution='truncated_normal')
        )(x_img)
        x = KL.BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=training)
        x = tf.nn.relu(x)

        # ---- ResNet layers (2,2,2,2) with channels 32,64,128,256 and stride on first block except layer1
        x = make_layer(x,  32, blocks=2, stride=1, training=training)
        x = make_layer(x,  64, blocks=2, stride=2, training=training)
        x = make_layer(x, 128, blocks=2, stride=2, training=training)
        x = make_layer(x, 256, blocks=2, stride=2, training=training)

        # ---- GAP + FC
        x = KL.GlobalAveragePooling2D()(x)        # shape [B, 256]
        logits = KL.Dense(
            self.n_classes,
            kernel_initializer=KI.VarianceScaling(mode='fan_out', distribution='truncated_normal')
        )(x)

        # ---- losses/metrics
        self.probs = tf.nn.softmax(logits)
        self.loss_tensor = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=logits)
        )

        # ---- optimizer + gradient op over trainable vars
        lr = learning_rate if learning_rate else 1e-3
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        # NOTE: we build gradients w.r.t. *trainable vars* exactly once
        self.trainable_vars = tf.trainable_variables()
        grads = tf.gradients(self.loss_tensor, self.trainable_vars)
        grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, self.trainable_vars)]

        # apply gradients op (kept for compatibility—HCSFL does manual w update on client)
        self.optimizer_op = optimizer.minimize(self.loss_tensor)

        # ---- accuracy
        correct = tf.equal(tf.argmax(self.probs, 1), tf.argmax(self.y_, 1))
        self.acc_tensor = tf.reduce_mean(tf.cast(correct, tf.float32))

        # ---- flat weights + assign ops (to interop with server/client using flat w vectors)
        self.var_shapes = [v.shape.as_list() for v in self.trainable_vars]
        self.var_sizes  = [int(np.prod(s)) for s in self.var_shapes]
        self.total_params = int(np.sum(self.var_sizes))

        self.flat_weights = tf.concat([tf.reshape(v, [-1]) for v in self.trainable_vars], axis=0)

        self._assign_phs = [tf.placeholder(v.dtype, shape=s) for v, s in zip(self.trainable_vars, self.var_shapes)]
        self._assign_ops = [tf.assign(v, p) for v, p in zip(self.trainable_vars, self._assign_phs)]

        # ---- flat gradient op
        self.flat_grad = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)

        # ---- session init (GPU friendly)
        cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        cfg.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cfg)
        self.sess.run(tf.global_variables_initializer())

        self.graph_created = True

    # ---------- helpers for flat weights ----------
    def _split_flat(self, w_flat):
        out, offset = [], 0
        for size, shape in zip(self.var_sizes, self.var_shapes):
            chunk = w_flat[offset:offset+size]
            out.append(chunk.reshape(shape))
            offset += size
        return out

    def _assign_from_flat(self, w_flat):
        chunks = self._split_flat(np.asarray(w_flat, dtype=np.float32))
        feed = {ph: arr for ph, arr in zip(self._assign_phs, chunks)}
        self.sess.run(self._assign_ops, feed_dict=feed)

    # ---------- API used by HCSFL client ----------
    def gradient(self, train_image, train_label, w, train_indices):
        if train_indices is None:
            x = train_image
            y = train_label
        else:
            x = train_image[train_indices]
            y = train_label[train_indices]

        # assign incoming weights
        self._assign_from_flat(w)

        # compute flat gradient and cache current loss
        grad_val, loss_val = self.sess.run(
            [self.flat_grad, self.loss_tensor],
            feed_dict={self.x: x, self.y_: y}
        )
        self._last_loss_val = float(loss_val)
        return grad_val

    def loss_from_prev_gradient_computation(self):
        return self._last_loss_val

    def loss(self, train_image, train_label, w, train_indices):
        if train_indices is None:
            x = train_image
            y = train_label
        else:
            x = train_image[train_indices]
            y = train_label[train_indices]
        self._assign_from_flat(w)
        return float(self.sess.run(self.loss_tensor, feed_dict={self.x: x, self.y_: y}))

    # convenience
    def get_weights_flat(self):
        return self.sess.run(self.flat_weights)