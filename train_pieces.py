#!python

import tensorflow as tf
import numpy as np

from tensorflow.python.ops import math_ops

BATCH_SIZE = 200

def pair_cosine_loss(tensor, labels, num_pairs):
    """Generate a cosine loss function for a tensor of pairs.

    Args:
      tensor: a tensor, with dimension 0 equal to num_pairs * 2
      labels: a one-dimensional tensor with shape num_pairs * 2
      num_pairs: integer. The number of input pairs.

    Returns: a tensor computing my custom cosine loss function.
    """
    flat = tf.reshape(tensor, [num_pairs * 2, -1])

    # Flat is (2 * num_pairs) x n, where n is the number of values
    # representing each input vector.

    # Separate the pairs into different tensors
    flat_1 = tf.slice(flat, [0, 0], [num_pairs, -1])
    flat_2 = tf.slice(flat, [num_pairs, 0], [-1, -1])

    # Compute cosine distance between pairs. The built-in
    # cosine_distance only accepts a single axis, not a vector, so
    # it's inlined here.
    conv_1_normal = tf.nn.l2_normalize(flat_1, [1])
    conv_2_normal = tf.nn.l2_normalize(flat_2, [1])
    conv_1_float = math_ops.to_float(conv_1_normal)
    conv_2_float = math_ops.to_float(conv_2_normal)
    radial_diffs = math_ops.multiply(conv_1_float, conv_2_float)
    # diffs = 1 - math_ops.reduce_sum(radial_diffs, axis=[1], keep_dims=True)
    diffs = math_ops.reduce_sum(radial_diffs, axis=[1], keep_dims=True)

    # diffs has shape [num_pairs]

    labels_1 = tf.slice(labels, [0], [num_pairs])
    labels_2 = tf.slice(labels, [num_pairs], [-1])

    # labels_1 and labels_2 have shape [num_pairs]

    # Three variations on the loss function.
    # 1: absolute value, only different labels

    # Throw out pairs where the labels match, because we just want
    # different labels to have different outputs.
    labels_eq = math_ops.equal(labels_1, labels_2)

    # Mixing check
    num_diff = tf.count_nonzero(labels_eq)

    different_diffs = tf.boolean_mask(diffs, tf.logical_not(labels_eq))
    loss1 = math_ops.reduce_sum(tf.abs(different_diffs))

    # 2: absolute value, all labels
    # If labels are equal, flip sign so that minimization makes
    # same-label elements closer, not farther away.
    all_to_minimize = tf.multiply(
        diffs,
        tf.where(labels_eq,
                 x=tf.ones([num_pairs]) * -1,
                 y=tf.ones([num_pairs])))
    loss2 = math_ops.reduce_sum(tf.abs(all_to_minimize))

    # 3: no absolute value, all labels
    loss3 = math_ops.reduce_sum(all_to_minimize)

    # 4: just equal labels. flip sign
    loss4 = math_ops.reduce_sum(tf.boolean_mask(diffs, labels_eq)) * -1

    return [tf.divide(num_diff, num_pairs)], [loss1, loss2, loss3, loss4]


def generate_graph(data, labels):
    """Generate the graph, with multiple loss functions."""

    losses = []
    num_pairs = BATCH_SIZE // 2

    # Data is 2 x N x 28 x 28.
    #  2 is because we want pairs of images
    #  N is the batch size
    #  28 x 28 are the pixels for each image

    # Labels is 2 x N

    # layers.conv2d wants a simple array of images to work with, so
    # here we go
    conv_input = tf.reshape(data, [-1, 28, 28, 1])

    # conv_input is 2N x 28 x 28 x 1

    # Convolutional Layer.
    conv1 = tf.layers.conv2d(
      inputs=conv_input,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    # Shape: 2N x 28 x 28 x 32.

    data, losses1 = pair_cosine_loss(conv1, labels, num_pairs)
    losses.extend(losses1)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    _, losses2 = pair_cosine_loss(conv2, labels, num_pairs)
    losses.extend(losses2)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                            activation=tf.nn.relu)

    _, losses3 = pair_cosine_loss(dense, labels, num_pairs)
    losses.extend(losses3)

    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    # Calculate Overall Loss
    losses.append(
        tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits))

    return data, losses


def get_features_labels():
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

  dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
  dataset_2 = dataset.shuffle(55000).repeat().batch(BATCH_SIZE)
  features, labels = dataset_2.make_one_shot_iterator().get_next()

  return features, labels


def train_last(sess, trains, losses, num_iterations=100):
    ops_to_eval = [trains[-1]] + losses

    for i in range(num_iterations):
        vals = sess.run(tuple(ops_to_eval))
        print('Step:', i, 'losses:', vals[1:])


def main():
  features, labels = get_features_labels()

  data, losses = generate_graph(features, labels)

  optimizer = tf.train.GradientDescentOptimizer(0.001)

  # Choo-choo!
  trains = [optimizer.minimize(loss)
            for loss in losses]

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())


if __name__ == '__main__':
    main()
