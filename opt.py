#!python

import csv
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops

BATCH_SIZE = 200
LOSS = 'loss'

### HOW THIS WORKS
#
# generate_graph generates an MNIST CNN model graph. The parts that do
# work are the same as in the Tensorflow documentation at
# https://www.tensorflow.org/tutorials/layers.
#
# It also adds the ability to insert monitoring ops after every
# layer. The 'run_test' function will print these out after every
# step, which makes it possible to investigate different parts of the
# training process. 'generate_and_test' is a convenience function
# which encapsulates the whole workflow.


def generate_mnist_cnn(data, labels):
    """Generate the graph.

    Args:
      data: the input data tensor.
      labels: the input labels tensor.

    Returns: a tuple of (intermediates, loss). The loss is an overall
      loss for the graph.
    """
    intermediates = []

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
    intermediates.append(conv1)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    intermediates.append(conv2)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                            activation=tf.nn.relu)

    intermediates.append(dense)

    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    intermediates.append(logits)

    # Calculate Overall Loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    return intermediates, loss


## Cosine loss-based monitor ops

def pair_cosine_loss(tensor, labels=None, num_pairs=None, **kwargs):
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

    return {'abs(diff)': loss1,
            'abs(all)': loss2,
            'all': loss3,
            'equal': loss4}


## Make sure the model isn't sending all the vectors to zero.

def norm_ratio(tensor, num_pairs=None, in_tensor=None, **kwargs):
    in_flat = tf.reshape(in_tensor, [num_pairs * 2, -1])
    out_flat = tf.reshape(tensor, [num_pairs * 2, -1])

    ratios = tf.divide(tf.norm(out_flat, axis=1),
                       tf.norm(in_flat, axis=1))

    return {'norm ratio min': tf.reduce_min(ratios),
            'norm ratio max': tf.reduce_max(ratios)}


def mean_based(tensor, labels=None, num_pairs=None, **kwargs):
    num_rows = num_pairs * 2
    flat_tensor = tf.reshape(tensor, [num_rows, -1])

    rows_by_label = [
        tf.boolean_mask(flat_tensor, tf.equal(labels, i))
        for i in range(10)]

    means_by_label = [
        tf.reduce_mean(rows_by_label[i], axis=0)
        for i in range(10)]

    stacked_means = tf.stack(means_by_label)
    means_for_rows = tf.gather(stacked_means, labels)

    return {
        'not normalized': tf.reduce_sum(
            tf.subtract(flat_tensor, means_for_rows)),
        'cosine': tf.reduce_sum(
            tf.losses.cosine_distance(
                tf.nn.l2_normalize(flat_tensor, axis=1),
                tf.nn.l2_normalize(means_for_rows, axis=1),
                axis=1))}


MONITORS = [pair_cosine_loss, norm_ratio, mean_based]


def generate_graph(features, labels):
    """Generate a graph including the MNIST CNN model and monitor ops.

    The returned monitor ops will include a special op called 'loss'
    with the overall loss function.
    """

    intermediates, loss = generate_mnist_cnn(features, labels)

    monitor_ops = {}

    last_layer = features
    layer_num = 0

    for tensor in intermediates:
        for func in MONITORS:
            ops = func(tensor,
                       labels=labels,
                       num_pairs=BATCH_SIZE // 2,
                       in_tensor = last_layer)
            for op_name, op in ops.items():
                monitor_ops[op_name + str(layer_num)] = op
        last_layer = tensor
        layer_num += 1

    monitor_ops[LOSS] = loss

    return monitor_ops


COLUMNS = ['step', 'time', 'strategy', 'monitor', 'value']

def train_one_op(sess, optimizer, monitor_ops, loss_name, writer,
                 fp, strategy, base_time, num_iterations=1000):
    train = optimizer.minimize(monitor_ops[loss_name])

    for i in range(num_iterations):
        if i % 10 == 0:
            print('Starting step', i)

        _, monitors = sess.run((train, monitor_ops))
        step_time = time.perf_counter()
        for monitor, value in monitors.items():
            writer.writerow({'step': i,
                             'time': step_time - base_time,
                             'strategy': strategy,
                             'monitor': monitor,
                             'value': value})
        fp.flush()


def generate_and_test(features, labels, sess):
    # loss, monitor_ops = generate_graph(
    #     features, labels,
    #     [pair_cosine_loss, norm_ratio, mean_based])

    monitor_ops = generate_graph(features, labels)

    optimizer = tf.train.GradientDescentOptimizer(0.001)

    sess.run(tf.global_variables_initializer())

    with open('out.csv', 'w') as fp:
        writer = csv.DictWriter(fp, COLUMNS)
        writer.writeheader()
        # start_default = time.perf_counter()
        # train_one_op(sess, optimizer, monitor_ops, LOSS, writer, fp,
        #              'default', start_default)
        start_cosine = time.perf_counter()
        train_one_op(sess, optimizer, monitor_ops, 'cosine0', writer, fp,
                     'cosine', start_cosine, num_iterations=100)
        train_one_op(sess, optimizer, monitor_ops, 'cosine1', writer, fp,
                     'cosine', start_cosine, num_iterations=100)
        train_one_op(sess, optimizer, monitor_ops, 'cosine2', writer, fp,
                     'cosine', start_cosine, num_iterations=100)
        train_one_op(sess, optimizer, monitor_ops, LOSS, writer, fp,
                     'cosine', start_cosine)


def get_features_labels():
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

  dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
  dataset_2 = dataset.shuffle(55000).repeat().batch(BATCH_SIZE)
  features, labels = dataset_2.make_one_shot_iterator().get_next()

  return features, labels


def main():
  features, labels = get_features_labels()

  sess = tf.Session()
  generate_and_test(features, labels, sess)


if __name__ == '__main__':
    main()
