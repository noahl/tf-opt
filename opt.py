#!python

import csv
import tensorflow as tf
import numpy as np

from tensorflow.python.ops import math_ops

BATCH_SIZE = 200

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


def generate_graph(data, labels, monitors):
    """Generate the graph.

    Args:
      data: the input data tensor.
      labels: the input labels tensor.
      monitors: a list of functions. The functions will be applied to
        the intermediate tensors at every stage of the graph, and
        should return a dict of {name, tensor} pairs, to be used as
        monitors.

    Returns: a tuple of (loss, monitor_ops). The loss is an overall
      loss for the graph.

    """

    losses = []
    num_pairs = BATCH_SIZE // 2
    monitor_ops = {}

    def add_monitor_ops(tensor, in_tensor, layer_num):
        for func in monitors:
            ops = func(tensor,
                       labels=labels,
                       num_pairs=num_pairs,
                       in_tensor=in_tensor)
            for op_name, op in ops.items():
                monitor_ops[op_name + str(layer_num)] = op

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

    add_monitor_ops(conv1, conv_input, 0)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    add_monitor_ops(conv2, conv1, 1)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                            activation=tf.nn.relu)

    add_monitor_ops(dense, conv2, 2)

    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    add_monitor_ops(logits, dense, 3)

    # Calculate Overall Loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    return loss, monitor_ops


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


COLUMNS = ['step', 'monitor', 'value']

def run_test(sess, train, monitor_ops, fp, num_iterations=1000):
    writer = csv.DictWriter(fp, COLUMNS)
    writer.writeheader()

    for i in range(num_iterations):
        if i % 10 == 0:
            print('Starting step', i)

        _, monitors = sess.run((train, monitor_ops))
        for monitor, value in monitors.items():
            writer.writerow({'step': i,
                             'monitor': monitor,
                             'value': value})
        fp.flush()


def generate_and_test(features, labels, sess):
    loss, monitor_ops = generate_graph(
        features, labels,
        [pair_cosine_loss, norm_ratio, mean_based])

    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess.run(tf.global_variables_initializer())

    with open('out.csv', 'w') as fp:
        run_test(sess, train, monitor_ops, fp)


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
