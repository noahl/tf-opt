#!python

import tensorflow as tf
import numpy as np

from tensorflow.python.ops import math_ops

BATCH_SIZE = 200


def generate_just_conv1(data, labels):
    """Train the first convolution layer."""

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

    # Now we separate the pairs into different tensors
    num_pairs = BATCH_SIZE // 2
    convolved_1 = tf.slice(conv1, [0, 0, 0, 0], [num_pairs, -1, -1, -1])
    convolved_2 = tf.slice(conv1, [num_pairs, 0, 0, 0], [-1, -1, -1, -1])

    # convolved_1 and convolved_2 are N x 28 x 28 x 32

    # Compute cosine distance between pairs. The built-in
    # cosine_distance only accepts a single axis, not a vector, so
    # it's inlined here.
    conv_1_normal = tf.nn.l2_normalize(convolved_1, [1,2,3])
    conv_2_normal = tf.nn.l2_normalize(convolved_2, [1,2,3])
    conv_1_float = math_ops.to_float(conv_1_normal)
    conv_2_float = math_ops.to_float(conv_2_normal)
    radial_diffs = math_ops.multiply(conv_1_float, conv_2_float)
    diffs = 1 - math_ops.reduce_sum(radial_diffs, axis=[1,2,3], keep_dims=True)

    # diffs has shape [N]

    labels_1 = tf.slice(labels, [0], [num_pairs])
    labels_2 = tf.slice(labels, [num_pairs], [-1])

    # labels_1 and labels_2 have shape [N]

    # loss functions are minimized by convention, so flip the
    # sign. And throw out pairs where the labels match, because we
    # just want different labels to have different outputs.
    labels_eq = math_ops.equal(labels_1, labels_2)
    different_diffs = tf.boolean_mask(diffs, labels_eq)
    loss = math_ops.reduce_sum(different_diffs) * -1

    return loss


def get_features_labels():
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

  dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
  dataset_2 = dataset.shuffle(1000).batch(BATCH_SIZE)
  features, labels = dataset_2.make_one_shot_iterator().get_next()

  return features, labels


def just_conv1_main():
  features, labels = get_features_labels()

  loss = generate_just_conv1(features, labels)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = optimizer.minimize(loss)

  for i in range(100):
    _, loss_value = sess.run((train, loss))
    print('Step:', i, 'loss:', loss_value)


if __name__ == '__main__':
    just_conv1_main()
