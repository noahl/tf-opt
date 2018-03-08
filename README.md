# Introduction

This is a project to see if we can radically speed up training for
deep learning models by training one layer at a time, starting from
the input, instead of the whole model at once.

The project is not complete, nor is it all production-quality code. If
I finish it and get positive results, I hope to clean up the code and
contribute it back to Tensorflow.

# How It Works

Imagine you have a deep learning model. I'm working with the MNIST CNN
described at https://www.tensorflow.org/tutorials/layers to have
something concrete, but none of the ideas are specific to that
model. Your model classifies its inputs into categories `label_1`
through `label_n`. For specific values of its parameters, your model
is a function `M` from inputs to labels.

Now pick your favorite intermediate layer and divide your model into
two parts: everything up to and including that layer, and everything
after. Again, for specifc parameter values, both of those parts are
functions `B` and `E` (beginning and end), and `M(x) = E(B(x))`. The
range of `B` is the set of values of the intermediate nodes that
separate the `B` and `E` portions of the model.

The goal is to optimize the parameters of `B` without having to
evaluate `E`. I have three distinct but related ideas for how to do
this, which I hope to compare as part of this project.

1. Maximize the difference in `B(x)` for x with different
   labels. Intuitively, the more that different labels have different
   node values, the easier it should be to distinguish them. Since
   neural nets work by dot products, cosine difference is a natural
   loss function here.
1. Use Kullback-Leibler divergence to measure how different the
   distributions of `B(x)` values are for different labels, and try to
   maximize that difference. This is a bit trickier than the previous
   idea because computing the K-L divergence requires having a
   distribution, not just samples, so we would have to construct a
   distribution from the sample data. Alternatively, could try to make
   a technique like
   https://www.researchgate.net/publication/224324902_Kullback-Leibler_Divergence_Estimation_of_Continuous_Distributions
   work.
1. Construct a simple classification model `G(B(x))` starting with `B`
   and using as few additional layers as possible. (Maybe just take
   all of `B`'s outputs as input to one final dense layer.)  Optimize
   this model via the usual techniques, then just take the weights
   learned for the intermediate layer that we cared about.

# To-Do List

 - [x] Implement baseline model with standard gradient descent optimization.
 - [ ] Benchmark baseline model: tried, took too long for my
   computer. Will try again.
 - [x] Implement idea 1: works, but optimization is unstable. Needs
   more investigation.
 - [ ] Implement idea 2
 - [ ] Implement idea 3
 - [ ] If the ideas work, investigate how much time to spend training
   each layer for fastest overall results.
 - [ ] Investigate whether a final whole-model optimization stage
   is worth the time or not.
