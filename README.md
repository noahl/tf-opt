# Introduction

This is a project to see if we can radically speed up training for
deep learning models by training one layer at a time, starting from
the input, instead of the whole model at once.

The project is not complete, nor is it all production-quality code. If
I finish it and get positive results, I hope to clean up the code and
contribute it back to Tensorflow.

# How It Works

Start with a deep learning model that classifies its inputs into
categories. I'm working with the MNIST CNN described at
https://www.tensorflow.org/tutorials/layers to have something
concrete, but none of the ideas are specific to that model. For
specific values of its parameters, your model is a function `M` from
inputs to labels.

Now pick your favorite intermediate layer and divide your model into
two parts: everything up to and including that layer, and everything
after. Again, for specifc parameter values, both of those parts are
functions `B` and `E` (beginning and end), and `M(x) = E(B(x))`. The
range of `B` is the set of values of the intermediate nodes that
separate the `B` and `E` portions of the model.

The goal is to optimize the parameters of `B` without having to
evaluate `E`. I have three distinct but related ideas for how to do
this, which I hope to compare as part of this project.

## Maximum Difference

Maximize the difference in `B(x)` for x with different
labels. Intuitively, the more that different labels have different
node values, the easier it should be to distinguish them.

That leaves the question of how to measure difference. Being a bit
hand-wavy, a linear layer assigns to each output node the dot product
of some special vector and the vector of all of the inputs. Cosine
similarity is a natural way to measure difference when you're planning
to distinguish things with dot products, so I plan to start there.

## Kullback-Leibler

Each label has a corresponding distribution of inputs. When we apply
`B` to those inputs, we get a distribution of intermediate values
corresponding to each label. This idea is to use Kullback-Leibler
divergence to measure the difference between these distributions, and
then maximize that difference.

The hard part about this is that in order to measure K-L divergence,
we need to be able to compute the likelihoods of these distributions
at different points. We aren't given distributions; just samples. In
order for this to work, we would have to build some sort of generative
model for each label so we would have per-label distributions, and
then measure the difference between those.

Alternatively, could try to make a technique like
https://www.researchgate.net/publication/224324902_Kullback-Leibler_Divergence_Estimation_of_Continuous_Distributions
work.

Given the number of open questions for this idea, and the fact that I
have two other ideas with clear implementation strategies to try, I'm
going to put this on the shelf for now.

## Sub-Models

Construct a simple classification model `G(B(x))` starting with `B`
and using as few additional layers as possible. (Maybe just take all
of `B`'s outputs as input to one final dense layer.)  Optimize this
model via the usual techniques, then just take the weights learned for
the intermediate layer that we cared about.

# The Goal

The goal of the proof-of-concept stage is to produce a single graph of
full-graph loss vs. training time. The graph should have different
sets of points (or lines) corresponding to different training
strategies. All training must be on the same hardware - my laptop.

# Design

I'm starting from Tensorflow's MNIST CNN model, so I'll start from
their implementation too unless I see a strong benefit to rewriting.

The maximum difference and sub-model ideas will both be implemented by
writing functions (`maximum_difference` and `sub_model`) which take as
input an intermediate tensor in a graph and generate a loss function
for the graph just up to that intermediate stage.

By calling this function at different points in the MNIST CNN model
generator, I can produce a list of loss functions - `[loss_1, loss_2,
...]` that include different portions of the graph. Then it's a matter
of generating training ops for each and running them one at a
time. The standard strategy would generate a list with just one loss
function.

# To-Do List

## Baseline model
 - [x] Implement baseline model with standard gradient descent optimization.
 - [ ] Rewrite baseline model to output (wall time, loss) pairs.
 - [ ] Benchmark baseline model: tried, took too long for my
   computer. Will try again.
 - [ ] Experiment with speeding up the baseline model using variations
   on gradient descent. It's important to compare against the best the
   standard techniques can do.

## Maximum difference
 - [x] Implement the basic idea: works, but optimization is unstable. Needs
   more investigation.
 - [ ] Make the implementation output (wall time, loss) pairs.
 - [ ] Run the benchmark.

## Sub-Models
 - [ ] Implement the sub-model idea.
 - [ ] Run the benchmark.

## Other
 - [ ] If the ideas work, investigate how much time to spend training
   each layer for fastest overall results.
 - [ ] Investigate whether a final whole-model optimization stage
   is worth the time or not.
 - [ ] Once I have data, generate graphs and write a better report
   than this readme.
