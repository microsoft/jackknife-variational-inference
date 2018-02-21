
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import sys

import mnist

# The only function to access data sets and models for particular datasets.
def prepare_dataset_and_model(dataset, nhidden, nlatent, fancy=False):
    if dataset == "mnist":
        print "Using dynamically binarized MNIST handwritten digit dataset"
        train, val, test = mnist.get_mnist_vae()
        din = len(train[0])

        encoder = mnist.MNISTEncoder(din, nhidden, nlatent)
        decoder = mnist.MNISTDecoder(din, nhidden, nlatent)
    else:
        sys.exit("Unknown dataset name ('%s') supplied to option '-d'." % dataset)

    return train, val, test, encoder, decoder
