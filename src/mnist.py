
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import math
import chainer
import chainer.functions as F
from chainer.functions.loss.vae import bernoulli_nll
import chainer.links as L
from chainer.datasets import mnist
from chainer.datasets import TransformDataset

import model

# Dynamic binarization.  x is an array with values 0 <= v <= 1.
def mnist_transform(x):
    x = np.copy(x)
    U = np.random.uniform(size=x.shape)
    # e.g. U = 0.6, x = 0.92
    x[U <= x] = 1.0
    x[U > x] = 0.0
    return x

# Binarized MNIST with dynamic binarization, see
# [Salakhutdinov and Murray, ICML 2008]
def get_mnist_vae():
    train, test = mnist.get_mnist(withlabel=False)
    val = train[50000:60000]
    train = train[0:50000]
    train = TransformDataset(train, mnist_transform)
    val = TransformDataset(val, mnist_transform)
    test = TransformDataset(test, mnist_transform)

    return train, val, test

def bernoulli_logp_inst(x, h):
    """log B(x; p=sigmoid(h))"""
    L = bernoulli_nll(x, h, reduce='no')
    return -F.sum(L,axis=1)

# MNIST encoder
class MNISTEncoder(chainer.Chain):
    def __init__(self, dim_in, dim_hidden, dim_latent):
        super(MNISTEncoder, self).__init__(
            # encoder
            qlin0 = L.Linear(dim_in, dim_hidden),
            qlin1 = L.Linear(2*dim_hidden, dim_hidden),
            qlin_mu = L.Linear(2*dim_hidden, dim_latent),
            qlin_ln_var = L.Linear(2*dim_hidden, dim_latent),
        )

    def __call__(self, x):
        h = F.crelu(self.qlin0(x))
        h = F.crelu(self.qlin1(h))
        qmu = self.qlin_mu(h)
        qln_var = self.qlin_ln_var(h)

        return qmu, qln_var

class MNISTLikelihood:
    def __init__(self, ph):
        self.ph = ph

    def __call__(self, x):
        return bernoulli_logp_inst(x, self.ph)

class MNISTDecoder(chainer.Chain):
    def __init__(self, dim_in, dim_hidden, dim_latent):
        super(MNISTDecoder, self).__init__(
            # decoder
            plin0 = L.Linear(dim_latent, dim_hidden),
            plin1 = L.Linear(2*dim_hidden, dim_hidden),
            plin2 = L.Linear(2*dim_hidden, dim_in),
        )
        self.nz = dim_latent

    def __call__(self, z):
        h = F.crelu(self.plin0(z))
        h = F.crelu(self.plin1(h))
        ph = self.plin2(h)

        return MNISTLikelihood(ph)

