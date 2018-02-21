
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import math
import numpy as np

import cupy
import chainer
import chainer.functions as F
from chainer import cuda
from chainer import reporter
import cupy

import model

# Convenience wrapper to define the HMC energy function.
# This is the only code that is VAE specific, the code that follows is general
# HMC code.
class EnergyFunction:
    def __init__(self, zprior, decoder, X, inv_temp):
        self.zprior = zprior
        self.decode = decoder
        self.X = X
        self.batchsize = X.shape[0]
        self.inv_temp = inv_temp

    def E(self, Z):
        M = Z.shape[0] / self.batchsize
        zs = F.split_axis(Z, M, 0)
        Es = list() # energies

        # Process one (batchsize,nlatent) sample at a time
        for z in zs:
            #logpz = model.gaussian_logp01_inst(z)
            logpz = self.zprior(z)
            pxz = self.decode(z)
            logpxz = pxz(self.X)
            energy = -logpz - self.inv_temp*logpxz
            Es.append(energy)

        Efull = F.flatten(F.vstack(Es))
        return Efull

    # \nabla_z (-log p(z) - inv_temp log p(x|z))
    def grad(self, Z):
        with chainer.force_backprop_mode():
            ZV = chainer.Variable(Z, requires_grad=True)
            energy = F.sum(self.E(ZV))
            energy.backward()

            return ZV.grad

    # Z: (M*batchsize,nlatent), where
    #    Z[0:(batchsize-1),:] is the first sample.
    #
    # Return E, (M*batchsize,), one energy for each sample.
    #
    # -log p(z) - inv_temp log p(x|z)
    def __call__(self, Z):
        with chainer.no_backprop_mode():
            return self.E(Z)

# Perform Hamiltonian Monte Carlo step with leapfrog integrator.
#    n: number of leapfrog steps
#    leapfrog_eps: stepsize.
#
# The samples are updated in place in 'sinit' and we return the average
# acceptance rate over samples.
def leapfrog(efun, sinit, n=10, leapfrog_eps=0.1, moment_sigma=1.0):
    xp = cupy.get_array_module(sinit)
    moment_var = moment_sigma**2.0

    phi = moment_sigma*xp.random.normal(0,1, sinit.shape).astype(np.float32)
    phi_prev = xp.empty_like(phi)
    phi_prev[:] = phi

    s = xp.empty_like(sinit)
    s[:] = sinit

    phi -= 0.5*leapfrog_eps * efun.grad(s)  # initial half-step for momentum
    for m in xrange(2,n):
        s += leapfrog_eps*phi/moment_var
        if m < n:
            phi -= leapfrog_eps * efun.grad(s)
    phi -= 0.5*leapfrog_eps * efun.grad(s)  # final half-step 

    # Compute acceptance probability
    log_alpha = efun(sinit) + 0.5*xp.sum(phi_prev*phi_prev, axis=1)/moment_var
    log_alpha -= efun(s) + 0.5*xp.sum(phi*phi, axis=1)/moment_var
    log_uniform = xp.log(xp.random.uniform(size=log_alpha.shape))
    accept = log_uniform <= log_alpha.data
    sinit[accept,:] = s[accept,:]

    return xp.mean(accept.astype(np.float32))

# Compute inverse temperature ladder;
#   t: temperature index, 1 <= t <= T
#   T: number of stages, T >= 2
#   beta1: initial temperature at t=2, with 0 < beta1 < 1.0
def ais_beta(t, T, beta1=1.0e-4):
    if t == 1:
        return 0.0

    gamma = (1.0/beta1)**(1.0/(T-2))
    return beta1*(gamma**(t-2))

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

# Sigmoid ladder from Wu et al., 2016
def ais_beta_sigmoid(t, T, rad=4.0):
    min_s = sigmoid(-rad)
    max_s = sigmoid(rad)
    s = sigmoid(-rad + ((t-1.0)/(T-1.0))*2*rad)
    return (s - min_s) / (max_s - min_s)

class ZDistribution:
    # Return log w, where w = log r(z) - log p(z).
    # Here r is the sampling distribution of z, and p is the prior
    # distribution of z.
    def initial_logw(self, X, Z):
        raise NotImplementedError

    # Return log p(z)
    def __call__(self, Z):
        raise NotImplementedError

class ZPrior(ZDistribution):
    def __init__(self, nz):
        self.nz = nz

    def initial_logw(self, X, Z):
        xp = cupy.get_array_module(X)
        Mb = Z.shape[0]
        logw = xp.zeros((Mb,))

        return logw

    # X: (batchsize,**) data matrix
    # M: number of z to sample for each datum.
    def sample(self, X, M):
        batchsize = X.shape[0]
        xp = cupy.get_array_module(X)
        Z = xp.random.normal(size=(M*batchsize, self.nz)).astype(np.float32)

        return Z

    def __call__(self, Z):
        return model.gaussian_logp01_inst(Z)

class ZEncoder(ZDistribution):
    def __init__(self, encoder, X):
        self.qmu, self.qln_var = encoder(X)    # pre-compute q(z|x)

    def initial_logw(self, X, Z):
        batchsize = X.shape[0]
        M = Z.shape[0] / batchsize
        zs = F.split_axis(Z, M, 0)
        logw = list() # energies

        # Process one (batchsize,nlatent) sample at a time
        for z in zs:
            # log w = log p(z) - log q(z|x)
            # FIXME: we should sample from q(z|x) as prior but then treat
            # p(z) p(x|z) as target.  See [Wu et al., 2016]
            #logw_i = model.gaussian_logp01_inst(z)
            #logw_i -= model.gaussian_logp_inst(z, self.qmu, self.qln_var)
            logw_i = model.gaussian_logp_inst(z, self.qmu, self.qln_var)
            logw.append(logw_i)

        logw = F.flatten(F.vstack(logw))

        return logw

    def sample(self, X, M):
        Z = list()
        for i in xrange(M):
            Zm = F.gaussian(self.qmu, self.qln_var)
            Z.append(Zm)

        Z = F.vstack(Z)  # (M*batchsize, nz)
        return Z.data

    def __call__(self, Z):
        return model.gaussian_logp01_inst(Z)


# Run annealed importance sampling (AIS) to estimate the marginal
# log-probability log p(x) of the samples X given the decoder model.
#
#    M: number of AIS chains to run per sample.
#    T: number of temperatures in the temperature ladder.
def ais(decoder, X, M=32, T=100, steps=10, stepsize=0.1, sigma=1.0,
        encoder=None):

    xp = cupy.get_array_module(X)
    batchsize = X.shape[0]  # number of samples in X
    nz = decoder.nz     # number of latent dimensions

    # Sample initial z and initialize log weights
    if encoder == None:
        print "Using p(z)"
        zprior = ZPrior(nz)
    else:
        print "Using q(z|x)"
        zprior = ZEncoder(encoder, X)

    Z = zprior.sample(X, M)
    #logw = xp.zeros((M*batchsize,))
    logw = zprior.initial_logw(X, Z)

    for t in xrange(2,T+1):
        efun_cur = EnergyFunction(zprior, decoder, X, ais_beta_sigmoid(t, T))
        efun_prev = EnergyFunction(zprior, decoder, X, ais_beta_sigmoid(t-1, T))
        accept_rate = leapfrog(efun_cur, Z,
            n=steps, leapfrog_eps=stepsize, moment_sigma=sigma)
        if t % 100 == 0:
            print "AIS t=%d  accept rate %.3f" % (t, accept_rate)
        logw += efun_prev(Z).data - efun_cur(Z).data

    logw = F.reshape(logw, (M, batchsize))
    logZ = F.logsumexp(logw, axis=0) - math.log(M)

    return logZ

class AIS(chainer.Chain):
    def __init__(self, decoder, M=32, T=100, steps=10, stepsize=0.1,
        sigma=1.0, encoder=None):
        super(AIS, self).__init__(
            decode = decoder,
        )
        if encoder == None:
            self.encode = None
        else:
            self.add_link('encode', encoder)

        self.M = M
        self.T = T
        self.steps = steps
        self.stepsize = stepsize
        self.sigma = sigma

    def __call__(self, x):
        print "Calling ais()"
        logpx = ais(self.decode, x, M=self.M, T=self.T,
            steps=self.steps, stepsize=self.stepsize, sigma=self.sigma,
            encoder=self.encode)
        batchsize = x.shape[0]

        logpx_mean = F.sum(logpx) / batchsize
        print "E[log p(x)] %.5f" % logpx_mean.data
        obj = -logpx_mean

        # Variance computation
        obj_c = logpx - F.broadcast_to(logpx_mean, logpx.shape)
        obj_var = F.sum(obj_c*obj_c) / (batchsize-1)
        reporter.report({'obj': obj, 'obj_var': obj_var}, self)

        return obj

