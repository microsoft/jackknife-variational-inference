
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import math
import cupy
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import reporter
import numfun
import jvi

def gaussian_kl_divergence_inst(mu, ln_var):
    """D_{KL}(N(mu,var) | N(0,1))"""
    axis_sum = tuple(di for di in range(1,mu.data.ndim))
    dim = np.prod([mu.data.shape[i] for i in axis_sum])
    S = F.exp(ln_var)

    KL_sum = 0.5*(F.sum(S,axis=axis_sum) + F.sum(mu*mu,axis=axis_sum) -
        F.sum(ln_var,axis=axis_sum) - dim)

    return KL_sum

def gaussian_kl_divergence(mu, ln_var):
    """D_{KL}(N(mu,var) | N(0,1))"""
    batchsize = mu.data.shape[0]
    S = F.exp(ln_var)
    D = mu.data.size

    KL_sum = 0.5*(F.sum(S) + F.sum(mu*mu) - F.sum(ln_var) - D)

    return KL_sum / batchsize

def gaussian_logp01_inst(x):
    """log N(x ; 0, 1)"""
    batchsize = x.shape[0]
    axis_sum = tuple(di for di in range(1,x.ndim))
    dim = np.prod([x.shape[i] for i in axis_sum])

    logp_sum = -0.5*(F.sum(x*x,axis=axis_sum) + dim*math.log(2.0*math.pi))

    return logp_sum

def gaussian_logp01(x):
    """log N(x ; 0, 1)"""
    batchsize = x.shape[0]
    D = x.size

    logp_sum = -0.5*(F.sum(x*x) + D*math.log(2.0*math.pi))

    return logp_sum / batchsize

def gaussian_logp_inst(x, mu, ln_var):
    """log N(x ; mu, var)"""
    batchsize = mu.data.shape[0]
    axis_sum = tuple(di for di in range(1,mu.data.ndim))
    dim = np.prod([mu.data.shape[i] for i in axis_sum])
    S = F.exp(ln_var)
    xc = x - mu

    logp_sum = -0.5*(F.sum((xc*xc) / S, axis=axis_sum) + F.sum(ln_var, axis=axis_sum)
        + dim*math.log(2.0*math.pi))

    return logp_sum

def gaussian_logp(x, mu, ln_var):
    """log N(x ; mu, var)"""
    batchsize = mu.data.shape[0]
    #D = x.data.size
    D = x.size
    S = F.exp(ln_var)
    xc = x - mu

    logp_sum = -0.5*(F.sum((xc*xc) / S) + F.sum(ln_var)
        + D*math.log(2.0*math.pi))

    return logp_sum / batchsize

class GaussianLikelihood:
    def __init__(self, pmu, pln_var):
        self.pmu = pmu
        self.pln_var = pln_var

    def __call__(self, x):
        return gaussian_logp_inst(x, self.pmu, self.pln_var)

class IWAEObjective(chainer.Chain):
    def __init__(self, encoder, decoder, num_zsamples=1):
        super(IWAEObjective, self).__init__(
            encode = encoder,
            decode = decoder,
        )
        self.num_zsamples = num_zsamples

    def compute_elbo(self, logw):
        k = logw.shape[0]
        batchsize = logw.shape[1]
        # ELBO = (1/k) sum_i log w_i
        elbo = F.sum(logw, axis=0) / k
        elbo = F.sum(elbo) / batchsize

        return elbo

    # Return (num_zsamples, batchsize)
    def compute_logw(self, x):
        # Compute q(z|x)
        qmu, qln_var = self.encode(x)

        logw = list()
        for j in xrange(self.num_zsamples):
            # z ~ q(z|x)
            z = F.gaussian(qmu, qln_var)

            # Compute p(x|z)
            pxz = self.decode(z)

            logpxz = pxz(x)
            logpz = gaussian_logp01_inst(z)
            logqz = gaussian_logp_inst(z, qmu, qln_var)

            logwi = logpxz + logpz - logqz
            logw.append(logwi)

        logw = F.stack(logw)    # (num_zsamples,batchsize)
        return logw

    def __call__(self, x):
        batchsize = x.shape[0]

        logw = self.compute_logw(x)

        # IWAE = log (1/k) sum_i w_i
        logp = F.logsumexp(logw, axis=0) - math.log(self.num_zsamples)
        logp_mean = F.sum(logp) / batchsize
        obj = -logp_mean

        # Variance computation
        obj_c = logp - F.broadcast_to(logp_mean, logp.shape)
        obj_var = F.sum(obj_c*obj_c) / (batchsize-1)

        obj_elbo = -self.compute_elbo(logw)

        reporter.report({'obj': obj, 'obj_var': obj_var, 'obj_elbo': obj_elbo}, self)

        return obj

# Jackknife variational inference
class JVIObjective(chainer.Chain):
    def __init__(self, encoder, decoder, num_zsamples=2, jvi_order=0, device=0):
        super(JVIObjective, self).__init__(
            encode = encoder,
            decode = decoder,
        )
        self.num_zsamples = num_zsamples
        self.jvi_order = jvi_order

        # Pre-generate JVI matrix
        self.A, B = jvi.jvi_matrix(num_zsamples, jvi_order)
        self.A = np.reshape(self.A, (self.A.shape[0],1))
        self.A = self.A.astype(np.float32)
        self.logB = np.log(B).T    # (num_zsamples,M)
        self.logB = self.logB.astype(np.float32)    # (num_zsamples,M)
        M = self.logB.shape[1]
        print "Using %d JVI subsets (%d z-samples, jvi order %d)" % (M, num_zsamples, jvi_order)

        # Copy to GPU
        self.A = cuda.to_gpu(self.A, device=device)
        self.logB = cuda.to_gpu(self.logB, device=device)

    def __call__(self, x):
        batchsize = x.shape[0]

        iwae = IWAEObjective(self.encode, self.decode, self.num_zsamples)
        logw = iwae.compute_logw(x) # (num_zsamples,batchsize)
        obj_elbo = -iwae.compute_elbo(logw)

        M = self.logB.shape[1]  # number of subsets
        n = self.num_zsamples

        # (n,M,batchsize)
        logw = F.broadcast_to(F.reshape(logw, (n,1,batchsize)), (n,M,batchsize))
        logB = F.broadcast_to(F.reshape(self.logB, (n,M,1)), (n,M,batchsize))
        R = F.logsumexp(logw + logB, axis=0)    # (M,batchsize)
        logp = F.matmul(self.A, R, transa=True) # (batchsize,)

        obj_c = logp - F.broadcast_to(F.mean(logp), logp.shape)
        obj_var = F.sum(obj_c*obj_c) / (batchsize-1)
        obj = -F.mean(logp)

        reporter.report({'obj': obj, 'obj_var': obj_var, 'obj_elbo': obj_elbo}, self)
        return obj

class ImprovedIWAEObjective(chainer.Chain):
    def __init__(self, encoder, decoder, num_zsamples=1):
        super(ImprovedIWAEObjective, self).__init__(
            encode = encoder,
            decode = decoder,
        )
        self.num_zsamples = num_zsamples

    def __call__(self, x):
        batchsize = x.shape[0]

        iwae = IWAEObjective(self.encode, self.decode, self.num_zsamples)
        logw = iwae.compute_logw(x) # (num_zsamples,batchsize)
        obj_elbo = -iwae.compute_elbo(logw)

        # Jackknife bias corrected logp estimate
        A = F.logsumexp(logw, axis=0)
        logp_iwae = A - math.log(self.num_zsamples)
        logp_iwae = F.sum(logp_iwae) / batchsize

        k = float(self.num_zsamples)
        wnorm = F.exp(logw - F.broadcast_to(A, logw.shape))
        #wmax = F.max(wnorm)
        #print wmax
        #ess = F.sum(1.0 / F.sum(wnorm*wnorm, axis=0)) / batchsize
        #B = F.sum(F.log1p(-F.exp(logw - F.broadcast_to(A, logw.shape))), axis=0)
        #print logw
        B = F.sum(numfun.log1mexp(logw - F.broadcast_to(A, logw.shape) - 1.0e-6), axis=0)
        #print B
        logp_jk = A - ((k-1)/k)*B - k*math.log(k) + (k-1)*math.log(k-1)
        logp_jk_mean = F.sum(logp_jk) / batchsize
        obj = -logp_jk_mean
        correction = logp_jk_mean - logp_iwae

        # Variance computation
        obj_c = logp_jk - F.broadcast_to(logp_jk_mean, logp_jk.shape)
        obj_var = F.sum(obj_c*obj_c) / (batchsize-1)

        reporter.report({'obj': obj, 'obj_var': obj_var, 'obj_elbo': obj_elbo,
            'corr': correction}, self)

        return obj


class ELBOObjective(chainer.Chain):
    def __init__(self, encoder, decoder, num_zsamples=1):
        super(ELBOObjective, self).__init__(
            encode = encoder,
            decode = decoder,
        )
        self.num_zsamples = num_zsamples

    # ELBO objective: E_{z ~ q(z|x)}[log p(x|z)] - D(q(z|x) | p(z))
    def __call__(self, x):
        batchsize = x.shape[0]

        # Compute q(z|x)
        qmu, qln_var = self.encode(x)

        kl_inst = gaussian_kl_divergence_inst(qmu, qln_var)
        logp_inst = None
        self.kl = F.sum(kl_inst)/batchsize
        self.logp = 0
        for j in xrange(self.num_zsamples):
            # z ~ q(z|x)
            z = F.gaussian(qmu, qln_var)

            # Compute p(x|z)
            pxz = self.decode(z)
            logpxz = pxz(x)
            if logp_inst is None:
                logp_inst = logpxz
            else:
                logp_inst += logpxz

            # Compute objective
            batchsize = logpxz.shape[0]
            self.logp += F.sum(logpxz) / batchsize

        # Compute standard deviation
        logp_inst /= self.num_zsamples
        obj_inst = kl_inst - logp_inst
        obj_inst_mean = F.sum(obj_inst) / batchsize
        obj_c = obj_inst - F.broadcast_to(obj_inst_mean, obj_inst.shape)
        obj_var = F.sum(obj_c*obj_c)/(batchsize-1)

        self.logp /= self.num_zsamples
        self.obj = self.kl - self.logp

        reporter.report({'obj': self.obj, 'obj_var': obj_var, 'kl': self.kl, 'logp': self.logp}, self)

        return self.obj

class ISObjective(chainer.Chain):
    def __init__(self, encoder, decoder, num_zsamples=1):
        super(ISObjective, self).__init__(
            encode = encoder,
            decode = decoder,
        )
        self.num_zsamples = num_zsamples

    # Importance sampling estimator
    def __call__(self, x):
        # Compute q(z|x)
        qmu, qln_var = self.encode(x)
        batchsize = qmu.data.shape[0]

        # Perform unnormalized importance sampling
        logw = list()
        logpxz = list()
        for i in xrange(self.num_zsamples):
            # z ~ q(z|x)
            z = F.gaussian(qmu, qln_var)
            logqz = gaussian_logp_inst(z, qmu, qln_var)
            logpz = gaussian_logp01_inst(z)

            # Compute p(x|z)
            pxz = self.decode(z)
            logpxz_i = pxz(x)
            logpxz.append(logpxz_i)

            logw_i = logpz + logpxz_i - logqz
            logw.append(logw_i)

        # Self-normalize importance weights
        logw = F.stack(logw)    # (num_zsamples,batchsize)
        lse = F.logsumexp(logw, axis=0)
        logw -= F.broadcast_to(lse, logw.shape)
        w = F.exp(logw)

        # Compute effective sample size
        ess = F.sum(1.0 / F.sum(w*w, axis=0)) / batchsize

        logpxz = F.stack(logpxz)    # (num_zsamples,batchsize)

        # XXX: break dependency in computational graph
        w = chainer.Variable(w.data)
        obj = -F.sum(w*logpxz) / batchsize

        reporter.report({'obj': obj, 'ess': ess}, self)

        return obj

