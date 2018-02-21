#!/usr/bin/env python

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Evaluate log-likelihood of VAE models.

Usage:
  evaluate.py (-h | --help)
  evaluate.py [options] <modelprefix>

Options:
  -h --help                    Show this help screen.
  -d <dataset>, --dataset      Dataset to use, must be "mnist" in public release [default: mnist].
  -g <device>, --device        GPU id to train model on.  Use -1 for CPU [default: -1].
  -s <samples>, --samples      Number of evaluation samples [default: 64].
  -b <batchsize>, --batchsize  Evaluation minibatch size [default: 4096].
  -r <reps>, --reps            Replications [default: 1].
  --bootstrap                  Perform resampling with replacement from test set.
  --resultfile <resultfile>    Write logp estimate to file, or "+" to auto-filename.
  --logw <logwdata.mat>        Write logw array for first 1024 samples.
  -E <evaltype>, --eval        Evaluation type, "vae", "iwae", "iwae++", or "ais" [default: iwae]
  --jvi-order <order>          Order of jackknife, zero is IWAE [default: 1].
  --ais-temps <T>              Number of temperatures [default: 3000].
  --ais-prior                  Use p(z) instead of q(z|x) for initial z sample.
  --ais-sigma <sigma>          Moment parameter standard deviation [default: 1.0].
  --ais-steps <steps>          Number of leapfrog steps [default: 10].
  --ais-stepsize <stepsize>    Leapfrog step size [default: 0.03].

The modelprefix.meta.yaml file must exist.
"""

import sys
import time
import timeit
import yaml
import math
import numpy as np
import scipy.io as sio
from docopt import docopt

import chainer
from chainer.training import extensions
from chainer import reporter
from chainer import serializers
from chainer import optimizers
from chainer import cuda
from chainer import computational_graph
import chainer.functions as F
import cupy

import data
import model
import util
import ais

args = docopt(__doc__, version='evaluate 0.4')
print(args)

mpref = args['<modelprefix>']
model_file = mpref+'/snapshot_bestvalobj'
yaml_file = mpref+'.meta.yaml'

# Read meta data
with open(yaml_file, 'r') as f:
    argsy = yaml.load(f)

nhidden = int(argsy['--nhidden'])
nlatent = int(argsy['--nlatent'])
zcount_train = int(argsy['--vae-samples'])
print "Model was trained using %d vae samples" % zcount_train

batchsize = int(args['--batchsize'])
print "Using a batchsize of %d instances for evaluation" % batchsize

reps = int(args['--reps'])
print "Performing %d replications" % reps

# Load data and instantiate matching encoder/decoder models
dataset = args['--dataset']
train, val, test, encoder, decoder = data.prepare_dataset_and_model(dataset, nhidden, nlatent)

vae_type_train = argsy['--vae-type']
zcount = int(args['--samples'])
print "Using %d samples for likelihood evaluation" % zcount

# Check GPU
gpu_id = int(args['--device'])
print "Running on GPU %d" % gpu_id
if gpu_id >= 0:
    cuda.check_cuda_available()
    xp = cuda.cupy
else:
    xp = np

# Evaluation type
vae_type = args['--eval']
jvi_order = int(args['--jvi-order'])
if vae_type == "vae":
    vae = model.ELBOObjective(encoder, decoder, zcount)
elif vae_type == "iwae":
    vae = model.IWAEObjective(encoder, decoder, zcount)
elif vae_type == "iwae++":
    vae = model.ImprovedIWAEObjective(encoder, decoder, zcount)
elif vae_type == "jvi":
    vae = model.JVIObjective(encoder, decoder, zcount, jvi_order, device=gpu_id)
elif vae_type == "is":
    vae = model.ISObjective(encoder, decoder, zcount)
elif vae_type == "ais":
    ais_temps = int(args['--ais-temps'])
    ais_sigma = float(args['--ais-sigma'])
    ais_steps = int(args['--ais-steps'])
    ais_stepsize = float(args['--ais-stepsize'])
    print "Annealed importance sampling, %d samples, %d temperatures" % (zcount, ais_temps)
    print "Leapfrog integrator, %d steps, %f stepsize" % (ais_steps, ais_stepsize)
    if args['--ais-prior']:
        vae = ais.AIS(decoder, M=zcount, T=ais_temps,
            steps=ais_steps, stepsize=ais_stepsize, sigma=ais_sigma,
            encoder=None)
    else:
        vae = ais.AIS(decoder, M=zcount, T=ais_temps,
            steps=ais_steps, stepsize=ais_stepsize, sigma=ais_sigma,
            encoder=encoder)
else:
    sys.exit("Unsupported VAE type")

#serializers.load_hdf5(model_file, vae)
#serializers.load_npz(model_file, vae, path='updater/model:main/')
try:
    with np.load(model_file) as f:
        d = serializers.NpzDeserializer(f,path='updater/model:main/')
        d.load(vae)
except:
    with np.load(model_file) as f:
        d = serializers.NpzDeserializer(f,path='updater/model:elbo/')
        d.load(vae)

print "Deserialized model '%s' of type '%s'" % (model_file, vae_type_train)

if gpu_id >= 0:
    vae.to_gpu(gpu_id)
print "Moved model to GPU %d" % gpu_id

# For debugging purposes, optionally, obtain and write logw value for the
# first few test samples
if '--logw' in args and args['--logw'] is not None:
    logw_file = args['--logw']
    with cupy.cuda.Device(gpu_id):
        with chainer.no_backprop_mode():
            xt = test[0:256,:]
            print xt.shape
            xt = chainer.Variable(xp.asarray(xt, dtype=np.float32))
            logw = vae.compute_logw(xt)
            logw.to_cpu()
            sio.savemat(logw_file, {"logw": logw.data})

    print "Wrote logw for first 256 samples to file '%s'." % logw_file

print "Evaluating..."

for ri in xrange(reps):
    if args['--bootstrap']:
        print "Bootstrap resampling..."
        test_idx = np.random.choice(len(test), size=(len(test),), replace=True)
        test_cur = test[test_idx,:]
        print test_cur.shape
    else:
        test_cur = test

    test_iter = chainer.iterators.SerialIterator(test_cur, batchsize, repeat=False, shuffle=False)
    obs = {}
    reprt = reporter.Reporter()
    reprt.add_observer('main', vae)
    with cupy.cuda.Device(gpu_id):
        start_time = timeit.default_timer()
        with reprt.scope(obs):
            teval = extensions.Evaluator(test_iter, vae, device=gpu_id)
            res = teval.evaluate()

        runtime = timeit.default_timer() - start_time
        print "Evaluation took %.2fs" % runtime
        print res
        obj_mean = -res['main/obj']
        obj_sem = res['main/obj_var']
        obj_sem = math.sqrt(obj_sem/len(test))
        print "%.8f +/- %.8f # logp(%s) %d" % (obj_mean, obj_sem, vae_type, zcount)

    if '--resultfile' in args and args['--resultfile'] is not None:
        resultfile = args['--resultfile']
        if resultfile == "+":
            resultfile = mpref+".test.perf"

        print "Writing test set results to '%s'." % resultfile
        with open(resultfile, "a") as rf:
            rf.write("%.8f,%.8f,%.8f\n" % (obj_mean, obj_sem, runtime))

sys.exit()

