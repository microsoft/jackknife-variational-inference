#!/usr/bin/env python

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Train a variational autoencoder (VAE/IWAE/JVI) model.

Usage:
  train.py (-h | --help)
  train.py [options]

Options:
  -h --help                    Show this help screen.
  -d <dataset>, --dataset      Dataset to use, one of "mnist", "msrc12", "celeb1m", "celeb1m-small" [default: msrc12].
  -o <modelprefix>             Write trained model to given file.h5 [default: output].
  -g <device>, --device        GPU id to train model on.  Use -1 for CPU [default: -1].
  -e <epochs>, --epochs        Number of epochs to train [default: 1000].
  -b <batchsize>, --batchsize  Minibatch size [default: 8192].
  --fancy                      Use fancy model variant.
  --lr <lr>                    Initial learning rate [default: 1.0e-4].
  --opt <opt>                  Optimizer to use, one of "sgd", "smorms3" or "adam" [default: smorms3].
  --vae-type <vaetype>         VAE type, one of "vae", "iwae", "iwae++", "jvi", "jvi+elbo" [default: vae].
  --vae-samples <zcount>       Number of samples in VAE z [default: 1].
  --jvi-order <order>          Order of jackknife, zero is IWAE [default: 1].
  --nhidden <nhidden>          Number of hidden dimensions [default: 128].
  --nlatent <nz>               Number of latent VAE dimensions [default: 16].
  --vis <graph.ext>            Visualize computation graph.

For "msrc12" the data.mat file must contain a (N,d) array of N instances, d
dimensions each.
"""

import sys
import time
import yaml
import numpy as np
from docopt import docopt

import chainer
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from chainer import reporter
from chainer import serializers
from chainer import optimizers
from chainer import cuda
from chainer import computational_graph
import chainer.functions as F
import cupy

import model
import updater
import util
import data
from elbo_updater import ELBOUpdater

class TestModeEvaluator(extensions.Evaluator):
    def evaluate(self):
        model = self.get_target('main')
        #use_elbo = model.use_elbo
        #model.use_elbo = True
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        #model.use_elbo = use_elbo
        return ret

def save_model(args, vae):
    # Save model
    if args['-o'] is not None:
        modelmeta = args['-o'] + '.meta.yaml'
        print "Writing model metadata to '%s' ..." % modelmeta
        with open(modelmeta, 'w') as outfile:
            outfile.write(yaml.dump(dict(args), default_flow_style=False))

        modelfile = args['-o'] + '.h5'
        print "Writing model to '%s' ..." % modelfile
        serializers.save_hdf5(modelfile, vae)


args = docopt(__doc__, version='train 0.2')
print(args)

print "Using chainer version %s" % chainer.__version__
nhidden = int(args['--nhidden'])
print "%d hidden dimensions" % nhidden
nlatent = int(args['--nlatent'])
print "%d latent VAE dimensions" % nlatent

fancy = False
if args['--fancy']:
    fancy = True
    print "Using fancy model version."

# Load data and instantiate matching encoder/decoder models
dataset = args['--dataset']
train, val, test, encoder, decoder = data.prepare_dataset_and_model(dataset, nhidden, nlatent, fancy)

print "Training set size: %d instances" % len(train)
print "Validation set size: %d instances" % len(val)

epochs = int(args['--epochs'])
print "Training for %d epochs" % epochs

# Setup model
zcount = int(args['--vae-samples'])
print "Using %d VAE samples per instance" % zcount

gpu_id = int(args['--device'])

vae_type = args['--vae-type']
jvi_order = int(args['--jvi-order'])
elbo = None
print "Training using '%s' objective" % vae_type
if vae_type == "vae":
    vae = model.ELBOObjective(encoder, decoder, zcount)
elif vae_type == "iwae":
    vae = model.IWAEObjective(encoder, decoder, zcount)
elif vae_type == "iwae++":
    vae = model.ImprovedIWAEObjective(encoder, decoder, zcount)
elif vae_type == "jvi":
    vae = model.JVIObjective(encoder, decoder, zcount, jvi_order, device=gpu_id)
elif vae_type == "jvi+elbo":
    vae = model.JVIObjective(encoder, decoder, zcount, jvi_order, device=gpu_id)
    elbo = model.ELBOObjective(encoder, decoder, zcount)
elif vae_type == "is":
    vae = model.ISObjective(encoder, decoder, zcount)
else:
    sys.exit("Unsupported VAE type (%s)." % vae_type)

lr = float(args['--lr'])
print "Using initial learning rate %f" % lr
opt_type = args['--opt']
if opt_type == "adam":
    opt = optimizers.Adam(alpha=lr)
    opt_elbo = optimizers.Adam(alpha=lr)
elif opt_type == "smorms3":
    opt = optimizers.SMORMS3(lr=lr)
    opt_elbo = optimizers.SMORMS3(lr=lr)
elif opt_type == "sgd":
    opt = optimizers.SGD(lr=lr)
    opt_elbo = optimizers.SGD(lr=lr)
else:
    sys.exit("Unsupported optimizer type (%s)." % opt_type)

opt.setup(vae)
opt.add_hook(chainer.optimizer.GradientClipping(4.0))

if elbo:
    opt_elbo.setup(elbo)
    opt_elbo.add_hook(chainer.optimizer.GradientClipping(4.0))

# Move to GPU
if gpu_id >= 0:
    cuda.check_cuda_available()
if gpu_id >= 0:
    xp = cuda.cupy
    vae.to_gpu(gpu_id)
    if elbo:
        elbo.to_gpu(gpu_id)
else:
    xp = np

# Setup training parameters
batchsize = int(args['--batchsize'])
print "Using a batchsize of %d instances" % batchsize

# Save model meta data
if args['-o'] is not None:
    modelmeta = args['-o'] + '.meta.yaml'
    print "Writing model metadata to '%s' ..." % modelmeta
    with open(modelmeta, 'w') as outfile:
        outfile.write(yaml.dump(dict(args), default_flow_style=False))

train_iter = chainer.iterators.SerialIterator(train, batchsize)
val_iter = None
if val is not None:
    val_iter = chainer.iterators.SerialIterator(val, batchsize,
        repeat=False, shuffle=False)

if vae_type == "jvi+elbo":
    vae = model.JVIObjective(encoder, decoder, zcount, jvi_order, device=gpu_id)
    elbo = model.ELBOObjective(encoder, decoder, zcount)
    updater = ELBOUpdater(
        models=(elbo, vae),
        iterator=train_iter,
        optimizer={ 'elbo': opt_elbo, 'p_obj': opt },
        device=gpu_id)
else:
    updater = training.StandardUpdater(train_iter, opt, device=gpu_id)

trainer = training.Trainer(updater, (epochs, 'epoch'), out=args['-o'])
if val is not None:
    trainer.extend(TestModeEvaluator(val_iter, vae, device=gpu_id))
#trainer.extend(extensions.ExponentialShift('lr', 0.5), trigger=(50, 'epoch'))
#trainer.extend(extensions.dump_graph('main/obj'))
#trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))
trainer.extend(extensions.LogReport())
valtrigger=chainer.training.triggers.MinValueTrigger('validation/main/obj',
    trigger=(1, 'epoch'))
trainer.extend(extensions.snapshot(filename='snapshot_bestvalobj'),
    trigger=valtrigger)

if val is not None:
    if elbo:
        trainer.extend(extensions.PrintReport(
            ['epoch', 'iteration', 'elbo/elbo', 'p_obj/obj',
                'validation/main/obj', 'validation/main/obj_elbo', 'elapsed_time']))
        trainer.reporter.add_observer('elbo', elbo)
        trainer.reporter.add_observer('p_obj', vae)
    else:
        trainer.extend(extensions.PrintReport(
            ['epoch', 'iteration', 'main/obj', 'main/obj_elbo',
                'validation/main/obj', 'validation/main/obj_elbo', 'elapsed_time']))
else:
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/obj', 'elapsed_time']))
trainer.extend(extensions.ProgressBar(update_interval=1))

print "Training..."
trainer.run()
print "Minimum validation loss: %.4f" % valtrigger._best_value
perffilename = args['-o'] + '.perf'
print "Writing model validation performance to '%s' ..." % perffilename
perffile = open(perffilename, 'w')
perffile.write("%.6f" % valtrigger._best_value)

