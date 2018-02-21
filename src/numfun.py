
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class Log1mExp(function.Function):
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(x_type.dtype.kind == 'f')

    def forward_cpu(self, inputs):
        x, = inputs
        # y = log(1 - exp(x))
        y = numpy.log1p(-numpy.exp(x))
        return utils.force_array(y, x.dtype),

    def forward_gpu(self, inputs):
        x, = inputs
        y = cuda.elementwise(
            'T x', 'T y',
            '''
              y = log1p(-exp(x));
            ''',
            'log1mexp_fwd'
        )(x)
        return y,

    def backward_cpu(self, inputs, grads):
        x, = inputs
        g, = grads
        gx = (-1 / (numpy.exp(-x) - 1)) * g
        return utils.force_array(gx, x.dtype),

    def backward_gpu(self, inputs, grads):
        x, = inputs
        g, = grads
        gx = cuda.elementwise(
            'T x, T g', 'T gx',
            'gx = - 1 / (exp(-x) - 1) * g',
            'log1mexp_bwd'
        )(x, g)
        return gx,


def log1mexp(x):
    return Log1mExp()(x)
