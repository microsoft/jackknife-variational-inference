
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import subprocess
from subprocess import PIPE

from chainer import cuda
from chainer.cuda import cupy

def print_compute_graph(file, g):
    format = file.split('.')[-1]
    cmd = 'dot -T%s -o %s'%(format,file)
    p=subprocess.Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    p.stdin.write(g.dump())
    p.communicate()
    return p.returncode

