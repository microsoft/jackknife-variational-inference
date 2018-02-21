
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Count number of JVI terms for a given sample size and JVI order.

Usage:
  jvicount.py <n> <order>

The sample size is n and JVI order zero corresponds to the IWAE bound.
The tool returns the total number of terms in the JVI objective.
"""

from docopt import docopt
import jvi

args = docopt(__doc__, version='jvicount 0.1')

n = int(args['<n>'])
order = int(args['<order>'])

count = jvi.jvi_size(n,order)
print count

