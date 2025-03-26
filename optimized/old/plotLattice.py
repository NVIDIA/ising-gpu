#!/usr/bin/env python

import sys
import numpy as np
from matplotlib import pyplot as plt

data = []
f=open(sys.argv[1])
for l in f:
    data.append([int(c) for c in l.strip(" \n\r")])

print len(data), 'x', len(data[0])

plt.imshow(data, interpolation='nearest')

outFile = sys.argv[1]+".png"
plt.savefig(outFile)
