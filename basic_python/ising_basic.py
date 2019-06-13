 # Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a
 # copy of this software and associated documentation files (the "Software"),
 # to deal in the Software without restriction, including without limitation
 # the rights to use, copy, modify, merge, publish, distribute, sublicense,
 # and/or sell copies of the Software, and to permit persons to whom the
 # Software is furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in
 # all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 # THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 # FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 # DEALINGS IN THE SOFTWARE.

import argparse
import math
import sys
import time

import cupy.cuda.curand as curand
from mpi4py import MPI
from numba import cuda
from numba import vectorize
import numpy as np

# Set constants
TCRIT = 2.26918531421 # critical temperature

# Setup MPI and get neighbor ranks
comm = MPI.COMM_WORLD
rank = comm.rank
rank_up = comm.rank - 1 if (comm.rank - 1 >= 0) else comm.size - 1
rank_down = comm.rank + 1 if (comm.rank + 1 < comm.size) else 0

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--lattice-n", '-x', type=int, default=40*128, help="number of lattice rows")
parser.add_argument("--lattice-m", '-y', type=int, default=40*128, help="number of lattice columns")
parser.add_argument("--nwarmup", '-w', type=int, default=100, help="number of warmup iterations")
parser.add_argument("--niters", '-n', type=int, default=1000, help="number of trial iterations")
parser.add_argument("--alpha", '-a', type=float, default=0.1, help="coefficient of critical temperature")
parser.add_argument("--seed", '-s', type=int, default=1234, help="seed for random number generation")
parser.add_argument("--write-lattice", '-o', action='store_true', help="write final lattice configuration to file/s")
parser.add_argument("--use-common-seed", '-c', action='store_true', help="Use common seed for all ranks + updating offset. " +
                                                                         "Yields consistent results independent of number " +
                                                                         "of GPUs but is slower.")
args = parser.parse_args()

# Check arguments
if args.lattice_m % 2 != 0:
    raise Exception("lattice_m must be an even value. Aborting.")
if args.lattice_n % comm.size != 0:
    raise Exception("lattice_n must be evenly divisible by number of GPUs. Aborting.")
if (args.lattice_n / comm.size) % 2 != 0:
    raise Exception("Slab width (lattice_n / nGPUs) must be an even value. Aborting.")

# Compute slab width
lattice_slab_n = args.lattice_n // comm.size

inv_temp = (1.0) / (args.alpha * TCRIT)

# Generate lattice with random spins with shape of randval array
@vectorize(['int8(float32)'], target='cuda')                             
def generate_lattice(randval):
    return 1 if randval > 0.5 else -1 

@cuda.jit
def update_lattice_multi(lattice, op_lattice, op_lattice_up, op_lattice_down, randvals, is_black):
    n,m = lattice.shape
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = tid % m
    i = tid // m

    if (i >= n or j >= m): return

    # Set stencil indices with periodicity
    jpp = (j + 1) if (j + 1) < m else 0
    jnn = (j - 1) if (j - 1) >= 0 else (m - 1)

    # Select off-column index based on color and row index parity
    if (is_black):
        joff = jpp if (i % 2) else jnn
    else:
        joff = jnn if (i % 2) else jpp

    # Compute sum of nearest neighbor spins (taking values from neighboring
    # lattice slabs if required)
    nn_sum = op_lattice[i, j] + op_lattice[i, joff]
    nn_sum += op_lattice[i - 1, j] if (i - 1) >= 0 else op_lattice_up[n - 1, j]
    nn_sum += op_lattice[i + 1, j] if (i + 1) < n else op_lattice_down[0, j]

    # Determine whether to flip spin
    lij = lattice[i, j]
    acceptance_ratio = math.exp(-2.0 * inv_temp * nn_sum * lij)
    if (randvals[i, j] < acceptance_ratio):
        lattice[i, j] = -lij

# Create lattice update kernel (for single GPU case, this version with fewer arguments
# is a bit faster due to launch overhead introduced by numba)
@cuda.jit
def update_lattice(lattice, op_lattice, randvals, is_black):
    n,m = lattice.shape
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = tid // m
    j = tid % m

    if (i >= n or j >= m): return

    # Set stencil indices with periodicity
    ipp = (i + 1) if (i + 1) < n else 0
    jpp = (j + 1) if (j + 1) < m else 0
    inn = (i - 1) if (i - 1) >= 0 else (n - 1)
    jnn = (j - 1) if (j - 1) >= 0 else (m - 1)

    # Select off-column index based on color and row index parity
    if (is_black):
        joff = jpp if (i % 2) else jnn
    else:
        joff = jnn if (i % 2) else jpp

    # Compute sum of nearest neighbor spins
    nn_sum = op_lattice[inn, j] + op_lattice[i, j] + op_lattice[ipp, j] + op_lattice[i, joff]

    # Determine whether to flip spin
    lij = lattice[i, j]
    acceptance_ratio = math.exp(-2.0 * inv_temp * nn_sum * lij)
    if (randvals[i, j] < acceptance_ratio):
        lattice[i, j] = -lij

# Write lattice configuration to file
def write_lattice(prefix, lattice_b, lattice_w):
  lattice_b_h = lattice_b.copy_to_host()
  lattice_w_h = lattice_w.copy_to_host()
  lattice = np.zeros((lattice_slab_n, args.lattice_m), dtype=np.int8)
  for i in range(lattice_slab_n):
      for j in range(args.lattice_m // 2):
          if (i % 2):
              lattice[i, 2*j+1] = lattice_b_h[i, j]
              lattice[i, 2*j] = lattice_w_h[i, j]
          else:
              lattice[i, 2*j] = lattice_b_h[i, j]
              lattice[i, 2*j+1] = lattice_w_h[i, j]

  print("Writing lattice to {}_rank{}.txt...".format(prefix, rank))
  np.savetxt("{}_rank{}.txt".format(prefix, rank), lattice, fmt='%d')

# Helper class for random number generation
class curandUniformRNG:
    def __init__(self, seed=0):
        rng = curand.createGenerator(curand.CURAND_RNG_PSEUDO_PHILOX4_32_10)
        curand.setPseudoRandomGeneratorSeed(rng, seed)
        if (args.use_common_seed):
            self.offset = rank * lattice_slab_n * args.lattice_m // 2
            curand.setGeneratorOffset(rng, self.offset)
        self._rng = rng

    def fill_random(self, arr):
        ptr = arr.__cuda_array_interface__['data'][0]
        curand.generateUniform(self._rng, ptr, arr.size)
        if (args.use_common_seed):
            self.offset += args.lattice_n * args.lattice_m // 2
            curand.setGeneratorOffset(self._rng, self.offset)

# Helper function to perform device sync plus MPI barrier
def sync():
  cuda.synchronize()
  comm.barrier()

def update(lattices_b, lattices_w, randvals, rng):
    # Setup CUDA launch configuration
    threads = 128
    blocks = (args.lattice_m // 2 * lattice_slab_n + threads - 1) // threads

    if (comm.size > 1):
        # Update black
        rng.fill_random(randvals)
        update_lattice_multi[blocks, threads](lattices_b[rank], lattices_w[rank], lattices_w[rank_up], lattices_w[rank_down], randvals, True)
        sync()
        # Update white
        rng.fill_random(randvals)
        update_lattice_multi[blocks, threads](lattices_w[rank], lattices_b[rank], lattices_b[rank_up], lattices_b[rank_down], randvals, False)
        sync()
    else:
        # Update black
        rng.fill_random(randvals)
        update_lattice[blocks, threads](lattices_b[rank], lattices_w[rank], randvals, True)
        # Update white
        rng.fill_random(randvals)
        update_lattice[blocks, threads](lattices_w[rank], lattices_b[rank], randvals, False)


# Set device
cuda.select_device(rank)

# Setup cuRAND generator
rng = curandUniformRNG(seed=args.seed if args.use_common_seed else args.seed + 42 * rank)
randvals = cuda.device_array((lattice_slab_n, args.lattice_m // 2), dtype=np.float32)

# Setup black and white lattice arrays on device
rng.fill_random(randvals)
lattice_b = generate_lattice(randvals)
rng.fill_random(randvals)
lattice_w = generate_lattice(randvals)

# Setup/open CUDA IPC handles
ipch_b = comm.allgather(lattice_b.get_ipc_handle())
ipch_w = comm.allgather(lattice_w.get_ipc_handle())
lattices_b = [x.open() if i != rank else lattice_b for i,x in enumerate(ipch_b)]
lattices_w = [x.open() if i != rank else lattice_w for i,x in enumerate(ipch_w)]

# Warmup iterations
if rank == 0:
    print("Starting warmup...")
    sys.stdout.flush()
sync()
for i in range(args.nwarmup):
    update(lattices_b, lattices_w, randvals, rng)
sync()

# Trial iterations
if rank == 0:
    print("Starting trial iterations...")
    sys.stdout.flush()
t0 = time.time()
for i in range(args.niters):
    update(lattices_b, lattices_w, randvals, rng)
    if (rank == 0 and i % 1000 == 0):
        print("Completed {}/{} iterations...".format(i+1, args.niters))
        sys.stdout.flush()
sync()

t1 = time.time()
t = t1 - t0

# Compute average magnetism
m = (np.sum(lattices_b[rank], dtype=np.int64) + np.sum(lattices_w[rank], dtype=np.int64)) / float(args.lattice_n * args.lattice_m)
m_global = comm.allreduce(m, MPI.SUM)

if (rank == 0):
  print("REPORT:")
  print("\tnGPUs: {}".format(comm.size))
  print("\ttemperature: {} * {}".format(args.alpha, TCRIT))
  print("\tseed: {}".format(args.seed))
  print("\twarmup iterations: {}".format(args.nwarmup))
  print("\ttrial iterations: {}".format(args.niters))
  print("\tlattice dimensions: {} x {}".format(args.lattice_n, args.lattice_m))
  print("\telapsed time: {} sec".format(t))
  print("\tupdates per ns: {}".format((args.lattice_n * args.lattice_m * args.niters) / t * 1e-9))
  print("\taverage magnetism (absolute): {}".format(np.abs(m_global)))
  sys.stdout.flush()

sync()

if (args.write_lattice):
    write_lattice("final", lattices_b[rank], lattices_w[rank])
