/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <chrono>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>

#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

#include <cub/cub.cuh>

#include "cudamacro.h"

#define LATTICE_SUP_N (256)
#define LATTICE_SUB_N (LATTICE_SUP_N / 2)
#define TCRIT 2.26918531421f
#define THREADS  (LATTICE_SUB_N)

#define SUP_OFFSET(i,j,nbx) (((j)*(long long)(nbx) + (i))*LATTICE_SUP_N*LATTICE_SUP_N)
#define SUB_OFFSET(i,j) (((j)*LATTICE_SUP_N + (i)*LATTICE_SUB_N)*LATTICE_SUB_N)
#define SUB_ELEM(i,j) ((j)*LATTICE_SUB_N + (i))

#define CUB_CHUNK_SIZE ((1ll<<31) - (1ll<<28))

__global__ void set_k(__half* k, __half* kT) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int i = tid % LATTICE_SUB_N;
  const int j = tid / LATTICE_SUB_N;
  if (j >= LATTICE_SUB_N) return;

  __half val = __float2half(0.0f);
  if (i == j || i + 1 == j) {
    val = __float2half(1.0f);
  }

  k[j*LATTICE_SUB_N + i] = val;
  kT[i*LATTICE_SUB_N + j] = val;
}

__global__ void init_spins(__half* lattice,
                           const unsigned long long seed,
                           const int nbx,
                           const int nby,
                           const long long offset) {
  const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x + offset;
  const long long nx = nbx * LATTICE_SUP_N;
  const long long ny = nby * LATTICE_SUP_N;
  if (tid >= nx * ny) return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, tid, 0, &state);
  float randval = curand_uniform(&state);
  __half val = (randval < 0.5f) ? __float2half(-1.0f) : __float2half(1.0f);

  lattice[tid] = val;
}

template <int N>
struct __align__(sizeof(__half)*N) halfn {
  __half val[N];
};

#define NLOOPS 2
#define SPINSPERTHREAD 8
template<bool is_black>
__global__ void update_spins(__half* lattice,
                             float inv_temp,
                             const __half* __restrict__ nn_sums,
                             const unsigned long long seed,
                             const unsigned long long iter,
                             const int nbx,
                             const int nby,
                             const long long  offset) {
  const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x + offset;

  const int threads_per_subblock = LATTICE_SUB_N * LATTICE_SUB_N / (NLOOPS * SPINSPERTHREAD);

  int bi = tid / threads_per_subblock % (2 * nbx);
  int bj = tid / (threads_per_subblock * 2 * nbx);

  // subblock local thread idx
  int tl = tid % threads_per_subblock;

  if (bj >= nby) return;

  // Offset threads depending on parity and color
  if (is_black) {
    if (bi % 2) {
      bj = 2*bj + 1;
    } else {
      bj = 2*bj;
    }
  } else {
    if (bi % 2) {
      bj = 2*bj;
    } else {
      bj = 2*bj + 1;
    }
  }

  curandStatePhilox4_32_10_t state;
  curand_init(seed, tid, iter, &state);

  #pragma unroll
  for (int n = 0; n < NLOOPS; n++) {
    size_t elem_offset = SUP_OFFSET(bi/2, bj/2, nbx) + SUB_OFFSET(bi%2, bj%2) + (tl + n * threads_per_subblock) * SPINSPERTHREAD;

    halfn<SPINSPERTHREAD> lij = *(reinterpret_cast<halfn<SPINSPERTHREAD>*>(lattice + elem_offset));
    const halfn<SPINSPERTHREAD> nn = *(reinterpret_cast<const halfn<SPINSPERTHREAD>*>(nn_sums + elem_offset));

    #pragma unroll
    for (int m = 0; m < SPINSPERTHREAD; m++) {
      float randval = curand_uniform(&state);
      float accept = exp(-2.0f * inv_temp * __half2float(nn.val[m] * lij.val[m]));
      if (randval < accept) {
        lij.val[m] = -lij.val[m];
      }
    }

    *reinterpret_cast<halfn<SPINSPERTHREAD>*>(lattice + elem_offset) = lij;

  }
}

template<bool is_black>
__global__ void add_boundaries(const __half* __restrict__ lattice,
                               __half* nn_sums,
                               const int nbx,
                               const int nby,
                               const long long offset) {
  const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x + offset;

  // subblock i,j (1 thread block per subblock)
  int bi = tid / LATTICE_SUB_N % (2 * nbx);
  int bj = tid / (LATTICE_SUB_N * 2 * nbx);

  // subblock local i
  int il = tid % LATTICE_SUB_N;

  if (bj >= nby) return;

  // Offset threads depending on parity and color
  int jl, jb;
  if (is_black) {
    if (bi % 2) {
      bj = 2*bj + 1;
      jl = LATTICE_SUB_N - 1;
      jb = 0;
    } else {
      bj = 2*bj;
      jl = 0;
      jb = LATTICE_SUB_N - 1;
    }
  } else {
    if (bi % 2) {
      bj = 2*bj;
      jl = 0;
      jb = LATTICE_SUB_N - 1;
    } else {
      bj = 2*bj + 1;
      jl = LATTICE_SUB_N - 1;
      jb = 0;
    }
  }

  int bn = 2*nbx;
  int bm = 2*nby;
  int bin = (bi - 1 >= 0) ? bi - 1 : bn - 1;
  int bip = (bi + 1 < bn) ? bi + 1 : 0;
  int bjn = (bj - 1 >= 0) ? bj - 1 : bm - 1;
  int bjp = (bj + 1 < bm) ? bj + 1 : 0;

  // Update LR
  size_t boundary_offset;
  if (jl == 0) {
    boundary_offset = SUP_OFFSET(bi/2, bjn/2, nbx) + SUB_OFFSET(bi%2, bjn%2);
  } else {
    boundary_offset = SUP_OFFSET(bi/2, bjp/2, nbx) + SUB_OFFSET(bi%2, bjp%2);
  }

  size_t local_offset = SUP_OFFSET(bi/2, bj/2, nbx) + SUB_OFFSET(bi%2, bj%2);
  *(nn_sums + local_offset + SUB_ELEM(il, jl)) += *(lattice + boundary_offset + SUB_ELEM(il, jb));


  // Update UD
  if (!is_black) {
    jl = (jl == 0) ? LATTICE_SUB_N - 1 : 0;
    jb = (jb == 0) ? LATTICE_SUB_N - 1 : 0;
  }

  if (jl == 0) {
    boundary_offset = SUP_OFFSET(bin/2, bj/2, nbx) + SUB_OFFSET(bin%2, bj%2);
  } else {
    boundary_offset = SUP_OFFSET(bip/2, bj/2, nbx) + SUB_OFFSET(bip%2, bj%2);
  }

  __half bval = *(lattice + boundary_offset + SUB_ELEM(jb, il));

  __syncthreads();

  *(nn_sums + local_offset + SUB_ELEM(jl, il)) += bval;

}

void sync(int nGPUs) {
  // Sync all devices
  for (int dev = 0; dev < nGPUs; dev++) {
    CHECK_CUDA(cudaSetDevice(dev));
    CHECK_CUDA(cudaDeviceSynchronize());
  }
}


void update(__half **Ab0, __half **Bb0, __half **Ab1, __half **Bb1, __half **Cb,
            __half **Aw0, __half **Bw0, __half **Aw1, __half **Bw1, __half **Cw,
            __half *lattice, float inv_temp, __half *nn_sums, cublasHandle_t *cublas_handles, int iter,
            int nbx, int nby, unsigned long long seed, int nGPUs) {

  int batchCount = 2 * nbx * nby;
  int batchCountPerGPU = batchCount / nGPUs;

  __half alpha = __float2half(1.0f);
  __half beta0 =  __float2half(0.0f);
  __half beta1 =  __float2half(1.0f);

  // Update black
  for (int dev = 0; dev < nGPUs; dev++) {
    CHECK_CUDA(cudaSetDevice(dev));
    CHECK_CUBLAS(cublasGemmBatchedEx(cublas_handles[dev], CUBLAS_OP_N, CUBLAS_OP_N, LATTICE_SUB_N, LATTICE_SUB_N, LATTICE_SUB_N,
                                     &alpha, (void**) &Ab0[dev * batchCountPerGPU], CUDA_R_16F, LATTICE_SUB_N,
                                     (void**) &Bb0[dev * batchCountPerGPU], CUDA_R_16F, LATTICE_SUB_N, &beta0,
                                     (void**) &Cb[dev * batchCountPerGPU], CUDA_R_16F, LATTICE_SUB_N, batchCountPerGPU,
                                     CUDA_R_16F, CUBLAS_GEMM_ALGO0_TENSOR_OP));

    CHECK_CUBLAS(cublasGemmBatchedEx(cublas_handles[dev], CUBLAS_OP_N, CUBLAS_OP_N, LATTICE_SUB_N, LATTICE_SUB_N, LATTICE_SUB_N,
                                     &alpha, (void**) &Ab1[dev * batchCountPerGPU], CUDA_R_16F, LATTICE_SUB_N,
                                     (void**) &Bb1[dev * batchCountPerGPU], CUDA_R_16F, LATTICE_SUB_N, &beta1,
                                     (void**) &Cb[dev * batchCountPerGPU], CUDA_R_16F, LATTICE_SUB_N, batchCountPerGPU,
                                     CUDA_R_16F, CUBLAS_GEMM_ALGO0_TENSOR_OP));

    int blocks = (2 * nbx * nby);
    int blocksPerGPU = blocks / nGPUs;
    add_boundaries<true><<<blocksPerGPU, THREADS>>>(lattice, nn_sums, nbx, nby, dev * ((long long)blocksPerGPU * THREADS));
    blocks = (2 * nbx * nby * LATTICE_SUB_N) / (NLOOPS * SPINSPERTHREAD);
    blocksPerGPU = blocks / nGPUs;
    update_spins<true><<<blocksPerGPU, THREADS>>>(lattice, inv_temp, nn_sums, seed, (2*iter) * (NLOOPS * SPINSPERTHREAD), nbx, nby, dev * ((long long)blocksPerGPU * THREADS));
  }

  sync(nGPUs);

  // Update white
  for (int dev = 0; dev < nGPUs; dev++) {
    cudaSetDevice(dev);
    CHECK_CUBLAS(cublasGemmBatchedEx(cublas_handles[dev], CUBLAS_OP_N, CUBLAS_OP_N, LATTICE_SUB_N, LATTICE_SUB_N, LATTICE_SUB_N,
                                     &alpha, (void**) &Aw0[dev * batchCountPerGPU], CUDA_R_16F, LATTICE_SUB_N,
                                     (void**) &Bw0[dev * batchCountPerGPU], CUDA_R_16F, LATTICE_SUB_N, &beta0,
                                     (void**) &Cw[dev * batchCountPerGPU], CUDA_R_16F, LATTICE_SUB_N, batchCountPerGPU,
                                     CUDA_R_16F, CUBLAS_GEMM_ALGO0_TENSOR_OP));

    CHECK_CUBLAS(cublasGemmBatchedEx(cublas_handles[dev], CUBLAS_OP_N, CUBLAS_OP_N, LATTICE_SUB_N, LATTICE_SUB_N, LATTICE_SUB_N,
                                     &alpha, (void**) &Aw1[dev * batchCountPerGPU], CUDA_R_16F, LATTICE_SUB_N,
                                     (void**) &Bw1[dev * batchCountPerGPU], CUDA_R_16F, LATTICE_SUB_N, &beta1,
                                     (void**) &Cw[dev * batchCountPerGPU], CUDA_R_16F, LATTICE_SUB_N, batchCountPerGPU,
                                     CUDA_R_16F, CUBLAS_GEMM_ALGO0_TENSOR_OP));

    int blocks = (2 * nbx * nby);
    int blocksPerGPU = blocks / nGPUs;
    add_boundaries<false><<<blocksPerGPU, THREADS>>>(lattice, nn_sums, nbx, nby, dev * ((long long)blocksPerGPU * THREADS));
    blocks = (2 * nbx * nby * LATTICE_SUB_N) / (NLOOPS * SPINSPERTHREAD);
    blocksPerGPU = blocks / nGPUs;
    update_spins<false><<<blocksPerGPU, THREADS>>>(lattice, inv_temp, nn_sums, seed, (2*iter + 1) * (NLOOPS * SPINSPERTHREAD), nbx, nby, dev * ((long long)blocksPerGPU * THREADS));
  }

  sync(nGPUs);
}

void write_lattice(__half *lattice, std::string filename, int nbx, int nby, int nGPUs) {
  printf("Writing lattice to %s...\n", filename.c_str());

  long long nx = nbx * LATTICE_SUP_N;
  long long ny = nby * LATTICE_SUP_N;

  __half* lattice_h;
  float* lattice_true_h;
  lattice_h = (__half*) malloc(nx * ny * sizeof(*lattice_h));
  lattice_true_h = (float*) malloc(nx * ny * sizeof(*lattice_true_h));

  long spinsPerGPU = nx * (ny/nGPUs);
  // Copy out full lattice to host
  for (int dev = 0; dev < nGPUs; dev++) {
    CHECK_CUDA(cudaSetDevice(dev));
    CHECK_CUDA(cudaMemcpy(&lattice_h[dev * spinsPerGPU], &lattice[dev * spinsPerGPU], spinsPerGPU * sizeof(*lattice_h), cudaMemcpyDeviceToHost));
  }

  // Write file
  for (int bj = 0; bj < nby; bj++) {
    for (int bi = 0; bi < nbx; bi++) {
      __half* l00 = lattice_h + SUP_OFFSET(bi, bj, nbx) + SUB_OFFSET(0, 0);
      __half* l01 = lattice_h + SUP_OFFSET(bi, bj, nbx) + SUB_OFFSET(0, 1);
      __half* l10 = lattice_h + SUP_OFFSET(bi, bj, nbx) + SUB_OFFSET(1, 0);
      __half* l11 = lattice_h + SUP_OFFSET(bi, bj, nbx) + SUB_OFFSET(1, 1);

      long long offset = (bj * LATTICE_SUP_N) * nx + (bi * LATTICE_SUP_N);
      for(int j = 0; j < LATTICE_SUB_N; j++) {
        for(int i = 0; i < LATTICE_SUB_N; i++) {
          lattice_true_h[offset + (2*j) * nx + (2*i)] = __half2float(*(l00 + SUB_ELEM(i, j)));
          lattice_true_h[offset + (2*j + 1) * nx + (2*i + 1)] = __half2float(*(l11 + SUB_ELEM(i, j)));
          lattice_true_h[offset + (2*j) * nx + (2*i + 1)] = __half2float(*(l10 + SUB_ELEM(i, j)));
          lattice_true_h[offset + (2*j + 1) * nx + (2*i)] = __half2float(*(l01 + SUB_ELEM(i, j)));
        }
      }
    }
  }

  std::ofstream f;
  f.open(filename);
  if (f.is_open()) {
    for (long long j = 0; j < ny; j++) {
      for (long long i = 0; i < nx; i++) {
         f << lattice_true_h[j * nx + i] << " ";
      }
      f << std::endl;
    }
  }
  f.close();

  free(lattice_h);
  free(lattice_true_h);
}

static void usage(const char *pname) {

  const char *bname = rindex(pname, '/');
  if (!bname) {bname = pname;}
  else        {bname++;}

  fprintf(stdout,
          "Usage: %s [options]\n"
          "options:\n"
          "\t-x|--lattice-nbx <LATTICE_NBX>\n"
          "\t\tnumber of blocks along lattice rows (number of rows / 256)\n"
          "\n"
          "\t-y|--lattice-nby <LATTICE_NBY>\n"
          "\t\tnumber of blocks along lattice columns (number of columns / 256)\n"
          "\n"
          "\t-g|--ngpus <NGPUS>\n"
          "\t\tnumber of GPUs to use for simulation\n"
          "\n"
          "\t-w|--nwarmup <NWARMUP>\n"
          "\t\tnumber of warmup iterations\n"
          "\n"
          "\t-n|--niters <NITERS>\n"
          "\t\tnumber of trial iterations\n"
          "\n"
          "\t-a|--alpha <ALPHA>\n"
          "\t\tcoefficient of critical temperature\n"
          "\n"
          "\t-s|--seed <SEED>\n"
          "\t\tseed for random number generation\n"
          "\n"
          "\t-o|--write-lattice\n"
          "\t\twrite final lattice configuration to file\n\n",
          bname);
  exit(EXIT_SUCCESS);
}

int main(int argc, char **argv) {

  // Defaults
  int nbx = 10; // Lattice rows dimension (in number of super blocks)
  int nby = 10; // Lattice columns dimension (in number of super blocks)
  float alpha = 0.1f; // coefficient of critical temperature
  int niter = 1000;
  int nwarmup = 100;
  bool write = false;
  int nGPUs = 1;
  unsigned long long seed = 1234ULL;

  while (1) {
    static struct option long_options[] = {
        {   "lattice-nbx", required_argument, 0, 'x'},
        {   "lattice-nby", required_argument, 0, 'y'},
        {         "ngpus", required_argument, 0, 'g'},
        {          "seed", required_argument, 0, 's'},
        {       "nwarmup", required_argument, 0, 'w'},
        {         "niter", required_argument, 0, 'n'},
        { "write-lattice",       no_argument, 0, 'o'},
        {          "help",       no_argument, 0, 'h'},
        {               0,                 0, 0,   0}
    };

    int option_index = 0;
    int ch = getopt_long(argc, argv, "x:y:g:a:s:w:n:oh", long_options, &option_index);
    if (ch == -1) break;

    switch(ch) {
      case 0:
        break;
      case 'x':
        nbx = atoi(optarg); break;
      case 'y':
        nby = atoi(optarg); break;
      case 'g':
        nGPUs = atoi(optarg); break;
      case 'a':
        alpha = atof(optarg); break;
      case 's':
        seed = atoll(optarg); break;
      case 'w':
        nwarmup = atoi(optarg); break;
      case 'n':
        niter = atoi(optarg); break;
      case 'o':
        write = true; break;
      case 'h':
        usage(argv[0]); break;
      case '?':
        exit(EXIT_FAILURE);
      default:
        fprintf(stderr, "unknown option: %c\n", ch);
        exit(EXIT_FAILURE);
    }
  }

  if (nby % nGPUs != 0) {
    fprintf(stderr, "ERROR: Number of super blocks in y dimension must be multiple of number of gpus.\n");
    exit(EXIT_FAILURE);
  }

  long long nx = nbx * LATTICE_SUP_N;
  long long ny = nby * LATTICE_SUP_N;

  __half* lattice;
  __half* nn_sums;
  __half* k;
  __half* kT;
  CHECK_CUDA(cudaMallocManaged(&lattice, nx * ny * sizeof(*lattice)));
  CHECK_CUDA(cudaMallocManaged(&nn_sums, nx * ny * sizeof(*nn_sums)));
  CHECK_CUDA(cudaMallocManaged(&k, LATTICE_SUB_N * LATTICE_SUB_N * sizeof(*k)));
  CHECK_CUDA(cudaMallocManaged(&kT, LATTICE_SUB_N * LATTICE_SUB_N * sizeof(*kT)));

  for (int dev = 0; dev < nGPUs; dev++) {
    CHECK_CUDA(cudaMemAdvise(k, LATTICE_SUB_N * LATTICE_SUB_N * sizeof(*k), cudaMemAdviseSetReadMostly, dev));
    CHECK_CUDA(cudaMemAdvise(k, LATTICE_SUB_N * LATTICE_SUB_N * sizeof(*k), cudaMemAdviseSetAccessedBy, dev));
    CHECK_CUDA(cudaMemAdvise(kT, LATTICE_SUB_N * LATTICE_SUB_N * sizeof(*kT), cudaMemAdviseSetReadMostly, dev));
    CHECK_CUDA(cudaMemAdvise(kT, LATTICE_SUB_N * LATTICE_SUB_N * sizeof(*kT), cudaMemAdviseSetAccessedBy, dev));
  }

  long long spinsPerGPU = nx * ny / nGPUs;
  for (int dev = 0; dev < nGPUs; dev++) {
    CHECK_CUDA(cudaMemAdvise(&lattice[dev * spinsPerGPU], spinsPerGPU * sizeof(*lattice), cudaMemAdviseSetPreferredLocation, dev));
    CHECK_CUDA(cudaMemAdvise(&nn_sums[dev * spinsPerGPU], spinsPerGPU * sizeof(*nn_sums), cudaMemAdviseSetPreferredLocation, dev));
  }

  cublasHandle_t* cublas_handles;
  cublas_handles = (cublasHandle_t*) malloc(nGPUs * sizeof(cublasHandle_t));
  for (int dev = 0; dev < nGPUs; dev++) {
    CHECK_CUDA(cudaSetDevice(dev));
    CHECK_CUBLAS(cublasCreate(&cublas_handles[dev]));
    CHECK_CUBLAS(cublasSetMathMode(cublas_handles[dev], CUBLAS_TENSOR_OP_MATH));
  }

  // Setup k and k transpose matrices
  CHECK_CUDA(cudaSetDevice(0));
  int blocks = (LATTICE_SUB_N * LATTICE_SUB_N +  THREADS - 1) / THREADS;
  set_k<<<blocks, THREADS>>>(k, kT);

  // Initialize lattice spins randomly
  for (int dev = 0; dev < nGPUs; dev++) {
    CHECK_CUDA(cudaSetDevice(dev));
    blocks = (nx * ny + THREADS - 1) / THREADS;
    int blocksPerGPU = blocks/nGPUs;
    init_spins<<<blocksPerGPU, THREADS>>>(lattice, seed, nbx, nby, dev * nx * (ny/nGPUs));
  }

  sync(nGPUs);

  // Setup pointers for batched GEMMS
  __half **Ab0, **Bb0;
  __half **Ab1, **Bb1;
  __half **Aw0, **Bw0;
  __half **Aw1, **Bw1;
  __half **Cb, **Cw;

  int batchCount = 2 * (nbx * nby);
  int batchCountPerGPU = batchCount / nGPUs;
  CHECK_CUDA(cudaMallocManaged(&Ab0, batchCount * sizeof(*Ab0)));
  CHECK_CUDA(cudaMallocManaged(&Bb0, batchCount * sizeof(*Bb0)));
  CHECK_CUDA(cudaMallocManaged(&Ab1, batchCount * sizeof(*Ab1)));
  CHECK_CUDA(cudaMallocManaged(&Bb1, batchCount * sizeof(*Bb1)));
  CHECK_CUDA(cudaMallocManaged(&Aw0, batchCount * sizeof(*Aw0)));
  CHECK_CUDA(cudaMallocManaged(&Bw0, batchCount * sizeof(*Bw0)));
  CHECK_CUDA(cudaMallocManaged(&Aw1, batchCount * sizeof(*Aw1)));
  CHECK_CUDA(cudaMallocManaged(&Bw1, batchCount * sizeof(*Bw1)));
  CHECK_CUDA(cudaMallocManaged(&Cb, batchCount * sizeof(*Cb)));
  CHECK_CUDA(cudaMallocManaged(&Cw, batchCount * sizeof(*Cw)));

  for (int dev = 0; dev < nGPUs; dev++) {
    CHECK_CUDA(cudaMemAdvise(&Ab0[dev * batchCountPerGPU], batchCountPerGPU * sizeof(*Ab0), cudaMemAdviseSetPreferredLocation, dev));
    CHECK_CUDA(cudaMemAdvise(&Bb0[dev * batchCountPerGPU], batchCountPerGPU * sizeof(*Bb0), cudaMemAdviseSetPreferredLocation, dev));
    CHECK_CUDA(cudaMemAdvise(&Ab1[dev * batchCountPerGPU], batchCountPerGPU * sizeof(*Ab1), cudaMemAdviseSetPreferredLocation, dev));
    CHECK_CUDA(cudaMemAdvise(&Bb1[dev * batchCountPerGPU], batchCountPerGPU * sizeof(*Bb1), cudaMemAdviseSetPreferredLocation, dev));
    CHECK_CUDA(cudaMemAdvise(&Aw0[dev * batchCountPerGPU], batchCountPerGPU * sizeof(*Aw0), cudaMemAdviseSetPreferredLocation, dev));
    CHECK_CUDA(cudaMemAdvise(&Bw0[dev * batchCountPerGPU], batchCountPerGPU * sizeof(*Bw0), cudaMemAdviseSetPreferredLocation, dev));
    CHECK_CUDA(cudaMemAdvise(&Aw1[dev * batchCountPerGPU], batchCountPerGPU * sizeof(*Aw1), cudaMemAdviseSetPreferredLocation, dev));
    CHECK_CUDA(cudaMemAdvise(&Bw1[dev * batchCountPerGPU], batchCountPerGPU * sizeof(*Bw1), cudaMemAdviseSetPreferredLocation, dev));
    CHECK_CUDA(cudaMemAdvise(&Cb[dev * batchCountPerGPU], batchCountPerGPU * sizeof(*Cb), cudaMemAdviseSetPreferredLocation, dev));
    CHECK_CUDA(cudaMemAdvise(&Cw[dev * batchCountPerGPU], batchCountPerGPU * sizeof(*Cw), cudaMemAdviseSetPreferredLocation, dev));
  }

  int idx = 0;

  for (int bj = 0; bj < nby; bj++) {
    for (int bi = 0; bi < nbx; bi++) {
      __half* nn_sums00 = nn_sums + SUP_OFFSET(bi, bj, nbx) + SUB_OFFSET(0, 0);
      __half* nn_sums11 = nn_sums + SUP_OFFSET(bi, bj, nbx) + SUB_OFFSET(1, 1);
      __half* nn_sums01 = nn_sums + SUP_OFFSET(bi, bj, nbx) + SUB_OFFSET(0, 1);
      __half* nn_sums10 = nn_sums + SUP_OFFSET(bi, bj, nbx) + SUB_OFFSET(1, 0);
      __half* lat00 = lattice + SUP_OFFSET(bi, bj, nbx) + SUB_OFFSET(0, 0);
      __half* lat11 = lattice + SUP_OFFSET(bi, bj, nbx) + SUB_OFFSET(1, 1);
      __half* lat01 = lattice + SUP_OFFSET(bi, bj, nbx) + SUB_OFFSET(0, 1);
      __half* lat10 = lattice + SUP_OFFSET(bi, bj, nbx) + SUB_OFFSET(1, 0);

      // Black:
      //nn_sum(0,0) = lattice(0,1) x K   + K^T x lattice(1,0)
      //nn_sum(1,1) = lattice(1,0) x K^T + K x lattice(0,1)
      Ab0[idx  ] = lat01; Bb0[idx  ] = k;
      Ab0[idx+1] = lat10; Bb0[idx+1] = kT;

      Ab1[idx  ] = kT; Bb1[idx  ] = lat10;
      Ab1[idx+1] = k;  Bb1[idx+1] = lat01;

      Cb[idx  ] = nn_sums00;
      Cb[idx+1] = nn_sums11;

      // White:
      //nn_sum(1,0) = lattice(1,1) x K   + K x lattice(0,0)
      //nn_sum(0,1) = lattice(0,0) x K^T + K^T x lattice(1,1)
      Aw0[idx  ] = lat00 ; Bw0[idx  ] = kT;
      Aw0[idx+1] = lat11 ; Bw0[idx+1] = k;

      Aw1[idx  ] = kT; Bw1[idx  ] = lat11;
      Aw1[idx+1] = k;  Bw1[idx+1] = lat00;

      Cw[idx  ] = nn_sums01;
      Cw[idx+1] = nn_sums10;

      idx += 2;

    }
  }

  sync(nGPUs);

  float inv_temp = 1.0f / (alpha*TCRIT);

  // Warmup
  printf("Starting warmup...\n");
  for (int n = 0; n < nwarmup; n++) {
    update(Ab0, Bb0, Ab1, Bb1, Cb, Aw0, Bw0, Aw1, Bw1, Cw,
           lattice, inv_temp, nn_sums, cublas_handles, n+1, nbx, nby, seed, nGPUs);
  }

  sync(nGPUs);
  printf("Starting trial iterations...\n");
  auto t0 = std::chrono::high_resolution_clock::now();

  for (int n = nwarmup; n < niter + nwarmup; n++) {
    update(Ab0, Bb0, Ab1, Bb1, Cb, Aw0, Bw0, Aw1, Bw1, Cw,
           lattice, inv_temp, nn_sums, cublas_handles, n+1, nbx, nby, seed, nGPUs);
    if ((n - nwarmup) % 1000 == 0) printf("Completed %d/%d iterations...\n", n - nwarmup + 1, niter);
  }

  sync(nGPUs);
  auto t1 = std::chrono::high_resolution_clock::now();

  double duration = (double) std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
  printf("REPORT:\n");
  printf("\tnGPUs: %d\n", nGPUs);
  printf("\ttemperature: %f * %f\n", alpha, TCRIT);
  printf("\tseed: %llu\n", seed);
  printf("\twarmup iterations: %d\n", nwarmup);
  printf("\ttrial iterations: %d\n", niter);
  printf("\tlattice dimensions: %lld x %lld\n", nx, ny);
  printf("\telapsed time: %f sec\n", duration * 1e-6);
  printf("\tupdates per ns: %f\n", (double) (nx * ny) * niter / duration * 1e-3);

  // Compute average magnetism
  double* devsums;
  int nchunks = (spinsPerGPU + CUB_CHUNK_SIZE - 1)/ CUB_CHUNK_SIZE;
  CHECK_CUDA(cudaMallocManaged(&devsums, nGPUs * nchunks * sizeof(*devsums)));
  for (int dev = 0 ; dev < nGPUs; dev++) {
    CHECK_CUDA(cudaSetDevice(dev));
    size_t cub_workspace_bytes = 0;
    void* workspace = NULL;

    CHECK_CUDA(cub::DeviceReduce::Sum(workspace, cub_workspace_bytes, &lattice[dev * spinsPerGPU], &devsums[dev*nchunks], CUB_CHUNK_SIZE));
    CHECK_CUDA(cudaMalloc(&workspace, cub_workspace_bytes));

    for (int n = 0; n < nchunks; n++) {
      CHECK_CUDA(cub::DeviceReduce::Sum(workspace, cub_workspace_bytes, &lattice[dev * spinsPerGPU + n*CUB_CHUNK_SIZE],
                             &devsums[dev * nchunks + n], std::min((long long) CUB_CHUNK_SIZE, spinsPerGPU - n * CUB_CHUNK_SIZE)));
    }
    CHECK_CUDA(cudaFree(workspace));
  }

  sync(nGPUs);

  double hostsum = 0;
  for (int n = 0; n < nGPUs * nchunks; n++) {
    hostsum += devsums[n];
  }
  std::cout << "\taverage magnetism (absolute): " << abs(hostsum / (nx * ny)) << std::endl;

  CHECK_CUDA(cudaFree(devsums));

  if (write) write_lattice(lattice, "final.txt", nbx, nby, nGPUs);

  return 0;
}
