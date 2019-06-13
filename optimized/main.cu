/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Mauro Bisson <maurob@nvidia.com>
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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <getopt.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "cudamacro.h" /* for time() */
#include "utils.h"

#define DIV_UP(a,b)     (((a)+((b)-1))/(b))

#define THREADS  128

#define BIT_X_SPIN (4)

#define CRIT_TEMP	(2.26918531421f)
#define	ALPHA_DEF	(0.1f)
#define MIN_TEMP	(0.05f*CRIT_TEMP)

#define MIN(a,b)	(((a)<(b))?(a):(b))
#define MAX(a,b)	(((a)>(b))?(a):(b))

// 2048+: 16, 16, 2, 1
//  1024: 16, 16, 1, 2
//   512:  8,  8, 1, 1
//   256:  4,  8, 1, 1
//   128:  2,  8, 1, 1

#define BLOCK_X (16)
#define BLOCK_Y (16)
#define BMULT_X (2)
#define BMULT_Y (1)

#define MAX_GPU	(256)

#define NUMIT_DEF (1)
#define SEED_DEF  (463463564571ull)

#define TGT_MAGN_MAX_DIFF (1.0E-3)

#define MAX_EXP_TIME (200)

__device__ __forceinline__ unsigned int __mypopc(const unsigned int x) {
	return __popc(x);
}

__device__ __forceinline__ unsigned long long int __mypopc(const unsigned long long int x) {
	return __popcll(x);
}

enum {C_BLACK, C_WHITE};

__device__ __forceinline__ uint2 __mymake_int2(const unsigned int x,
		                               const unsigned int y) {
	return make_uint2(x, y);
}

__device__ __forceinline__ ulonglong2 __mymake_int2(const unsigned long long x,
		                                    const unsigned long long y) {
	return make_ulonglong2(x, y);
}

template<int BDIM_X,
	 int BDIM_Y,
	 int LOOP_X, 
	 int LOOP_Y,
	 int BITXSP,
	 int COLOR,
	 typename INT_T,
	 typename INT2_T>
__global__  void latticeInit_k(const int devid,
			       const long long seed,
                               const int it,
                               const long long begY,
                               const long long dimX, // ld
                               INT2_T *__restrict__ vDst) {

	const int __i = blockIdx.y*BDIM_Y*LOOP_Y + threadIdx.y;
	const int __j = blockIdx.x*BDIM_X*LOOP_X + threadIdx.x;

	const int SPIN_X_WORD = 8*sizeof(INT_T)/BITXSP;

	const long long tid = ((devid*gridDim.y + blockIdx.y)*gridDim.x + blockIdx.x)*BDIM_X*BDIM_Y +
	                       threadIdx.y*BDIM_X + threadIdx.x;

	curandStatePhilox4_32_10_t st;
	curand_init(seed, tid, static_cast<long long>(2*SPIN_X_WORD)*LOOP_X*LOOP_Y*(2*it+COLOR), &st);

	INT2_T __tmp[LOOP_Y][LOOP_X];
	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__tmp[i][j] = __mymake_int2(INT_T(0),INT_T(0));
		}
	}

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			#pragma unroll
			for(int k = 0; k < 8*sizeof(INT_T); k += BITXSP) {
				if (curand_uniform(&st) < 0.5f) {
					__tmp[i][j].x |= INT_T(1) << k;
				}
				if (curand_uniform(&st) < 0.5f) {
					__tmp[i][j].y |= INT_T(1) << k;
				}
			}
		}
	}

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			vDst[(begY + __i+i*BDIM_Y)*dimX + __j+j*BDIM_X] = __tmp[i][j];
		}
	}
	return;
}

template<int BDIM_X,
	 int BDIM_Y,
	 int TILE_X,
	 int TILE_Y,
	 int FRAME_X,
	 int FRAME_Y,
	 typename INT_T,
	 typename INT2_T>
__device__ void loadTileOLD(const long long begY,
			 const long long dimY,
			 const long long dimX,
			 const INT2_T *__restrict__ v,
			       INT2_T tile[][TILE_X+2*FRAME_X]) {

	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;

	const int blkx = blockIdx.x;
	const int blky = blockIdx.y;

	const int FULL_X = TILE_X+2*FRAME_X;
	const int FULL_Y = TILE_Y+2*FRAME_Y;

	#pragma unroll
	for(int j = 0; j < FULL_Y; j += BDIM_Y) {

		const int yoff = begY + blky*TILE_Y + j+tidy - FRAME_Y;

		const int yoffAdj = (yoff < 0) ? dimY+yoff : (yoff >= dimY ? yoff-dimY : yoff);

		#pragma unroll
		for(int i = 0; i < FULL_X; i += BDIM_X) {

			const int xoff = blkx*TILE_X + i+tidx - FRAME_X;

			const int xoffAdj = (xoff < 0) ? dimX+xoff : (xoff >= dimX ? xoff-dimX : xoff);

			INT2_T __t = v[yoffAdj*dimX + xoffAdj];

			if (j+tidy < FULL_Y && i+tidx < FULL_X) {
				tile[j+tidy][i+tidx] = __t;
			}
		}
	}
	return;
}

template<int BDIM_X,
	 int BDIM_Y,
	 int TILE_X,
	 int TILE_Y,
	 int FRAME_X,
	 int FRAME_Y,
	 typename INT2_T>
__device__ void loadTile(const long long begY,
			 const long long dimY,
			 const long long dimX,
			 const INT2_T *__restrict__ v,
			       INT2_T tile[][TILE_X+2*FRAME_X]) {

	const int blkx = blockIdx.x;
	const int blky = blockIdx.y;

	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;

	#pragma unroll
	for(int j = 0; j < TILE_Y; j += BDIM_Y) {
		int yoff = begY + blky*TILE_Y + j+tidy;

		#pragma unroll
		for(int i = 0; i < TILE_X; i += BDIM_X) {
			const int xoff = blkx*TILE_X + i+tidx;
			tile[FRAME_Y + j+tidy][FRAME_X + i+tidx] = v[yoff*dimX + xoff];
		}
	}
	if (tidy == 0) {
		int yoff = (begY == 0 && blky == 0) ? dimY-1 : begY + blky*TILE_Y-1;
		#pragma unroll
		for(int i = 0; i < TILE_X; i += BDIM_X) {
			const int xoff = blkx*TILE_X + i+tidx;
			tile[0][FRAME_X + i+tidx] = v[yoff*dimX + xoff];
		}

		yoff = (begY+gridDim.y*TILE_Y == dimY && blky == gridDim.y-1) ? 0 : begY + blky*TILE_Y + TILE_Y;
		#pragma unroll
		for(int i = 0; i < TILE_X; i += BDIM_X) {
			const int xoff = blkx*TILE_X + i+tidx;
			tile[FRAME_Y + TILE_Y][FRAME_X + i+tidx] = v[yoff*dimX + xoff];
		}

		// the other branch in slower so skip it if possible
		if (BDIM_X <= TILE_Y) {
			int xoff = (blkx == 0) ? dimX-1 : blkx*TILE_X - 1;
			#pragma unroll
			for(int j = 0; j < TILE_Y; j += BDIM_X) {
				yoff = begY + blky*TILE_Y + j+tidx;
				tile[FRAME_Y + j+tidx][0] = v[yoff*dimX + xoff];
			}

			xoff = (blkx == gridDim.x-1) ? 0 : blkx*TILE_X + TILE_X;
			#pragma unroll
			for(int j = 0; j < TILE_Y; j += BDIM_X) {
				yoff = begY + blky*TILE_Y + j+tidx;
				tile[FRAME_Y + j+tidx][FRAME_X + TILE_X] = v[yoff*dimX + xoff];
			}
		} else {
			if (tidx < TILE_Y) {
				int xoff = (blkx == 0) ? dimX-1 : blkx*TILE_X - 1;
				yoff = begY + blky*TILE_Y + tidx;
				tile[FRAME_Y + tidx][0] = v[yoff*dimX + xoff];;

				xoff = (blkx == gridDim.x-1) ? 0 : blkx*TILE_X + TILE_X;
				yoff = begY + blky*TILE_Y + tidx;
				tile[FRAME_Y + tidx][FRAME_X + TILE_X] = v[yoff*dimX + xoff];
			}
		}
	}
	return;
}

template<int BDIM_X,
	 int BDIM_Y,
	 int LOOP_X, 
	 int LOOP_Y,
	 int BITXSP,
	 int COLOR,
	 typename INT_T,
	 typename INT2_T>
__global__ 
void spinUpdateV_2D_k(const int devid,
		      const long long seed,
		      const int it,
		      const long long begY,
		      const long long totY,
		      const long long dimX, // ld
		      const float vExp[][5],
		      const INT2_T *__restrict__ vSrc,
		            INT2_T *__restrict__ vDst) {

	const int SPIN_X_WORD = 8*sizeof(INT_T)/BITXSP;

	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;

	__shared__ INT2_T shTile[BDIM_Y*LOOP_Y+2][BDIM_X*LOOP_X+2];

	loadTile<BDIM_X, BDIM_Y,
		 BDIM_X*LOOP_X,
		 BDIM_Y*LOOP_Y, 
		 1, 1, INT2_T>(begY, totY, dimX, vSrc, shTile);

	// __shExp[cur_s{0,1}][sum_s{0,1}] = __expf(-2*cur_s{-1,+1}*F{+1,-1}(sum_s{0,1})*INV_TEMP)
	__shared__ float __shExp[2][5];

	// for small lattices BDIM_X/Y may be smaller than 2/5
	#pragma unroll
	for(int i = 0; i < 2; i += BDIM_Y) {
		#pragma unroll
		for(int j = 0; j < 5; j += BDIM_X) {
			if (i+tidy < 2 && j+tidx < 5) {
				__shExp[i+tidy][j+tidx] = vExp[i+tidy][j+tidx];
			}
		}
	}
	__syncthreads();

	const int __i = blockIdx.y*BDIM_Y*LOOP_Y + threadIdx.y;
	const int __j = blockIdx.x*BDIM_X*LOOP_X + threadIdx.x;

	const long long tid = ((devid*gridDim.y + blockIdx.y)*gridDim.x + blockIdx.x)*BDIM_X*BDIM_Y +
	                       threadIdx.y*BDIM_X + threadIdx.x;

	INT2_T __me[LOOP_Y][LOOP_X];

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__me[i][j] = vDst[(begY+__i+i*BDIM_Y)*dimX + __j+j*BDIM_X];
		}
	}

	INT2_T __up[LOOP_Y][LOOP_X];
	INT2_T __ct[LOOP_Y][LOOP_X];
	INT2_T __dw[LOOP_Y][LOOP_X];

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__up[i][j] = shTile[i*BDIM_Y +   tidy][j*BDIM_X + 1+tidx];
			__ct[i][j] = shTile[i*BDIM_Y + 1+tidy][j*BDIM_X + 1+tidx];
			__dw[i][j] = shTile[i*BDIM_Y + 2+tidy][j*BDIM_X + 1+tidx];
		}
	}

	// BDIM_Y is power of two so row parity won't change across loops
	const int readBack = (COLOR == C_BLACK) ? !(__i%2) : (__i%2);

	INT2_T __sd[LOOP_Y][LOOP_X];

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__sd[i][j] = (readBack) ? shTile[i*BDIM_Y + 1+tidy][j*BDIM_X +   tidx]:
						  shTile[i*BDIM_Y + 1+tidy][j*BDIM_X + 2+tidx];
		}
	}

	if (readBack) {
		#pragma unroll
		for(int i = 0; i < LOOP_Y; i++) {
			#pragma unroll
			for(int j = 0; j < LOOP_X; j++) {
				__sd[i][j].x = (__ct[i][j].x << BITXSP) | (__sd[i][j].y >> (8*sizeof(__sd[i][j].y)-BITXSP));
				__sd[i][j].y = (__ct[i][j].y << BITXSP) | (__ct[i][j].x >> (8*sizeof(__ct[i][j].x)-BITXSP));
			}
		}
	} else {
		#pragma unroll
		for(int i = 0; i < LOOP_Y; i++) {
			#pragma unroll
			for(int j = 0; j < LOOP_X; j++) {
				__sd[i][j].y = (__ct[i][j].y >> BITXSP) | (__sd[i][j].x << (8*sizeof(__sd[i][j].x)-BITXSP));
				__sd[i][j].x = (__ct[i][j].x >> BITXSP) | (__ct[i][j].y << (8*sizeof(__ct[i][j].y)-BITXSP));
			}
		}
	}
	curandStatePhilox4_32_10_t st;
	curand_init(seed, tid, static_cast<long long>(2*SPIN_X_WORD)*LOOP_X*LOOP_Y*(2*it+COLOR), &st);

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__ct[i][j].x += __up[i][j].x;
			__dw[i][j].x += __sd[i][j].x;
			__ct[i][j].x += __dw[i][j].x;

			__ct[i][j].y += __up[i][j].y;
			__dw[i][j].y += __sd[i][j].y;
			__ct[i][j].y += __dw[i][j].y;
		}
	}

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			#pragma unroll
			for(int z = 0; z < 8*sizeof(INT_T); z += BITXSP) {

				const int2 __src = make_int2((__me[i][j].x >> z) & 0xF,
							     (__me[i][j].y >> z) & 0xF);

				const int2 __sum = make_int2((__ct[i][j].x >> z) & 0xF,
							     (__ct[i][j].y >> z) & 0xF);

				const INT_T ONE = static_cast<INT_T>(1);

				if (curand_uniform(&st) <= __shExp[__src.x][__sum.x]) {
					__me[i][j].x ^= ONE << z;
				}
				if (curand_uniform(&st) <= __shExp[__src.y][__sum.y]) {
					__me[i][j].y ^= ONE << z;
				}
			}
		}
	}

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			vDst[(begY + __i+i*BDIM_Y)*dimX + __j+j*BDIM_X] = __me[i][j];
		}
	}
	return;
}

template<int BDIM_X,
	 int WSIZE,
	 typename T>
__device__ __forceinline__ T __block_sum(T v) {

	__shared__ T sh[BDIM_X/WSIZE];

	const int lid = threadIdx.x%32;
	const int wid = threadIdx.x/32;

	#pragma unroll
	for(int i = WSIZE/2; i; i >>= 1) {
		v += __shfl_down_sync(0xFFFFFFFF, v, i);
	}
	if (lid == 0) sh[wid] = v;

	__syncthreads();
	if (wid == 0) {
		v = (lid < (BDIM_X/WSIZE)) ? sh[lid] : 0;

		#pragma unroll
		for(int i = (BDIM_X/WSIZE)/2; i; i >>= 1) {
			v += __shfl_down_sync(0xFFFFFFFF, v, i);
		}
	}
	__syncthreads();
	return v;
}

// to be optimized
template<int BDIM_X,
	 int BITXSP,
         typename INT_T,
	 typename SUM_T>
__global__ void getMagn_k(const long long n,
			  const INT_T *__restrict__ v,
			        SUM_T *__restrict__ sum) {

	const int SPIN_X_WORD = 8*sizeof(INT_T)/BITXSP;

	const long long nth = static_cast<long long>(blockDim.x)*gridDim.x;
	const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

	SUM_T __cntP = 0;
	SUM_T __cntN = 0;

	for(long long i = 0; i < n; i += nth) {
		if (i+tid < n) {
			const int __c = __mypopc(v[i+tid]);
			__cntP += __c;
			__cntN += SPIN_X_WORD - __c;
		}
	}

	__cntP = __block_sum<BDIM_X, 32>(__cntP);
	__cntN = __block_sum<BDIM_X, 32>(__cntN);

	if (threadIdx.x == 0) {
		atomicAdd(sum+0, __cntP);
		atomicAdd(sum+1, __cntN);
	}
	return;
}

static void usage(const int SPIN_X_WORD, const char *pname) {

        const char *bname = rindex(pname, '/');
        if (!bname) {bname = pname;}
        else        {bname++;}

        fprintf(stdout,
                "Usage: %s [options]\n"
                "options:\n"
                "\t-x|--x <HORIZ_DIM>\n"
		"\t\tSpecifies the horizontal dimension of the entire  lattice  (black+white  spins),\n"
		"\t\tper GPU. This dimension must be a multiple of %d.\n"
                "\n"
                "\t-y|--y <VERT_DIM>\n"
		"\t\tSpecifies the vertical dimension of the entire lattice (black+white spins),  per\n"
		"\t\tGPU. This dimension must be a multiple of %d.\n"
                "\n"
                "\t-n|--n <NSTEPS>\n"
		"\t\tSpecifies the number of iteration to run.\n"
		"\t\tDefualt: %d\n"
                "\n"
                "\t-d|--devs <NUM_DEVICES>\n"
		"\t\tSpecifies the number of GPUs to use. Will use devices with ids [0, NUM_DEVS-1].\n"
		"\t\tDefualt: 1.\n"
                "\n"
                "\t-s|--seed <SEED>\n"
		"\t\tSpecifies the seed used to generate random numbers.\n"
		"\t\tDefault: %llu\n"
                "\n"
                "\t-a|--alpha <ALPHA>\n"
		"\t\tSpecifies the temperature in T_CRIT units.  If both this  option  and  '-t'  are\n"
		"\t\tspecified then the '-t' option is used.\n"
		"\t\tDefault: %f\n"
                "\n"
                "\t-t|--temp <TEMP>\n"
		"\t\tSpecifies the temperature in absolute units.  If both this option and  '-a'  are\n"
		"\t\tspecified then this option is used.\n"
		"\t\tDefault: %f\n"
                "\n"
                "\t-p|--print <STAT_FREQ>\n"
		"\t\tSpecifies the frequency, in no.  of  iteration,  with  which  the  magnetization\n"
		"\t\tstatistics is printed.  If this option is used together to the '-e' option, this\n"
		"\t\toption is ignored.\n"
		"\t\tDefault: only at the beginning and at end of the simulation\n"
                "\n"
                "\t-e|--exppr\n"
		"\t\tPrints the magnetization at time steps in the series 0 <= 2^(x/4) < NSTEPS.   If\n"
		"\t\tthis option is used  together  to  the  '-p'  option,  the  latter  is  ignored.\n"
		"\t\tDefault: disabled\n"
                "\n"
                "\t-m|--magn <TGT_MAGN>\n"
		"\t\tSpecifies the magnetization value at which the simulation is  interrupted.   The\n"
		"\t\tmagnetization of the system is checked against TGT_MAGN every STAT_FREQ, if  the\n"
		"\t\t'-p' option is specified, or according to the exponential  timestep  series,  if\n"
		"\t\tthe '-e' option is specified.  If neither '-p' not '-e' are specified then  this\n"
		"\t\toption is ignored.\n"
		"\t\tDefault: unset\n"
                "\n"
                "\t-o|--o\n"
		"\t\tEnables the file dump of  the lattice  every time  the magnetization is printed.\n"
		"\t\tDefault: off\n\n",
                bname,
		2*SPIN_X_WORD*2*BLOCK_X*BMULT_X,
		BLOCK_Y*BMULT_Y,
		NUMIT_DEF,
		SEED_DEF,
		ALPHA_DEF,
		ALPHA_DEF*CRIT_TEMP);
        exit(EXIT_SUCCESS);
}

static void countSpins(const int ndev,
		       const int redBlocks,
		       const size_t llen,
		       const size_t llenLoc,
		       const unsigned long long *black_d,
		       const unsigned long long *white_d,
			     unsigned long long **sum_d,
			     unsigned long long *bsum,
			     unsigned long long *wsum) {

	if (ndev == 1) {
		CHECK_CUDA(cudaMemset(sum_d[0], 0, 2*sizeof(*sum_d)));
		getMagn_k<THREADS, BIT_X_SPIN><<<redBlocks, THREADS>>>(llen, black_d, sum_d[0]);
		CHECK_ERROR("getMagn_k");
	} else {
		for(int i = 0; i < ndev; i++) {

			CHECK_CUDA(cudaSetDevice(i));
			CHECK_CUDA(cudaMemset(sum_d[i], 0, 2*sizeof(**sum_d)));
			getMagn_k<THREADS, BIT_X_SPIN><<<redBlocks, THREADS>>>(llenLoc, black_d + i*llenLoc, sum_d[i]);
			getMagn_k<THREADS, BIT_X_SPIN><<<redBlocks, THREADS>>>(llenLoc, white_d + i*llenLoc, sum_d[i]);
			CHECK_ERROR("getMagn_k");
		}
	}

	bsum[0] = 0;
	wsum[0] = 0;

	unsigned long long  sum_h[MAX_GPU][2];

	for(int i = 0; i < ndev; i++) {
		CHECK_CUDA(cudaMemcpy(sum_h[i], sum_d[i], 2*sizeof(**sum_h), cudaMemcpyDeviceToHost));
		bsum[0] += sum_h[i][0];
		wsum[0] += sum_h[i][1];
	}
	return;
}

static void dumpLattice(const char *fprefix,
			const int ndev,
			const int Y,
			const size_t lld,
		        const size_t llen,
		        const size_t llenLoc,
		        const unsigned long long *v_d) {

	char fname[256];

	if (ndev == 1) {
		unsigned long long *v_h = (unsigned long long *)Malloc(llen*sizeof(*v_h));
		CHECK_CUDA(cudaMemcpy(v_h, v_d, llen*sizeof(*v_h), cudaMemcpyDeviceToHost));

		unsigned long long *black_h = v_h;
		unsigned long long *white_h = v_h + llen/2;

		snprintf(fname, sizeof(fname), "%s0.txt", fprefix);
		FILE *fp = Fopen(fname, "w");

		for(int i = 0; i < Y; i++) {
			for(int j = 0; j < lld; j++) {
				unsigned long long __b = black_h[i*lld + j];
				unsigned long long __w = white_h[i*lld + j];

				for(int k = 0; k < 8*sizeof(*v_h); k += BIT_X_SPIN) {
					if (i&1) {
						fprintf(fp, "%llX",  (__w >> k) & 0xF);
						fprintf(fp, "%llX",  (__b >> k) & 0xF);
					} else {
						fprintf(fp, "%llX",  (__b >> k) & 0xF);
						fprintf(fp, "%llX",  (__w >> k) & 0xF);
					}
				}
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
		free(v_h);
	} else {
		#pragma omp parallel for schedule(static)
		for(int d = 0; d < ndev; d++) {
			const unsigned long long *black_h = v_d +          d*llenLoc;
			const unsigned long long *white_h = v_d + llen/2 + d*llenLoc;

			snprintf(fname, sizeof(fname), "%s%d.txt", fprefix, d);
			FILE *fp = Fopen(fname, "w");

			for(int i = 0; i < Y; i++) {
				for(int j = 0; j < lld; j++) {
					unsigned long long __b = black_h[i*lld + j];
					unsigned long long __w = white_h[i*lld + j];

					for(int k = 0; k < 8*sizeof(*black_h); k += BIT_X_SPIN) {
						if (i&1) {
							fprintf(fp, "%llX",  (__w >> k) & 0xF);
							fprintf(fp, "%llX",  (__b >> k) & 0xF);
						} else {
							fprintf(fp, "%llX",  (__b >> k) & 0xF);
							fprintf(fp, "%llX",  (__w >> k) & 0xF);
						}
					}
				}
				fprintf(fp, "\n");
			}
			fclose(fp);
		}
	}
	return;
}

static void generate_times(unsigned long long nsteps,
			   unsigned long long *list_times) {
	int nt = 0;

	list_times[0]=0;

	unsigned long long t = 0;
	for(unsigned long long j = 0; j < nsteps && t < nsteps; j++) {
		t = rint(pow(2.0, j/4.0));
		if (t > list_times[nt] && nt < MAX_EXP_TIME-1) {
			nt++;
			list_times[nt] = t;
			//printf("list_times[%d]: %llu\n", nt, list_times[nt]);
		}
	}
	return;
}

int main(int argc, char **argv) {

	unsigned long long *v_d;
	unsigned long long *black_d;
	unsigned long long *white_d;

	cudaEvent_t start, stop;
        float et;

	const int SPIN_X_WORD = (8*sizeof(*v_d)) / BIT_X_SPIN;

	int X = 0;
	int Y = 0;

	int dumpOut = 0;

	int nsteps = NUMIT_DEF;

	unsigned long long seed = SEED_DEF;

	int ndev = 1;

	float alpha = -1.0f;
	float temp  = -1.0f;

	float tempUpdStep = 0;
	int   tempUpdFreq = 0;

	int printFreq = 0;

	int printExp = 0;
	int printExpCur = 0;
	unsigned long long printExpSteps[MAX_EXP_TIME];

	double tgtMagn = -1.0;

	int och;
	while(1) {
		int option_index = 0;
		static struct option long_options[] = {
			{     "x", required_argument, 0, 'x'},
			{     "y", required_argument, 0, 'y'},
			{   "nit", required_argument, 0, 'n'},
			{  "seed", required_argument, 0, 's'},
			{   "out",       no_argument, 0, 'o'},
			{  "devs", required_argument, 0, 'd'},
			{ "alpha", required_argument, 0, 'a'},
			{  "temp", required_argument, 0, 't'},
			{ "print", required_argument, 0, 'p'},
			{"update", required_argument, 0, 'u'},
			{  "magn", required_argument, 0, 'm'},
			{ "exppr",       no_argument, 0, 'e'},
			{  "help",       no_argument, 0, 'h'},
			{       0,                 0, 0,   0}
		};

		och = getopt_long(argc, argv, "x:y:n:ohs:d:a:t:p:u:m:ec", long_options, &option_index);
		if (och == -1) break;
		switch (och) {
			case   0:// handles long opts with non-NULL flag field
				break;
			case 'x':
				X = atoi(optarg);
				break;
			case 'y':
				Y = atoi(optarg);
				break;
			case 'n':
				nsteps = atoi(optarg);
				break;
			case 'o':
				dumpOut = 1;
				break;
			case 'h':
				usage(SPIN_X_WORD, argv[0]);
				break;
			case 's':
				seed = atoll(optarg);
				break;
			case 'd':
				ndev = atoi(optarg);
				break;
			case 'a':
				alpha = atof(optarg);
				break;
			case 't':
				temp = atof(optarg);
				break;
			case 'p':
				printFreq = atoi(optarg);
				break;
			case 'e':
				printExp = 1;
				break;
			case 'u':
				// format -u FLT,INT
				{
					char *__tmp0 = strtok(optarg, ",");
					if (!__tmp0) {
						fprintf(stderr, "cannot find temperature step in parameter...\n");
						exit(EXIT_FAILURE);
					}
					char *__tmp1 = strtok(NULL, ",");
					if (!__tmp1) {
						fprintf(stderr, "cannot find iteration count in parameter...\n");
						exit(EXIT_FAILURE);
					}
					tempUpdStep = atof(__tmp0);
					tempUpdFreq = atoi(__tmp1);
					printf("tempUpdStep: %f, tempUpdFreq: %d\n", tempUpdStep, tempUpdFreq);
				}
				break;
			case 'm':
				tgtMagn = atof(optarg);
				break;
			case '?':
				exit(EXIT_FAILURE);
			default:
				fprintf(stderr, "unknown option: %c\n", och);
				exit(EXIT_FAILURE);
		}
	}

	if (!X || !Y) {
		if (!X) {
			if (Y && !(Y % (2*SPIN_X_WORD*2*BLOCK_X*BMULT_X))) {
				X = Y;
			} else {
				X = 2*SPIN_X_WORD*2*BLOCK_X*BMULT_X;
			}
		}
		if (!Y) {
			if (!(X%(BLOCK_Y*BMULT_Y))) {
				Y = X;
			} else {
				Y = BLOCK_Y*BMULT_Y;
			}
		}
	}

	if (!X || (X%2) || ((X/2)%(SPIN_X_WORD*2*BLOCK_X*BMULT_X))) {
		fprintf(stderr, "\nPlease specify an X dim multiple of %d\n\n", 2*SPIN_X_WORD*2*BLOCK_X*BMULT_X);
		usage(SPIN_X_WORD, argv[0]);
		exit(EXIT_FAILURE);
	}
	if (!Y || (Y%(BLOCK_Y*BMULT_Y))) {
		fprintf(stderr, "\nPlease specify a Y dim multiple of %d\n\n", BLOCK_Y*BMULT_Y);
		usage(SPIN_X_WORD, argv[0]);
		exit(EXIT_FAILURE);
	}

	if (temp == -1.0f) {
		if (alpha == -1.0f) {
			temp = ALPHA_DEF*CRIT_TEMP;
		} else {
			temp = alpha*CRIT_TEMP;
		}
	}

	if (printExp && printFreq) {
		printFreq = 0;
	}

	if (printExp) {
		generate_times(nsteps, printExpSteps);
	}

	cudaDeviceProp props;

	printf("\nUsing GPUs:\n");
	for(int i = 0; i < ndev; i++) {
		CHECK_CUDA(cudaGetDeviceProperties(&props, i));
		printf("\t%2d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n",
			i, props.name, props.multiProcessorCount,
			props.maxThreadsPerMultiProcessor,
			props.major, props.minor,
			props.ECCEnabled?"on":"off");
	}
	printf("\n");
	// we assums all gpus to be the same so we'll later
	// use the props filled for the last GPU...

	if (ndev > 1) {
		for(int i = 0; i < ndev; i++) {
			int attVal = 0;
			CHECK_CUDA(cudaDeviceGetAttribute(&attVal, cudaDevAttrConcurrentManagedAccess, i));
			if (!attVal) {
				fprintf(stderr,
					"error: device %d does not support concurrent managed memory access!\n", i);
				exit(EXIT_FAILURE);
			}
		}

		printf("GPUs direct access matrix:\n       ");
		for(int i = 0; i < ndev; i++) {
			printf("%4d", i);
		}
		int missingLinks = 0;
		printf("\n");
		for(int i = 0; i < ndev; i++) {
			printf("GPU %2d:", i);
			CHECK_CUDA(cudaSetDevice(i)); 
			for(int j = 0; j < ndev; j++) {
				int access = 1;
				if (i != j) {
					CHECK_CUDA(cudaDeviceCanAccessPeer(&access, i, j));
					if (access) {
						CHECK_CUDA(cudaDeviceEnablePeerAccess(j, 0));
					} else {
						missingLinks++;
					}
				}
				printf("%4c", access ? 'V' : 'X');
			}
			printf("\n");
		}
		printf("\n");
		if (missingLinks) {
			fprintf(stderr,
				"error: %d direct memory links among devices missing\n",
				missingLinks);
			exit(EXIT_FAILURE);
		}
	}

	size_t lld = (X/2)/SPIN_X_WORD;

	// length of a single color section per GPU
	size_t llenLoc = static_cast<size_t>(Y)*lld;
	size_t llen = 2ull*ndev*llenLoc;

	dim3 grid(DIV_UP(lld/2, BLOCK_X*BMULT_X),
		  DIV_UP(    Y, BLOCK_Y*BMULT_Y));

	dim3 block(BLOCK_X, BLOCK_Y);

	printf("Run configuration:\n");
	printf("\tspin/word: %d\n", SPIN_X_WORD);
	printf("\tspins: %zu\n", llen*SPIN_X_WORD);
	printf("\tseed: %llu\n", seed);
	printf("\titerations: %d\n", nsteps);
	printf("\tblock (X, Y): %d, %d\n", block.x, block.y);
	printf("\ttile  (X, Y): %d, %d\n", BLOCK_X*BMULT_X, BLOCK_Y*BMULT_Y);
	printf("\tgrid  (X, Y): %d, %d\n", grid.x, grid.y);

	if (printFreq) {
		printf("\tprint magn. every %d steps\n", printFreq);
	} else if (printExp) {
		printf("\tprint magn. following exponential series\n");
	} else {
		printf("\tprint magn. at 1st and last step\n");
	}
	if ((printFreq || printExp) && tgtMagn != -1.0) {
		printf("\tearly exit if magn. == %lf+-%lf\n", tgtMagn, TGT_MAGN_MAX_DIFF);
	}
	printf("\ttemp: %f (%f*T_crit)\n", temp, temp/CRIT_TEMP);
	if (!tempUpdFreq) {
		printf("\ttemp update not set\n");
	} else {
		printf("\ttemp update: %f / %d iterations\n", tempUpdStep, tempUpdFreq);
	}
	printf("\tlocal lattice size:      %7d x %7d\n",      Y, X);
	printf("\ttotal lattice size:      %7d x %7d\n", ndev*Y, X);
	printf("\tlocal lattice shape: 2 x %7d x %7zu (%12zu %s)\n",      Y, lld, llenLoc*2, sizeof(*v_d) == 4 ? "uints" : "ulls");
	printf("\ttotal lattice shape: 2 x %7d x %7zu (%12zu %s)\n", ndev*Y, lld,      llen, sizeof(*v_d) == 4 ? "uints" : "ulls");
	printf("\tmemory: %.2lf MB (%.2lf MB per GPU)\n", (llen*sizeof(*v_d))/(1024.0*1024.0), llenLoc*2*sizeof(*v_d)/(1024.0*1024.0));

	const int redBlocks = MIN(DIV_UP(llen, THREADS),
				  (props.maxThreadsPerMultiProcessor/THREADS)*props.multiProcessorCount);

	unsigned long long cntPos;
	unsigned long long cntNeg;
	unsigned long long *sum_d[MAX_GPU];

	if (ndev == 1) {
		CHECK_CUDA(cudaMalloc(&v_d, llen*sizeof(*v_d)));
		CHECK_CUDA(cudaMemset(v_d, 0, llen*sizeof(*v_d)));
		CHECK_CUDA(cudaMalloc(&sum_d[0], 2*sizeof(**sum_d)));
	} else {
		CHECK_CUDA(cudaMallocManaged(&v_d, llen*sizeof(*v_d), cudaMemAttachGlobal));
		printf("\nSetting up multi-gpu configuration:\n"); fflush(stdout);
		//#pragma omp parallel for schedule(static)
		for(int i = 0; i < ndev; i++) {

			CHECK_CUDA(cudaSetDevice(i));

			CHECK_CUDA(cudaMalloc(sum_d+i,     2*sizeof(**sum_d)));
        		CHECK_CUDA(cudaMemset(sum_d[i], 0, 2*sizeof(**sum_d)));

			// set preferred loc for black/white
			CHECK_CUDA(cudaMemAdvise(v_d +            i*llenLoc, llenLoc*sizeof(*v_d), cudaMemAdviseSetPreferredLocation, i));
			CHECK_CUDA(cudaMemAdvise(v_d + (llen/2) + i*llenLoc, llenLoc*sizeof(*v_d), cudaMemAdviseSetPreferredLocation, i));

			// black boundaries up/down
			//fprintf(stderr, "v_d + %12zu + %12zu, %12zu, ..., %2d)\n", i*llenLoc,  (Y-1)*lld, lld*sizeof(*v_d), (i+ndev+1)%ndev);
			CHECK_CUDA(cudaMemAdvise(v_d +            i*llenLoc,             lld*sizeof(*v_d), cudaMemAdviseSetAccessedBy, (i+ndev-1)%ndev));
			CHECK_CUDA(cudaMemAdvise(v_d +            i*llenLoc + (Y-1)*lld, lld*sizeof(*v_d), cudaMemAdviseSetAccessedBy, (i+ndev+1)%ndev));

			// white boundaries up/down
			CHECK_CUDA(cudaMemAdvise(v_d + (llen/2) + i*llenLoc,             lld*sizeof(*v_d), cudaMemAdviseSetAccessedBy, (i+ndev-1)%ndev));
			CHECK_CUDA(cudaMemAdvise(v_d + (llen/2) + i*llenLoc + (Y-1)*lld, lld*sizeof(*v_d), cudaMemAdviseSetAccessedBy, (i+ndev+1)%ndev));

			//CHECK_CUDA(cudaMemPrefetchAsync(v_d +            i*llenLoc, llenLoc*sizeof(*v_d), i, 0));
			//CHECK_CUDA(cudaMemPrefetchAsync(v_d + (llen/2) + i*llenLoc, llenLoc*sizeof(*v_d), i, 0));

			// reset black/white
			CHECK_CUDA(cudaMemset(v_d +            i*llenLoc, 0, llenLoc*sizeof(*v_d)));
			CHECK_CUDA(cudaMemset(v_d + (llen/2) + i*llenLoc, 0, llenLoc*sizeof(*v_d)));

			printf("\tGPU %2d done\n", i); fflush(stdout);
		}
	}

	black_d = v_d;
	white_d = v_d + llen/2;

	float *exp_d[MAX_GPU];
	float  exp_h[2][5];

	// precompute possible exponentials
	for(int i = 0; i < 2; i++) {
		for(int j = 0; j < 5; j++) {
			exp_h[i][j] = expf((i?-2.0f:2.0f)*static_cast<float>(j*2-4)*(1.0f/temp));
			//printf("exp[%2d][%d]: %E\n", i?1:-1, j, exp_h[i][j]);
		}
	}
	for(int i = 0; i < ndev; i++) {
		CHECK_CUDA(cudaSetDevice(i));
		CHECK_CUDA(cudaMalloc(exp_d+i, 2*5*sizeof(**exp_d)));
		CHECK_CUDA(cudaMemcpy(exp_d[i], exp_h, 2*5*sizeof(**exp_d), cudaMemcpyHostToDevice));
	}


	CHECK_CUDA(cudaEventCreate(&start));
	CHECK_CUDA(cudaEventCreate(&stop));

	for(int i = 0; i < ndev; i++) {
		CHECK_CUDA(cudaSetDevice(i));
		latticeInit_k<BLOCK_X, BLOCK_Y,
			      BMULT_X, BMULT_Y,
			      BIT_X_SPIN, C_BLACK,
			      unsigned long long><<<grid, block>>>(i,
								   seed,
								   0, i*Y, lld/2,
								   reinterpret_cast<ulonglong2 *>(black_d));
		CHECK_ERROR("initLattice_k");

		latticeInit_k<BLOCK_X, BLOCK_Y,
			      BMULT_X, BMULT_Y,
			      BIT_X_SPIN, C_WHITE,
			      unsigned long long><<<grid, block>>>(i,
								   seed,
								   0, i*Y, lld/2,
								   reinterpret_cast<ulonglong2 *>(white_d));
		CHECK_ERROR("initLattice_k");
	}

	countSpins(ndev, redBlocks, llen, llenLoc, black_d, white_d, sum_d, &cntPos, &cntNeg);
	printf("\nInitial magnetization: %9.6lf, up_s: %12llu, dw_s: %12llu\n",
	       abs(static_cast<double>(cntPos)-static_cast<double>(cntNeg)) / (llen*SPIN_X_WORD),
	       cntPos, cntNeg);

	for(int i = 0; i < ndev; i++) {
		CHECK_CUDA(cudaSetDevice(i));
		CHECK_CUDA(cudaDeviceSynchronize());
	}

	double __t0;
	if (ndev == 1) {
		CHECK_CUDA(cudaEventRecord(start, 0));
	} else {
		__t0 = Wtime();
	}
	int j = 0;
	for(j = 0; j < nsteps; j++) {
		for(int i = 0; i < ndev; i++) {
			CHECK_CUDA(cudaSetDevice(i));
			spinUpdateV_2D_k<BLOCK_X, BLOCK_Y,
					 BMULT_X, BMULT_Y,
					 BIT_X_SPIN, C_BLACK,
					 unsigned long long><<<grid, block>>>(i,
							 		      seed,
									      j+1, i*Y, ndev*Y, lld/2,
							 		      reinterpret_cast<float (*)[5]>(exp_d[i]),
									      reinterpret_cast<ulonglong2 *>(white_d),
									      reinterpret_cast<ulonglong2 *>(black_d));
		}
		if (ndev > 1) {
			for(int i = 0; i < ndev; i++) {
				CHECK_CUDA(cudaSetDevice(i));
				CHECK_CUDA(cudaDeviceSynchronize());
			}
		}
		for(int i = 0; i < ndev; i++) {
			CHECK_CUDA(cudaSetDevice(i));
			spinUpdateV_2D_k<BLOCK_X, BLOCK_Y,
					 BMULT_X, BMULT_Y,
					 BIT_X_SPIN, C_WHITE,
					 unsigned long long><<<grid, block>>>(i,
							 		      seed,
									      j+1, i*Y, ndev*Y, lld/2,
							 		      reinterpret_cast<float (*)[5]>(exp_d[i]),
									      reinterpret_cast<ulonglong2 *>(black_d),
									      reinterpret_cast<ulonglong2 *>(white_d));
		}
		if (ndev > 1) {
			for(int i = 0; i < ndev; i++) {
				CHECK_CUDA(cudaSetDevice(i));
				CHECK_CUDA(cudaDeviceSynchronize());
			}
		}
		if (printFreq && ((j+1) % printFreq) == 0) {
			countSpins(ndev, redBlocks, llen, llenLoc, black_d, white_d, sum_d, &cntPos, &cntNeg);
			const double magn = abs(static_cast<double>(cntPos)-static_cast<double>(cntNeg)) / (llen*SPIN_X_WORD);
			printf("        magnetization: %9.6lf, up_s: %12llu, dw_s: %12llu (iter: %8d)\n",
			       magn, cntPos, cntNeg, j+1);
			if (dumpOut) {
				char fname[256];
				snprintf(fname, sizeof(fname), "lattice_%dx%d_T_%f_IT_%08d_", Y, X, temp, j+1);
				dumpLattice(fname, ndev, Y, lld, llen, llenLoc, v_d);
			}
			if (abs(magn-tgtMagn) < TGT_MAGN_MAX_DIFF) {
				j++;
				break;
			}
		}
		//printf("j: %d, printExpSteps[%d]: %d\n", j, printExpCur, printExpSteps[printExpCur]);
		if (printExp && printExpSteps[printExpCur] == j) {
			printExpCur++;
			countSpins(ndev, redBlocks, llen, llenLoc, black_d, white_d, sum_d, &cntPos, &cntNeg);
			const double magn = abs(static_cast<double>(cntPos)-static_cast<double>(cntNeg)) / (llen*SPIN_X_WORD);
			printf("        magnetization: %9.6lf (^2: %9.6lf), up_s: %12llu, dw_s: %12llu (iter: %8d)\n",
			       magn, magn*magn, cntPos, cntNeg, j+1);
			if (dumpOut) {
				char fname[256];
				snprintf(fname, sizeof(fname), "lattice_%dx%d_T_%f_IT_%08d_", Y, X, temp, j+1);
				dumpLattice(fname, ndev, Y, lld, llen, llenLoc, v_d);
			}
			if (abs(magn-tgtMagn) < TGT_MAGN_MAX_DIFF) {
				j++;
				break;
			}
		}

		if (tempUpdFreq && ((j+1) % tempUpdFreq) == 0) {
			temp = MAX(MIN_TEMP, temp+tempUpdStep);
			printf("Changing temperature to %f\n", temp);
			for(int i = 0; i < 2; i++) {
				for(int k = 0; k < 5; k++) {
					exp_h[i][k] = expf((i?-2.0f:2.0f)*static_cast<float>(k*2-4)*(1.0f/temp));
					printf("exp[%2d][%d]: %E\n", i?1:-1, k, exp_h[i][k]);
				}
			}
			for(int i = 0; i < ndev; i++) {
				CHECK_CUDA(cudaMemcpy(exp_d[i], exp_h, 2*5*sizeof(**exp_d), cudaMemcpyHostToDevice));
			}
		}
	}
	if (ndev == 1) {
		CHECK_CUDA(cudaEventRecord(stop, 0));
		CHECK_CUDA(cudaEventSynchronize(stop));
	} else {
		for(int i = 0; i < ndev; i++) {
			CHECK_CUDA(cudaSetDevice(i));
			CHECK_CUDA(cudaDeviceSynchronize());
		}
		__t0 = Wtime()-__t0;
	}

	countSpins(ndev, redBlocks, llen, llenLoc, black_d, white_d, sum_d, &cntPos, &cntNeg);
	printf("Final   magnetization: %9.6lf, up_s: %12llu, dw_s: %12llu (iter: %8d)\n\n",
	       abs(static_cast<double>(cntPos)-static_cast<double>(cntNeg)) / (llen*SPIN_X_WORD),
	       cntPos, cntNeg, j);

	if (ndev == 1) {
		CHECK_CUDA(cudaEventElapsedTime(&et, start, stop));
	} else {
		et = __t0*1.0E+3;
	}

	printf("Kernel execution time for %d update steps: %E ms, %.2lf flips/ns (BW: %.2lf GB/s)\n",
		j, et, static_cast<double>(llen*SPIN_X_WORD)*j / (et*1.0E+6),
		(llen*sizeof(*v_d)*2*j/1.0E+9) / (et/1.0E+3));

	if (dumpOut) {
		char fname[256];
		snprintf(fname, sizeof(fname), "lattice_%dx%d_T_%f_IT_%08d_", Y, X, temp, j);
		dumpLattice(fname, ndev, Y, lld, llen, llenLoc, v_d);
	}

	CHECK_CUDA(cudaFree(v_d));
	if (ndev == 1) {
		CHECK_CUDA(cudaFree(exp_d[0]));
		CHECK_CUDA(cudaFree(sum_d[0]));
	} else {
		for(int i = 0; i < ndev; i++) {
			CHECK_CUDA(cudaSetDevice(i));
			CHECK_CUDA(cudaFree(exp_d[i]));
			CHECK_CUDA(cudaFree(sum_d[i]));
		}
	}
	for(int i = 0; i < ndev; i++) {
                CHECK_CUDA(cudaSetDevice(i));
                CHECK_CUDA(cudaDeviceReset());
        }
	return 0;
}

