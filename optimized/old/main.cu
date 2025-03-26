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
#include <unistd.h>
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
#define MIN_EXP_TIME (152)

#define MAX_CORR_LEN (128)

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
	 int LOOP_X, 
	 int LOOP_Y,
	 int BITXSP,
	 typename INT_T,
	 typename INT2_T>
__global__  void hamiltInitB_k(const int devid,
			       const float tgtProb,
			       const long long seed,
                               const long long begY,
                               const long long dimX, // ld
                                     INT2_T *__restrict__ hamB) {

	const int __i = blockIdx.y*BDIM_Y*LOOP_Y + threadIdx.y;
	const int __j = blockIdx.x*BDIM_X*LOOP_X + threadIdx.x;

	const long long tid = ((devid*gridDim.y + blockIdx.y)*gridDim.x + blockIdx.x)*BDIM_X*BDIM_Y +
	                       threadIdx.y*BDIM_X + threadIdx.x;

	curandStatePhilox4_32_10_t st;
	curand_init(seed, tid, 0, &st);

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
				#pragma unroll
				for(int l = 0; l < BITXSP; l++) {
					if (curand_uniform(&st) < tgtProb) {
						__tmp[i][j].x |= INT_T(1) << (k+l);
					}
					if (curand_uniform(&st) < tgtProb) {
						__tmp[i][j].y |= INT_T(1) << (k+l);
					}
				}
			}
		}
	}

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			hamB[(begY + __i+i*BDIM_Y)*dimX + __j+j*BDIM_X] = __tmp[i][j];
		}
	}
	return;
}

template<int BDIM_X,
	 int BDIM_Y,
	 int LOOP_X, 
	 int LOOP_Y,
	 int BITXSP,
	 typename INT_T,
	 typename INT2_T>
__global__ void hamiltInitW_k(const int xsl,
			      const int ysl,
			      const long long begY,
		              const long long dimX,
		              const INT2_T *__restrict__ hamB,
		                    INT2_T *__restrict__ hamW) {

	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;

	const int __i = blockIdx.y*BDIM_Y*LOOP_Y + tidy;
	const int __j = blockIdx.x*BDIM_X*LOOP_X + tidx;

	INT2_T __me[LOOP_Y][LOOP_X];

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__me[i][j] = hamB[(begY+__i+i*BDIM_Y)*dimX + __j+j*BDIM_X];
		}
	}

	INT2_T __up[LOOP_Y][LOOP_X];
	INT2_T __ct[LOOP_Y][LOOP_X];
	INT2_T __dw[LOOP_Y][LOOP_X];
	INT2_T __sd[LOOP_Y][LOOP_X];

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__up[i][j].x = (__me[i][j].x & 0x8888888888888888ull) >> 1; 
			__up[i][j].y = (__me[i][j].y & 0x8888888888888888ull) >> 1; 

			__dw[i][j].x = (__me[i][j].x & 0x4444444444444444ull) << 1; 
			__dw[i][j].y = (__me[i][j].y & 0x4444444444444444ull) << 1; 
		}
	}

	const int readBack = !(__i%2); // this kernel reads only BLACK Js

	const int BITXWORD = 8*sizeof(INT_T);

	if (!readBack) {
		#pragma unroll
		for(int i = 0; i < LOOP_Y; i++) {
			#pragma unroll
			for(int j = 0; j < LOOP_X; j++) {

				__ct[i][j].x = (__me[i][j].x & 0x2222222222222222ull) >> 1;
				__ct[i][j].y = (__me[i][j].y & 0x2222222222222222ull) >> 1;

				__ct[i][j].x |= (__me[i][j].x & 0x1111111111111111ull) << (BITXSP+1);
				__ct[i][j].y |= (__me[i][j].x & 0x1111111111111111ull) >> (BITXWORD-BITXSP - 1);
				__ct[i][j].y |= (__me[i][j].y & 0x1111111111111111ull) << (BITXSP+1);

				__sd[i][j].x = (__me[i][j].y & 0x1111111111111111ull) >> (BITXWORD-BITXSP - 1);
				__sd[i][j].y = 0;
			}
		}
	} else {
		#pragma unroll
		for(int i = 0; i < LOOP_Y; i++) {
			#pragma unroll
			for(int j = 0; j < LOOP_X; j++) {

				__ct[i][j].x = (__me[i][j].x & 0x1111111111111111ull) << 1;
				__ct[i][j].y = (__me[i][j].y & 0x1111111111111111ull) << 1;

				__ct[i][j].y |= (__me[i][j].y & 0x2222222222222222ull) >> (BITXSP+1);
				__ct[i][j].x |= (__me[i][j].y & 0x2222222222222222ull) << (BITXWORD-BITXSP - 1);
				__ct[i][j].x |= (__me[i][j].x & 0x2222222222222222ull) >> (BITXSP+1);

				__sd[i][j].y = (__me[i][j].x & 0x2222222222222222ull) << (BITXWORD-BITXSP - 1);
				__sd[i][j].x = 0;
			}
		}
	}

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {

		const int yoff = begY+__i + i*BDIM_Y;
			
		const int upOff = ( yoff   %ysl) == 0 ? yoff+ysl-1 : yoff-1;
		const int dwOff = ((yoff+1)%ysl) == 0 ? yoff-ysl+1 : yoff+1;

		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {

			const int xoff = __j + j*BDIM_X;

			atomicOr(&hamW[yoff*dimX + xoff].x, __ct[i][j].x);
			atomicOr(&hamW[yoff*dimX + xoff].y, __ct[i][j].y);
			
			atomicOr(&hamW[upOff*dimX + xoff].x, __up[i][j].x);
			atomicOr(&hamW[upOff*dimX + xoff].y, __up[i][j].y);

			atomicOr(&hamW[dwOff*dimX + xoff].x, __dw[i][j].x);
			atomicOr(&hamW[dwOff*dimX + xoff].y, __dw[i][j].y);

			const int sideOff = readBack ? (  (xoff   %xsl) == 0 ? xoff+xsl-1 : xoff-1 ):
						       ( ((xoff+1)%xsl) == 0 ? xoff-xsl+1 : xoff+1);

			atomicOr(&hamW[yoff*dimX + sideOff].x, __sd[i][j].x);
			atomicOr(&hamW[yoff*dimX + sideOff].y, __sd[i][j].y);
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
__device__ void loadTile(const int slX,
			 const int slY,
			 const long long begY,
			 const long long dimX,
			 const INT2_T *__restrict__ v,
			       INT2_T tile[][TILE_X+2*FRAME_X]) {

	const int blkx = blockIdx.x;
	const int blky = blockIdx.y;

	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;

	const int startX =        blkx*TILE_X;
	const int startY = begY + blky*TILE_Y;

	#pragma unroll
	for(int j = 0; j < TILE_Y; j += BDIM_Y) {
		int yoff = startY + j+tidy;

		#pragma unroll
		for(int i = 0; i < TILE_X; i += BDIM_X) {
			const int xoff = startX + i+tidx;
			tile[FRAME_Y + j+tidy][FRAME_X + i+tidx] = v[yoff*dimX + xoff];
		}
	}
	if (tidy == 0) {
		int yoff = (startY % slY) == 0 ? startY+slY-1 : startY-1;

		#pragma unroll
		for(int i = 0; i < TILE_X; i += BDIM_X) {
			const int xoff = startX + i+tidx;
			tile[0][FRAME_X + i+tidx] = v[yoff*dimX + xoff];
		}

		yoff = ((startY+TILE_Y) % slY) == 0 ? startY+TILE_Y - slY : startY+TILE_Y;

		#pragma unroll
		for(int i = 0; i < TILE_X; i += BDIM_X) {
			const int xoff = startX + i+tidx;
			tile[FRAME_Y + TILE_Y][FRAME_X + i+tidx] = v[yoff*dimX + xoff];
		}

		// the other branch in slower so skip it if possible
		if (BDIM_X <= TILE_Y) {

			int xoff = (startX % slX) == 0 ? startX+slX-1 : startX-1;

			#pragma unroll
			for(int j = 0; j < TILE_Y; j += BDIM_X) {
				yoff = startY + j+tidx;
				tile[FRAME_Y + j+tidx][0] = v[yoff*dimX + xoff];
			}

			xoff = ((startX+TILE_X) % slX) == 0 ? startX+TILE_X - slX : startX+TILE_X;

			#pragma unroll
			for(int j = 0; j < TILE_Y; j += BDIM_X) {
				yoff = startY + j+tidx;
				tile[FRAME_Y + j+tidx][FRAME_X + TILE_X] = v[yoff*dimX + xoff];
			}
		} else {
			if (tidx < TILE_Y) {
				int xoff = (startX % slX) == 0 ? startX+slX-1 : startX-1;

				yoff = startY + tidx;
				tile[FRAME_Y + tidx][0] = v[yoff*dimX + xoff];;

				xoff = ((startX+TILE_X) % slX) == 0 ? startX+TILE_X - slX : startX+TILE_X;
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
		      const int slX, // sublattice size X of one color (in words)
		      const int slY, // sublattice size Y of one color
		      const long long begY,
		      const long long dimX, // ld
		      const float vExp[][5],
		      const INT2_T *__restrict__ jDst,
		      const INT2_T *__restrict__ vSrc,
		            INT2_T *__restrict__ vDst) {

	const int SPIN_X_WORD = 8*sizeof(INT_T)/BITXSP;

	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;

	__shared__ INT2_T shTile[BDIM_Y*LOOP_Y+2][BDIM_X*LOOP_X+2];

	loadTile<BDIM_X, BDIM_Y,
		 BDIM_X*LOOP_X,
		 BDIM_Y*LOOP_Y, 
		 1, 1, INT2_T>(slX, slY, begY, dimX, vSrc, shTile);

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

	const int __i = blockIdx.y*BDIM_Y*LOOP_Y + tidy;
	const int __j = blockIdx.x*BDIM_X*LOOP_X + tidx;

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

	if (jDst != NULL) {
		INT2_T __J[LOOP_Y][LOOP_X];

		#pragma unroll
		for(int i = 0; i < LOOP_Y; i++) {
			#pragma unroll
			for(int j = 0; j < LOOP_X; j++) {
				__J[i][j] = jDst[(begY+__i+i*BDIM_Y)*dimX + __j+j*BDIM_X];
			}
		}

		// apply them
		// the 4 bits of J codify: <upJ, downJ, leftJ, rightJ>
		#pragma unroll
		for(int i = 0; i < LOOP_Y; i++) {
			#pragma unroll
			for(int j = 0; j < LOOP_X; j++) {

				__up[i][j].x ^= (__J[i][j].x & 0x8888888888888888ull) >> 3;
				__up[i][j].y ^= (__J[i][j].y & 0x8888888888888888ull) >> 3;

				__dw[i][j].x ^= (__J[i][j].x & 0x4444444444444444ull) >> 2;
				__dw[i][j].y ^= (__J[i][j].y & 0x4444444444444444ull) >> 2;

				if (readBack) {
					// __sd[][] holds "left" spins
					// __ct[][] holds "right" spins
					__sd[i][j].x ^= (__J[i][j].x & 0x2222222222222222ull) >> 1;
					__sd[i][j].y ^= (__J[i][j].y & 0x2222222222222222ull) >> 1;

					__ct[i][j].x ^= (__J[i][j].x & 0x1111111111111111ull);
					__ct[i][j].y ^= (__J[i][j].y & 0x1111111111111111ull);
				} else {
					// __ct[][] holds "left" spins
					// __sd[][] holds "right" spins
					__ct[i][j].x ^= (__J[i][j].x & 0x2222222222222222ull) >> 1;
					__ct[i][j].y ^= (__J[i][j].y & 0x2222222222222222ull) >> 1;

					__sd[i][j].x ^= (__J[i][j].x & 0x1111111111111111ull);
					__sd[i][j].y ^= (__J[i][j].y & 0x1111111111111111ull);
				}
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

	const int lid = threadIdx.x%WSIZE;
	const int wid = threadIdx.x/WSIZE;

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
                "Usage: %1$s [options]\n"
                "options:\n"
                "\t-x|--x <HORIZ_DIM>\n"
		"\t\tSpecifies the horizontal dimension of the entire  lattice  (black+white  spins),\n"
		"\t\tper GPU. This dimension must be a multiple of %2$d.\n"
                "\n"
                "\t-y|--y <VERT_DIM>\n"
		"\t\tSpecifies the vertical dimension of the entire lattice (black+white spins),  per\n"
		"\t\tGPU. This dimension must be a multiple of %3$d.\n"
                "\n"
                "\t-n|--n <NSTEPS>\n"
		"\t\tSpecifies the number of iteration to run.\n"
		"\t\tDefualt: %4$d\n"
                "\n"
                "\t-d|--devs <NUM_DEVICES>\n"
		"\t\tSpecifies the number of GPUs to use. Will use devices with ids [0, NUM_DEVS-1].\n"
		"\t\tDefualt: 1.\n"
                "\n"
                "\t-s|--seed <SEED>\n"
		"\t\tSpecifies the seed used to generate random numbers.\n"
		"\t\tDefault: %5$llu\n"
                "\n"
                "\t-a|--alpha <ALPHA>\n"
		"\t\tSpecifies the temperature in T_CRIT units.  If both this  option  and  '-t'  are\n"
		"\t\tspecified then the '-t' option is used.\n"
		"\t\tDefault: %6$f\n"
                "\n"
                "\t-t|--temp <TEMP>\n"
		"\t\tSpecifies the temperature in absolute units.  If both this option and  '-a'  are\n"
		"\t\tspecified then this option is used.\n"
		"\t\tDefault: %7$f\n"
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
                "\t-c|--corr\n"
		"\t\tDumps to a  file  named  corr_{X}x{Y}_T_{TEMP}  the  correlation  of each  point\n"
		"\t\twith the  %8$d points on the right and below.  The correlation is computed  every\n"
		"\t\ttime the magnetization is printed on screen (based on either the  '-p'  or  '-e'\n"
		"\t\toption) and it is written in the file one line per measure.\n"
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
		"\t-J|--J <PROB>\n"
		"\t\tSpecifies the probability [0.0-1.0] that links  connecting  any  two  spins  are\n"
		"\t\tanti-ferromagnetic. \n"
		"\t\tDefault: 0.0\n"
                "\n"
		"\t   --xsl <HORIZ_SUB_DIM>\n"
		"\t\tSpecifies the horizontal dimension of each sub-lattice (black+white spins),  per\n"
		"\t\tGPU.  This dimension must be a divisor of the horizontal dimension of the entire\n"
		"\t\tlattice per  GPU  (specified  with  the  '-x'  option) and a multiple of %2$d.\n"
		"\t\tDefault: sub-lattices are disabled.\n"
                "\n"
		"\t   --ysl <VERT_SUB_DIM>\n"
		"\t\tSpecifies the vertical  dimension of each  sub-lattice (black+white spins),  per\n"
		"\t\tGPU.  This dimension must be a divisor of the vertical dimension of  the  entire\n"
		"\t\tlattice per  GPU  (specified  with  the  '-y'  option) and a multiple of %3$d.\n"
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
		ALPHA_DEF*CRIT_TEMP,
		MAX_CORR_LEN);
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
		CHECK_CUDA(cudaMemset(sum_d[0], 0, 2*sizeof(**sum_d)));
		getMagn_k<THREADS, BIT_X_SPIN><<<redBlocks, THREADS>>>(llen, black_d, sum_d[0]);
		CHECK_ERROR("getMagn_k");
		CHECK_CUDA(cudaDeviceSynchronize());
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

template<int BDIM_X,
	 int BITXSP,
	 int N_CORR,
         typename INT_T,
	 typename SUM_T>
__global__ void getCorr2D_k(const int corrLen,
			    const long long dimX,
			    const long long dimY,
			    const long long begY,
			    const INT_T *__restrict__ black,
			    const INT_T *__restrict__ white,
				  SUM_T *__restrict__ corr) {

	const int SPIN_X_WORD = 8*sizeof(INT_T)/BITXSP;

	const int tid = threadIdx.x;

	const long long startY = begY + blockIdx.x;

	const int SH_LEN = BDIM_X + DIV_UP(N_CORR/2, SPIN_X_WORD);

	__shared__ INT_T __shB[SH_LEN];
	__shared__ INT_T __shW[SH_LEN];
	__shared__ SUM_T __shC[N_CORR];

	#pragma unroll
	for(int j = 0; j < N_CORR; j += BDIM_X) {
		if (j+tid < N_CORR) {
			__shC[j+tid] = 0;
		}
	}

	const int chunkDimX = 2*BDIM_X*SPIN_X_WORD;

	for(long long l = 0; l < dimX; l += BDIM_X) {

		__syncthreads();
		#pragma unroll
		for(int j = 0; j < SH_LEN; j += BDIM_X) {
			if (j+tid < SH_LEN) {
				const int off = (l+j+tid < dimX) ? l+j+tid : l+j+tid - dimX;
				__shB[j+tid] = black[startY*dimX + off];
				__shW[j+tid] = white[startY*dimX + off];
			}
		}
		__syncthreads();

		for(int j = 1; j <= corrLen; j++) {

			SUM_T myCorr = 0;

			for(long long i = tid; i < chunkDimX; i += BDIM_X) {

				// horiz corr
				const long long myWrdX = (i/2) / SPIN_X_WORD;
				const long long myOffX = (i/2) % SPIN_X_WORD;

				INT_T __tmp = ((startY ^ i) & 1) ? __shW[myWrdX] : __shB[myWrdX];
				const int mySpin = (__tmp >> (myOffX*BITXSP)) & 0xF;

				const long long nextX = i+j;

				const long long nextWrdX = (nextX/2) / SPIN_X_WORD;
				const long long nextOffX = (nextX/2) % SPIN_X_WORD;

				__tmp = ((startY ^ nextX) & 1) ? __shW[nextWrdX] : __shB[nextWrdX];
				const int nextSpin = (__tmp >> (nextOffX*BITXSP)) & 0xF;

				myCorr += (mySpin == nextSpin) ? SUM_T(1) : SUM_T(-1);

				// vert corr
				const long long nextY = (startY+j >= dimY) ? startY+j-dimY : startY+j;

				__tmp = ((nextY ^ i) & 1) ? white[nextY*dimX + l+myWrdX]:
							    black[nextY*dimX + l+myWrdX];
				const int vertSpin = (__tmp >> (myOffX*BITXSP)) & 0xF;

				myCorr += (mySpin == vertSpin) ? SUM_T(1) : SUM_T(-1);
			}

			myCorr = __block_sum<BDIM_X, 32>(myCorr);
			if (!tid) {
				__shC[j-1] += myCorr;
			}
		}
	}
	__syncthreads();

	#pragma unroll
	for(int j = 0; j < N_CORR; j += BDIM_X) {
		if (j+tid < N_CORR) {
			atomicAdd(corr + j+tid, __shC[j+tid]);
		}
	}
	return;
}

template<int BDIM_X,
	 int BITXSP,
	 int N_CORR,
         typename INT_T,
	 typename SUM_T>
__global__ void getCorr2DRepl_k(const int corrLen,
				const long long dimX,
				const long long begY,
			        const long long slX, // sublattice size X of one color (in words)
			        const long long slY, // sublattice size Y of one color 
				const INT_T *__restrict__ black,
				const INT_T *__restrict__ white,
				      SUM_T *__restrict__ corr) {
	const int tid = threadIdx.x;
	const int SPIN_X_WORD = 8*sizeof(INT_T)/BITXSP;

	const long long startY = begY + blockIdx.x;
	const long long mySLY = startY / slY;

	const long long NSLX = 2ull*dimX*SPIN_X_WORD / slX;

	const int SH_LEN = BDIM_X + DIV_UP(N_CORR/2, SPIN_X_WORD);

	__shared__ INT_T __shB[SH_LEN];
	__shared__ INT_T __shW[SH_LEN];

	__shared__ SUM_T __shC[N_CORR];

	#pragma unroll
	for(int j = 0; j < N_CORR; j += BDIM_X) {
		if (j+tid < N_CORR) {
			__shC[j+tid] = 0;
		}
	}

	const int chunkDimX = MIN(2*BDIM_X*SPIN_X_WORD, slX);

	const int slXLD = (slX/2) / SPIN_X_WORD;

	for(long long sl = 0; sl < NSLX; sl++) {

		for(long long l = 0; l < slXLD; l += BDIM_X) {

			__syncthreads();
			#pragma unroll
			for(int j = 0; j < SH_LEN; j += BDIM_X) {
				if (j+tid < SH_LEN) {
					const int off = (l+j+tid) % slXLD;
					__shB[j+tid] = black[startY*dimX + sl*slXLD + off];
					__shW[j+tid] = white[startY*dimX + sl*slXLD + off];
				}
			}
			__syncthreads();

			for(int j = 1; j <= corrLen; j++) {

				SUM_T myCorr = 0;

				for(long long i = tid; i < chunkDimX; i += BDIM_X) {

					// horiz corr
					const long long myWrdX = (i/2) / SPIN_X_WORD;
					const long long myOffX = (i/2) % SPIN_X_WORD;

					INT_T __tmp = ((startY ^ i) & 1) ? __shW[myWrdX] : __shB[myWrdX];
					const int mySpin = (__tmp >> (myOffX*BITXSP)) & 0xF;

					const long long nextX = i+j;

					const long long nextWrdX = (nextX/2) / SPIN_X_WORD;
					const long long nextOffX = (nextX/2) % SPIN_X_WORD;

					__tmp = ((startY ^ nextX) & 1) ? __shW[nextWrdX] : __shB[nextWrdX];
					const int nextSpin = (__tmp >> (nextOffX*BITXSP)) & 0xF;

					myCorr += (mySpin == nextSpin) ? SUM_T(1) : SUM_T(-1);

					// vert corr
					const long long nextY = (startY+j >= (mySLY+1)*slY) ? startY+j-slY : startY+j;

					__tmp = ((nextY ^ i) & 1) ? white[nextY*dimX + sl*slXLD + l+myWrdX]:
								    black[nextY*dimX + sl*slXLD + l+myWrdX];
					const int vertSpin = (__tmp >> (myOffX*BITXSP)) & 0xF;

					myCorr += (mySpin == vertSpin) ? SUM_T(1) : SUM_T(-1);
				}

				myCorr = __block_sum<BDIM_X, 32>(myCorr);
				if (!tid) {
					__shC[j-1] += myCorr;
				}
			}
		}
	}
	__syncthreads();

	#pragma unroll
	for(int j = 0; j < N_CORR; j += BDIM_X) {
		if (j+tid < N_CORR) {
			atomicAdd(corr + j+tid, __shC[j+tid]);
		}
	}
	return;
}

static void computeCorr(const char *fname,
			const int ndev,
			const int it,
			const int lld,
			const int useRepl,
			const int XSL,	// full sub-lattice (B+W) X
			const int YSL,	// full sub-lattice (B+W) X
			const int X,	// per-GPU full lattice (B+W) X
			const int Y,    // per-GPU full lattice (B+W) Y
		        const unsigned long long *black_d,
		        const unsigned long long *white_d,
			      double **corr_d,
			      double **corr_h) {

	const int n_corr = MAX_CORR_LEN;

	for(int i = 0; i < ndev; i++) {

		CHECK_CUDA(cudaSetDevice(i));
		CHECK_CUDA(cudaMemset(corr_d[i], 0, n_corr*sizeof(**corr_d)));

		if (!useRepl) {
			getCorr2D_k<THREADS, BIT_X_SPIN, MAX_CORR_LEN><<<Y, THREADS>>>(n_corr,
										       lld,
										       ndev*Y,
										       i*Y,
										       black_d,
										       white_d,
										       corr_d[i]);
			CHECK_ERROR("getCorr2D_k");
		} else {
			getCorr2DRepl_k<THREADS, BIT_X_SPIN, MAX_CORR_LEN><<<Y, THREADS>>>(n_corr,
											   lld,
											   i*Y,
											   XSL,
											   YSL,
											   black_d,
											   white_d,
											   corr_d[i]);
			CHECK_ERROR("getCorr2DRepl_k");
		}	
	}

	for(int i = 0; i < ndev; i++) {
		CHECK_CUDA(cudaSetDevice(i));
		CHECK_CUDA(cudaMemcpy(corr_h[i],
				      corr_d[i],
				      n_corr*sizeof(**corr_h),
				      cudaMemcpyDeviceToHost));
	}

	for(int d = 1; d < ndev; d++) {
		for(int i = 0; i < n_corr; i++) {
			corr_h[0][i] += corr_h[d][i];
		}
	}

	FILE *fp = Fopen(fname, "a");
	fprintf(fp,"%10d", it);
	for(int i = 0; i < n_corr; i++) {
		  fprintf(fp," % -12G", corr_h[0][i] / (2.0*X*Y*ndev));
	}
	fprintf(fp,"\n");
	fclose(fp);

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

	list_times[0]=MIN_EXP_TIME;

	unsigned long long t = 0;
	for(unsigned long long j = 0; j < nsteps && t < nsteps; j++) {
		t = rint(pow(2.0, j/4.0));
		if (t >= 2*list_times[nt] && nt < MAX_EXP_TIME-1) {
//		if (t > list_times[nt] && nt < MAX_EXP_TIME-1) {
			nt++;
			list_times[nt] = t;
			//printf("list_times[%d]: %llu\n", nt, list_times[nt]);
		}
	}
	return;
}

int main(int argc, char **argv) {

	unsigned long long *v_d=NULL;
	unsigned long long *black_d=NULL;
	unsigned long long *white_d=NULL;

	unsigned long long *ham_d=NULL;
	unsigned long long *hamB_d=NULL;
	unsigned long long *hamW_d=NULL;

	cudaEvent_t start, stop;
        float et;

	const int SPIN_X_WORD = (8*sizeof(*v_d)) / BIT_X_SPIN;

	int X = 0;
	int Y = 0;

	int dumpOut = 0;

        char cname[256];
        int corrOut = 0;

	double *corr_d[MAX_GPU];
	double *corr_h[MAX_GPU];

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

	int useGenHamilt = 0;
	float hamiltPerc1 = 0.0f;

	int useSubLatt = 0;
	int XSL = 0;
	int YSL = 0;
	int NSLX = 1;
	int NSLY = 1;

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
			{  "corr",       no_argument, 0, 'c'},
			{     "J", required_argument, 0, 'J'},
			{   "xsl", required_argument, 0,   1},
			{   "ysl", required_argument, 0,   2},
			{  "help", required_argument, 0, 'h'},
			{       0,                 0, 0,   0}
		};

		och = getopt_long(argc, argv, "x:y:n:ohs:d:a:t:p:u:m:ecJ:r:", long_options, &option_index);
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
				if(seed==0) {
					seed=((getpid()*rand())&0x7FFFFFFFF);
				}
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
			case 'c':
				corrOut = 1;
				break;
			case 'J':
				useGenHamilt = 1;
				hamiltPerc1 = atof(optarg);
				hamiltPerc1 = MIN(MAX(0.0f, hamiltPerc1), 1.0f);
				break;
			case 1:
				useSubLatt = 1;
				XSL = atoi(optarg);
				break;
			case 2:
				useSubLatt = 1;
				YSL = atoi(optarg);
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

	if (useSubLatt) {
		if (!XSL || !YSL) {
			if (!XSL) {
				if (YSL && !(YSL % (2*SPIN_X_WORD*2*BLOCK_X*BMULT_X))) {
					XSL = YSL;
				} else {
					XSL = 2*SPIN_X_WORD*2*BLOCK_X*BMULT_X;
				}
			}
			if (!YSL) {
				if (!(XSL%(BLOCK_Y*BMULT_Y))) {
					YSL = XSL;
				} else {
					YSL = BLOCK_Y*BMULT_Y;
				}
			}
		}
		if ((X%XSL) || !XSL || (XSL%2) || ((XSL/2)%(SPIN_X_WORD*2*BLOCK_X*BMULT_X))) {
			fprintf(stderr,
				"\nPlease specify an X sub-lattice dim multiple of %d and divisor of %d\n\n",
				2*SPIN_X_WORD*2*BLOCK_X*BMULT_X, X);
			usage(SPIN_X_WORD, argv[0]);
			exit(EXIT_FAILURE);
		}
		if ((Y%YSL) || !YSL || (YSL%(BLOCK_Y*BMULT_Y))) {
			fprintf(stderr,
				"\nPlease specify a Y sub-lattice dim multiple of %d divisor of %d\n\n",
				BLOCK_Y*BMULT_Y, Y);
			usage(SPIN_X_WORD, argv[0]);
			exit(EXIT_FAILURE);
		}

		NSLX = X / XSL;
		NSLY = Y / YSL;
	} else {
		XSL = X;
		YSL = Y*ndev;

		NSLX = 1;
		NSLY = 1;
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

	// total lattice length (all GPUs, all colors)
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
	if (useGenHamilt) {
		printf("\tusing Hamiltonian buffer, setting links to -1 with prob %G\n", hamiltPerc1);
	} else {
		printf("\tnot using Hamiltonian buffer\n");
	}

	printf("\n");
	if (useSubLatt) {
		printf("\tusing sub-lattices:\n");
		printf("\t\tno. of sub-lattices per GPU: %8d\n", NSLX*NSLY);
		printf("\t\tno. of sub-lattices (total): %8d\n", ndev*NSLX*NSLY);
		printf("\t\tsub-lattices size:           %7d x %7d\n\n", XSL, YSL);
	}
	printf("\tlocal lattice size:      %8d x %8d\n",      Y, X);
	printf("\ttotal lattice size:      %8d x %8d\n", ndev*Y, X);
	printf("\tlocal lattice shape: 2 x %8d x %8zu (%12zu %s)\n",      Y, lld, llenLoc*2, sizeof(*v_d) == 4 ? "uints" : "ulls");
	printf("\ttotal lattice shape: 2 x %8d x %8zu (%12zu %s)\n", ndev*Y, lld,      llen, sizeof(*v_d) == 4 ? "uints" : "ulls");
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

		if (useGenHamilt) {
			CHECK_CUDA(cudaMalloc(&ham_d, llen*sizeof(*ham_d)));
			CHECK_CUDA(cudaMemset(ham_d, 0, llen*sizeof(*ham_d)));
		}
	} else {
		CHECK_CUDA(cudaMallocManaged(&v_d, llen*sizeof(*v_d), cudaMemAttachGlobal));
		if (useGenHamilt) {
			CHECK_CUDA(cudaMallocManaged(&ham_d, llen*sizeof(*ham_d), cudaMemAttachGlobal));
		}
		printf("\nSetting up multi-gpu configuration:\n"); fflush(stdout);
		//#pragma omp parallel for schedule(static)
		for(int i = 0; i < ndev; i++) {

			CHECK_CUDA(cudaSetDevice(i));

			CHECK_CUDA(cudaMalloc(sum_d+i,     2*sizeof(**sum_d)));
        		CHECK_CUDA(cudaMemset(sum_d[i], 0, 2*sizeof(**sum_d)));

			// set preferred loc for black/white
			CHECK_CUDA(cudaMemAdvise(v_d +            i*llenLoc, llenLoc*sizeof(*v_d), cudaMemAdviseSetPreferredLocation, i));
			CHECK_CUDA(cudaMemAdvise(v_d + (llen/2) + i*llenLoc, llenLoc*sizeof(*v_d), cudaMemAdviseSetPreferredLocation, i));

			if (useGenHamilt) {
				CHECK_CUDA(cudaMemAdvise(ham_d +            i*llenLoc, llenLoc*sizeof(*ham_d), cudaMemAdviseSetPreferredLocation, i));
				CHECK_CUDA(cudaMemAdvise(ham_d + (llen/2) + i*llenLoc, llenLoc*sizeof(*ham_d), cudaMemAdviseSetPreferredLocation, i));
			}

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

			if (useGenHamilt) {
				CHECK_CUDA(cudaMemset(ham_d +            i*llenLoc, 0, llenLoc*sizeof(*ham_d)));
				CHECK_CUDA(cudaMemset(ham_d + (llen/2) + i*llenLoc, 0, llenLoc*sizeof(*ham_d)));
			}

			printf("\tGPU %2d done\n", i); fflush(stdout);
		}
	}

	if(corrOut) {
		snprintf(cname, sizeof(cname), "corr_%dx%d_T_%f_%llu", Y, X, temp, seed);
		Remove(cname);

		for(int i = 0; i < ndev; i++) {

			CHECK_CUDA(cudaSetDevice(i));

			corr_h[i] = (double *)Malloc(MAX_CORR_LEN*sizeof(**corr_h));
			CHECK_CUDA(cudaMalloc(corr_d+i, MAX_CORR_LEN*sizeof(**corr_d)));
		}
	}

	black_d = v_d;
	white_d = v_d + llen/2;
	if (useGenHamilt) {
		hamB_d = ham_d;
		hamW_d = ham_d + llen/2;
	}

	float *exp_d[MAX_GPU];
	float  exp_h[2][5];

	// precompute possible exponentials
	for(int i = 0; i < 2; i++) {
		for(int j = 0; j < 5; j++) {
			if(temp > 0) {
				exp_h[i][j] = expf((i?-2.0f:2.0f)*static_cast<float>(j*2-4)*(1.0f/temp));
			} else {
				if(j == 2) {
					exp_h[i][j] = 0.5f;
				} else {
					exp_h[i][j] = (i?-2.0f:2.0f)*static_cast<float>(j*2-4);
				}
			}
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

		if (useGenHamilt) {
			hamiltInitB_k<BLOCK_X, BLOCK_Y,
				      BMULT_X, BMULT_Y,
				      BIT_X_SPIN,
				      unsigned long long><<<grid, block>>>(i,
									   hamiltPerc1,
									   seed+1, // just use a different seed
									   i*Y, lld/2,
								   	   reinterpret_cast<ulonglong2 *>(hamB_d));
			hamiltInitW_k<BLOCK_X, BLOCK_Y,
				      BMULT_X, BMULT_Y,
				      BIT_X_SPIN,
				      unsigned long long><<<grid, block>>>((XSL/2)/SPIN_X_WORD/2, YSL, i*Y, lld/2,
								   	   reinterpret_cast<ulonglong2 *>(hamB_d),
								   	   reinterpret_cast<ulonglong2 *>(hamW_d));
		}
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
	int j;
	for(j = 0; j < nsteps; j++) {
		for(int i = 0; i < ndev; i++) {
			CHECK_CUDA(cudaSetDevice(i));
			spinUpdateV_2D_k<BLOCK_X, BLOCK_Y,
					 BMULT_X, BMULT_Y,
					 BIT_X_SPIN, C_BLACK,
					 unsigned long long><<<grid, block>>>(i,
							 		      seed,
									      j+1,
									      (XSL/2)/SPIN_X_WORD/2, YSL,
									      i*Y, /*ndev*Y,*/ lld/2,
							 		      reinterpret_cast<float (*)[5]>(exp_d[i]),
									      reinterpret_cast<ulonglong2 *>(hamW_d),
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
									      j+1,
									      (XSL/2)/SPIN_X_WORD/2, YSL,
									      i*Y, /*ndev*Y,*/ lld/2,
							 		      reinterpret_cast<float (*)[5]>(exp_d[i]),
									      reinterpret_cast<ulonglong2 *>(hamB_d),
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
			if (corrOut) {
				computeCorr(cname, ndev, j+1, lld, useSubLatt, XSL, YSL, X, Y, black_d, white_d, corr_d, corr_h);
			}
			if (dumpOut) {
				char fname[256];
				snprintf(fname, sizeof(fname), "lattice_%dx%d_T_%f_IT_%08d_", Y, X, temp, j+1);
				dumpLattice(fname, ndev, Y, lld, llen, llenLoc, v_d);
			}
			if (tgtMagn != -1.0) {
				if (abs(magn-tgtMagn) < TGT_MAGN_MAX_DIFF) {
					j++;
					break;
				}
			}
		}
		//printf("j: %d, printExpSteps[%d]: %d\n", j, printExpCur, printExpSteps[printExpCur]);
		if (printExp && printExpSteps[printExpCur] == j) {
			printExpCur++;
			countSpins(ndev, redBlocks, llen, llenLoc, black_d, white_d, sum_d, &cntPos, &cntNeg);
			const double magn = abs(static_cast<double>(cntPos)-static_cast<double>(cntNeg)) / (llen*SPIN_X_WORD);
			printf("        magnetization: %9.6lf (^2: %9.6lf), up_s: %12llu, dw_s: %12llu (iter: %8d)\n",
			       magn, magn*magn, cntPos, cntNeg, j+1);
			if (corrOut) {
				computeCorr(cname, ndev, j+1, lld, useSubLatt, XSL, YSL, X, Y, black_d, white_d, corr_d, corr_h);
			}
			if (dumpOut) {
				char fname[256];
				snprintf(fname, sizeof(fname), "lattice_%dx%d_T_%f_IT_%08d_", Y, X, temp, j+1);
				dumpLattice(fname, ndev, Y, lld, llen, llenLoc, v_d);
			}
			if (tgtMagn != -1.0) {
				if (abs(magn-tgtMagn) < TGT_MAGN_MAX_DIFF) {
					j++;
					break;
				}
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
		//(llen*sizeof(*v_d)*2*j/1.0E+9) / (et/1.0E+3));
		(2ull*j*
		 	( sizeof(*v_d)*((llen/2) + (llen/2) + (llen/2)) + // src color read, dst color read, dst color write
			  sizeof(*exp_d)*5*grid.x*grid.y ) /
		1.0E+9) / (et/1.0E+3));



	if (dumpOut) {
		char fname[256];
		snprintf(fname, sizeof(fname), "lattice_%dx%d_T_%f_IT_%08d_", Y, X, temp, j);
		dumpLattice(fname, ndev, Y, lld, llen, llenLoc, v_d);
	}

	CHECK_CUDA(cudaFree(v_d));
	if (useGenHamilt) {
		CHECK_CUDA(cudaFree(ham_d));
	}
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
	if (corrOut) {
		for(int i = 0; i < ndev; i++) {
			CHECK_CUDA(cudaSetDevice(i));
			CHECK_CUDA(cudaFree(corr_d[i]));
			free(corr_h[i]);
		}
	}
	for(int i = 0; i < ndev; i++) {
                CHECK_CUDA(cudaSetDevice(i));
                CHECK_CUDA(cudaDeviceReset());
        }
	return 0;
}

