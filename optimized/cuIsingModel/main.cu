/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
#include <assert.h>
#include <cuda.h>
//#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>

#define USE_MEMBRARIER

#ifdef USE_MNNVL
#include <mpi.h>
#include "vmm_alloc.h"
#endif

#include "cudamacro.h" /* for time() */
#include "utils.h"

#define DIV_UP(a,b)     (((a)+((b)-1))/(b))

#define THREADS  (64)

#define BIT_X_SPIN (1)
#define SPIN_MASK  ((1<<BIT_X_SPIN)-1)

// comment to use old kernel with one random number per spin
#define USE_UPDATE_LUT_K

#if defined(USE_UPDATE_LUT_K)

#define RND_TBL_T unsigned int

// log2(size) of the random-bits table used
// for binsearches only used in spinUpdate_fast_k:
// * 4: size = 16, one nibble (4 bits) per search;
// * 8: size = 256, one byte per search;
#define RNDBITS_TBL_LOG2_SIZE (4)

// 0: keep lut for binary searches in shared mem (necessary for RNDBITS_TBL_LOG2_SIZE==8)
// 1: keep lut for binary searches in registers (possible only for RNDBITS_TBL_LOG2_SIZE==4)
#define KERNEL_LUT_IN_REGS (1)

#if RNDBITS_TBL_LOG2_SIZE==8 && KERNEL_LUT_IN_REGS==1
#error "KERNEL_LUT_IN_REGS=1 is supported only for RNDBITS_TBL_LOG2_SIZE=4!\n"
#endif

#endif

#define WARP_SIZE (32)

#define CRIT_TEMP       (2.26918531421)
#define ALPHA_DEF       (0.1)

#define MIN(a,b)        (((a)<(b))?(a):(b))
#define MAX(a,b)        (((a)>(b))?(a):(b))

#define BLOCK_X (16)
#define BLOCK_Y (16)
#define BREAD_X (1)
#define BREAD_Y (1)

#define MAX_GPU   (256)

#define NUMIT_DEF (1)
#define SEED_DEF  (463463564571ull)

#define ONE_GB (1<<30)

#define __2POW32M1U  (-1u)
#define __2POW31M1U  ((1u<<31)-1)

#define CORR_ALLR_MAX (256)
#define CORR_EXPR_PP2  (32)

#define CORR_CHKB_SIDE (16)
#define CORR_MIXD_THRS (16)

#define MAX_TEMPS (10)

#define PRINTF1(...)  { if (!rank) { printf(__VA_ARGS__); fflush(stdout); } }
#define FPRINTF1(...) { if (!rank) { printf(__VA_ARGS__); } }

enum {C_BLACK, C_WHITE};
enum {S_NEG1, S_POS1};
enum {CORR_FULL, CORR_DIAG, CORR_CHKB, CORR_MIXD, CORR_NUM};

__device__ __forceinline__ unsigned int __mypopc(const unsigned int x) {
        return __popc(x);
}

__device__ __forceinline__ unsigned long long int __mypopc(const unsigned long long int x) {
        return __popcll(x);
}

__device__ __forceinline__ uint2 mymake_uint2(const unsigned int x,
                                              const unsigned int y) {
        return make_uint2(x, y);
}

__device__ __forceinline__ ulonglong2 mymake_uint2(const unsigned long long x,
                                                   const unsigned long long y) {
        return make_ulonglong2(x, y);
}

template<int BDIM_X,
         int BDIM_Y,
         int LOOP_X, 
         int LOOP_Y,
         int BITXSP,
         int COLOR,
         typename UINT2_T>
__global__  void spinInit_k(const int devid,
                            const unsigned long long seed,
                            const long long it,
                            const long long begY,
                            const long long dimX, // ld
                                  UINT2_T *dst) {

        static_assert(0 == (BDIM_X & (BDIM_X-1)));
        static_assert(0 == (BDIM_Y & (BDIM_Y-1)));
        static_assert(0 == (BITXSP & (BITXSP-1)));

        using UINT_T = decltype(dst[0].x);
	
        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;
        const int tid = tidy*BDIM_X + tidx;

	// kernel is launched with exaxtly:
	// * (dimX/(BDIM_X*LOOP_X)) * (dimY/((BDIM_Y*LOOP_Y)))
	// blocks
	const int bidx = blockIdx.x % (dimX / (BDIM_X*LOOP_X));
	const int bidy = blockIdx.x / (dimX / (BDIM_X*LOOP_X));

        const int __i = bidy*BDIM_Y*LOOP_Y + tidy;
        const int __j = bidx*BDIM_X*LOOP_X + tidx;

        const int SPIN_X_WORD = 8*sizeof(UINT_T)/BITXSP;

	const unsigned long long gid = ((unsigned long long)devid)*gridDim.x*BDIM_X*BDIM_Y + blockIdx.x*BDIM_X*BDIM_Y + tid;

        curandStateXORWOW_t st;

        const unsigned long long offset =  2ull*SPIN_X_WORD*LOOP_X*LOOP_Y*(2*it+COLOR);
        curand_init(seed, gid, offset, &st);

        UINT2_T __tmp[LOOP_Y][LOOP_X];
        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {
                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {
                        __tmp[i][j] = {0 , 0};
                }
        }

        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {
                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {
                        #pragma unroll
                        for(int k = 0; k < 8*sizeof(UINT_T); k += BITXSP) {
                                unsigned int r0 = curand(&st);
                                unsigned int r1 = curand(&st);

                                __tmp[i][j].x |= UINT_T(r0 >> 31) << k;
                                __tmp[i][j].y |= UINT_T(r1 >> 31) << k;
                        }
                }
        }

        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {
                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {
                        dst[(begY + __i+i*BDIM_Y)*dimX + __j+j*BDIM_X] = __tmp[i][j];
                }
        }
        return;
}

template<int BDIM_X,
         int BDIM_Y,
         int TILE_X,
         int TILE_Y,
         typename UINT2_T>
__device__ void loadTile_d(const int blkx,
			   const int blky,
			   const long long begY,
                           const long long dimY,
                           const long long dimX,
                           const UINT2_T *__restrict__ v,
                                 UINT2_T tile[][TILE_X+2]) {

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
                        tile[1 + j+tidy][1 + i+tidx] = v[yoff*dimX + xoff];
                }
        }
        if (tidy == 0) {
                //int yoff = (startY % slY) == 0 ? startY+slY-1 : startY-1;
                int yoff = startY ? startY-1 : dimY-1;

                #pragma unroll
                for(int i = 0; i < TILE_X; i += BDIM_X) {
                        const int xoff = startX + i+tidx;
                        tile[0][1 + i+tidx] = v[yoff*dimX + xoff];
                }

                //yoff = ((startY+TILE_Y) % slY) == 0 ? startY+TILE_Y - slY : startY+TILE_Y;
                yoff = (startY+TILE_Y == dimY) ? 0 : startY+TILE_Y;

                #pragma unroll
                for(int i = 0; i < TILE_X; i += BDIM_X) {
                        const int xoff = startX + i+tidx;
                        tile[1 + TILE_Y][1 + i+tidx] = v[yoff*dimX + xoff];
                }

                // the other branch in slower so skip it if possible
                if (BDIM_X <= TILE_Y) {

                        //int xoff = (startX % slX) == 0 ? startX+slX-1 : startX-1;
                        int xoff = startX ? startX-1 : dimX-1;

                        #pragma unroll
                        for(int j = 0; j < TILE_Y; j += BDIM_X) {
                                yoff = startY + j+tidx;
                                tile[1 + j+tidx][0] = v[yoff*dimX + xoff];
                        }

                        //xoff = ((startX+TILE_X) % slX) == 0 ? startX+TILE_X - slX : startX+TILE_X;
                        xoff = (startX+TILE_X == dimX) ? 0 : startX+TILE_X;

                        #pragma unroll
                        for(int j = 0; j < TILE_Y; j += BDIM_X) {
                                yoff = startY + j+tidx;
                                tile[1 + j+tidx][1 + TILE_X] = v[yoff*dimX + xoff];
                        }
                } else {
                        if (tidx < TILE_Y) {
                                //int xoff = (startX % slX) == 0 ? startX+slX-1 : startX-1;
                                int xoff = startX ? startX-1 : dimX-1;

                                yoff = startY + tidx;
                                tile[1 + tidx][0] = v[yoff*dimX + xoff];

                                //xoff = ((startX+TILE_X) % slX) == 0 ? startX+TILE_X - slX : startX+TILE_X;
                                xoff = (startX+TILE_X == dimX) ? 0 : startX+TILE_X;
                                tile[1 + tidx][1 + TILE_X] = v[yoff*dimX + xoff];
                        }
                }
        }
        return;
}

#define TOT_REGS (65536)
#define MAX_RXTH (48)
#if 0
// after testing on H100, 48 registers/th appears to be the 
// sweet spot to maximize occupancy while avoiding spilling;
// removing the second __launch_bounds__ parameter does causes
// the kernel instance with COLOR==BLACK to be compiled with
// 56 reg/th (instead of 48), lowering the theoretical occ. 
// from 62% to 50% (with a minor but measurable drop in perf)
template<int BDIM_X,
         int BDIM_Y,
         int LOOP_X, 
         int LOOP_Y,
         int BITXSP,
         int COLOR,
         typename UINT2_T>
__global__ 
__launch_bounds__(BDIM_X*BDIM_Y, TOT_REGS / (BDIM_X*BDIM_Y*MAX_RXTH))
void spinUpdate_k(const int devid,
                  const long long seed,
                  const long long it,
                  const long long begY,
                  const long long dimY,
                  const long long dimX, // ld 
                  const unsigned int *exp, //[2][5],
                  const UINT2_T *__restrict__ src,
                        UINT2_T *__restrict__ dst) {

        static_assert(0 == (BDIM_X & (BDIM_X-1)));
        static_assert(0 == (BDIM_Y & (BDIM_Y-1)));
        static_assert(0 == (BITXSP & (BITXSP-1)));

        using UINT_T = decltype(src[0].x);

        constexpr int BITXWD = 8*sizeof(UINT_T);
        constexpr int SPIN_X_WORD = BITXWD/BITXSP;

        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;

	// kernel il launched with exaxtly:
	// * (dimX/(BDIM_X*LOOP_X)) * (dimY/((BDIM_Y*LOOP_Y)))
	// blocks
	const int bidx = blockIdx.x % (dimX / (BDIM_X*LOOP_X));
	const int bidy = blockIdx.x / (dimX / (BDIM_X*LOOP_X));

        __shared__ UINT2_T shTile[BDIM_Y*LOOP_Y+2][BDIM_X*LOOP_X+2];

        loadTile_d<BDIM_X, BDIM_Y, BDIM_X*LOOP_X, BDIM_Y*LOOP_Y>(bidx, bidy, begY, dimY, dimX, src, shTile);

        __shared__ unsigned int shExp[2][5];

        for(int i = tidy*BDIM_X + tidx;
            i < sizeof(shExp)/sizeof(shExp[0][0]);
            i += BDIM_X*BDIM_Y) {

                reinterpret_cast<unsigned int *>(shExp)[i] = exp[i];
        }
        __syncthreads();

        //const long long tid = ((devid*gridDim.y + blockIdx.y)*gridDim.x + blockIdx.x)*BDIM_X*BDIM_Y +
        //                       tidy*BDIM_X + tidx;
	const unsigned long long gid = ((unsigned long long)devid)*gridDim.x*BDIM_X*BDIM_Y + 
				       blockIdx.x*BDIM_X*BDIM_Y + tidy*BDIM_X + tidx;

        UINT2_T me[LOOP_Y][LOOP_X];

        const int __i = bidy*BDIM_Y*LOOP_Y + tidy;
        const int __j = bidx*BDIM_X*LOOP_X + tidx;

        dst += (begY+__i)*dimX + __j;

        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {
                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {
                        me[i][j] = dst[i*BDIM_Y*dimX + j*BDIM_X];
                }
        }

        UINT2_T up[LOOP_Y][LOOP_X];
        UINT2_T ct[LOOP_Y][LOOP_X];
        UINT2_T dw[LOOP_Y][LOOP_X];
        UINT2_T sd[LOOP_Y][LOOP_X];

        // BDIM_Y is power of two so row parity won't change across loops
        const int readBack = (COLOR == C_BLACK) ? !(tidy%2) : (tidy%2);

        UINT2_T (*shLoad)[BDIM_X*LOOP_X+2] = reinterpret_cast<UINT2_T (*)[BDIM_X*LOOP_X+2]>(&shTile[tidy][tidx]);

        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {
                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {
#if 0
                        up[i][j] = shTile[i*BDIM_Y +   tidy][j*BDIM_X + 1+tidx];
                        ct[i][j] = shTile[i*BDIM_Y + 1+tidy][j*BDIM_X + 1+tidx];
                        sd[i][j] = shTile[i*BDIM_Y + 1+tidy][j*BDIM_X + 2*(!readBack)+tidx];
                        dw[i][j] = shTile[i*BDIM_Y + 2+tidy][j*BDIM_X + 1+tidx];
#else
                        up[i][j] = shLoad[i*BDIM_Y    ][j*BDIM_X + 1];
                        ct[i][j] = shLoad[i*BDIM_Y + 1][j*BDIM_X + 1];
                        dw[i][j] = shLoad[i*BDIM_Y + 2][j*BDIM_X + 1];
                        sd[i][j] = readBack ? shLoad[i*BDIM_Y + 1][j*BDIM_X + 0]:
                                              shLoad[i*BDIM_Y + 1][j*BDIM_X + 2];
#endif
                }
        }

        if (readBack) {
                #pragma unroll
                for(int i = 0; i < LOOP_Y; i++) {
                        #pragma unroll
                        for(int j = 0; j < LOOP_X; j++) {
                                sd[i][j].x = (sd[i][j].y >> (BITXWD-BITXSP)) | (ct[i][j].x << BITXSP);
                                sd[i][j].y = (ct[i][j].x >> (BITXWD-BITXSP)) | (ct[i][j].y << BITXSP);
                        }
                }
        } else {
                #pragma unroll
                for(int i = 0; i < LOOP_Y; i++) {
                        #pragma unroll
                        for(int j = 0; j < LOOP_X; j++) {
                                sd[i][j].y = (ct[i][j].y >> BITXSP) | (sd[i][j].x << (BITXWD-BITXSP));
                                sd[i][j].x = (ct[i][j].x >> BITXSP) | (ct[i][j].y << (BITXWD-BITXSP));
                        }
                }
        }

        if constexpr(BITXSP == 4) {
                #pragma unroll
                for(int i = 0; i < LOOP_Y; i++) {
                        #pragma unroll
                        for(int j = 0; j < LOOP_X; j++) {
                                ct[i][j].x += up[i][j].x + dw[i][j].x + sd[i][j].x;
                                ct[i][j].y += up[i][j].y + dw[i][j].y + sd[i][j].y;
                        }
                }
        } else if constexpr(BITXSP == 2) {
                #pragma unroll
                for(int i = 0; i < LOOP_Y; i++) {
                        #pragma unroll
                        for(int j = 0; j < LOOP_X; j++) {
                                ct[i][j].x += up[i][j].x;
                                dw[i][j].x += sd[i][j].x;

                                ct[i][j].y += up[i][j].y;
                                dw[i][j].y += sd[i][j].y;
                        }
                }
        } // BITXSP == 1 do nothing

        curandStatePhilox4_32_10_t st;
        curand_init(seed, gid, static_cast<long long>(2*SPIN_X_WORD)*LOOP_X*LOOP_Y*(2*it+COLOR), &st);

        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {
                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {

                        UINT2_T flip = {0, 0};

                        #pragma unroll
                        for(int z = 0; z < BITXWD; z += BITXSP) {

                                const uint2 src = make_uint2((me[i][j].x >> z) & SPIN_MASK,
                                                             (me[i][j].y >> z) & SPIN_MASK);

                                uint2 sum;
                                if constexpr(BITXSP == 4) {
                                        sum = make_uint2((ct[i][j].x >> z) & SPIN_MASK, 
                                                         (ct[i][j].y >> z) & SPIN_MASK);
                                }
                                if constexpr(BITXSP == 2) {
                                        sum = make_uint2(((ct[i][j].x >> z) & SPIN_MASK)+
                                                         ((dw[i][j].x >> z) & SPIN_MASK),

                                                         ((ct[i][j].y >> z) & SPIN_MASK)+
                                                         ((dw[i][j].y >> z) & SPIN_MASK));
                                }
                                if constexpr(BITXSP == 1) {
                                        sum = make_uint2(((ct[i][j].x >> z) & SPIN_MASK) +
                                                         ((up[i][j].x >> z) & SPIN_MASK) + 
                                                         ((dw[i][j].x >> z) & SPIN_MASK) +
                                                         ((sd[i][j].x >> z) & SPIN_MASK),

                                                         ((ct[i][j].y >> z) & SPIN_MASK) +
                                                         ((up[i][j].y >> z) & SPIN_MASK) + 
                                                         ((dw[i][j].y >> z) & SPIN_MASK) +
                                                         ((sd[i][j].y >> z) & SPIN_MASK)); 
                                }
#if 0
                                if (src.x > 2 || sum.x > 8) { printf("%s:%d: error: src.x > 2 (=%u) and/or sum.x > 8 (=%u)\n", __func__, __LINE__, src.x, sum.x); }
                                if (src.y > 2 || sum.y > 8) { printf("%s:%d: error: src.y > 2 (=%u) and/or sum.y > 8 (=%u)\n", __func__, __LINE__, src.y, sum.y); }
#endif
                                const uint2 sexp = {shExp[src.x][sum.x],
                                                    shExp[src.y][sum.y]};

                                unsigned int rnd0 = curand(&st);
                                unsigned int rnd1 = curand(&st);

                                flip.x |= ((rnd0 <= sexp.x)*1ull) << z; 
                                flip.y |= ((rnd1 <= sexp.y)*1ull) << z; 
                        }

                        me[i][j].x ^= flip.x;
                        me[i][j].y ^= flip.y;
#if 0
                        #pragma unroll
                        for(int z = 0; z < BITXWD; z += BITXSP) {
                                const uint2 src = make_uint2((me[i][j].x >> z) & SPIN_MASK,
                                                             (me[i][j].y >> z) & SPIN_MASK);
                                if (src.x > 2) { printf("%s:%d: error: found a spin > 2 (=%u)\n", __func__, __LINE__, src.x); }
                                if (src.y > 2) { printf("%s:%d: error: found a spin > 2 (=%u)\n", __func__, __LINE__, src.y); }
                        }
#endif
                }
        }

        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {
                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {
                        dst[i*BDIM_Y*dimX + j*BDIM_X] = me[i][j];
                }
        }

        return;
}
#endif


// this struct uses shared memory for the LUT
// so there are no restriction on TBL_LOG2_LEN;
// can be either 4 or 8
template<int BDIM_X, // using blockDim.xy in the methods 
	 int BDIM_Y, // causes an 18% drop in perf!!!
	 int TBL_LOG2_LEN,
         typename TABLE_T>
struct LUT_SHARED_T {
        
	static constexpr int TABLE_LEN = 1 << TBL_LOG2_LEN;

        // best on RTX 6000 Ada with sizeof(TABLE_T)==4:  8/3 - tid%TABLE_NREP -> ~1480 flip/ns
        // best on RTX 6000 Ada with sizeof(TABLE_T)==8:  8/3 - tid%TABLE_NREP -> ~1040 flip/ns
        // best on H100         with sizeof(TABLE_T)==4:  8/3 - tid%TABLE_NREP -> ~1432 flip/ns
        // best on H100         with sizeof(TABLE_T)==8: 16/3 - tid%TABLE_NREP -> ~1029 flip/ns
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
        static constexpr int TABLE_NREP = (TBL_LOG2_LEN > 4) ?  8 : 1;
#else
        static constexpr int TABLE_NREP = (TBL_LOG2_LEN > 4) ? 16 : 1;
#endif
        static constexpr int TABLE_PAD = (TBL_LOG2_LEN > 4) ? 3 : 0;
        
	TABLE_T *myTbl;

	__device__ void init(const TABLE_T *bitTbl, TABLE_T sh[][TABLE_LEN+TABLE_PAD]) {

		const int tid = threadIdx.y*BDIM_X + threadIdx.x;

		#pragma unroll
		for(int i = 0; i < TABLE_LEN; i += BDIM_X*BDIM_Y) {
			if (i+tid >= TABLE_LEN) {
				break;
			}
			TABLE_T v = bitTbl[i+tid];
			#pragma unroll
			for(int j = 0; j < TABLE_NREP; j++) {
				sh[j][i+tid] = v;
			}
		}
		__syncthreads();

		myTbl = &sh[(tid%TABLE_NREP)][0];
		return;
	}
#if 0
	// debug version with linear search
	__device__  unsigned int getMaxLE(const TABLE_T val) {

		unsigned int i;
		for(i = 0; i < TABLE_LEN; i++) {
			if (myTbl[i] > val) break;
		}
		return i-1;
	}
#endif
#if 0
	// original version with while()
	__device__ unsigned int getMaxLE(const TABLE_T val) {

		unsigned int N = TABLE_LEN;

		unsigned int _min = 0;
		unsigned int _max = N-1;
		unsigned int _mid = N >> 1;

		while(_min <= _max) {

			const TABLE_T vm = myTbl[_mid];
			if (vm == val) break;
			if (vm  < val) _min = _mid+1;
			else           _max = _mid-1;
			_mid = (_max + _min) / 2; // no need to take into account overflow here
		}
	#if 0
		if (_mid                    >=   N) { printf("%s:%d: error, returning index %u >= %u\n", __func__, __LINE__, _mid, N); }   
		if (_mid >   0 && v[_mid-1] == val) { printf("%s:%d: error, returning index %u but v[%u]=%u == val=%u\n", __func__, __LINE__, _mid, _mid-1, v[_mid-1], val); }
		if (              v[_mid]    > val) { printf("%s:%d: error, returning index %u but v[%u]=%u >  val=%u\n", __func__, __LINE__, _mid, _mid,   v[_mid],   val); }
		if (_mid < N-1 && v[_mid+1] <= val) { printf("%s:%d: error, returning index %u but v[%u]=%u <= val=%u\n", __func__, __LINE__, _mid, _mid+1, v[_mid+1], val); }
	#endif
		return _mid;
	}
#endif
#if 0
	// original optimized version with lots of SELs
	__device__ unsigned int getMaxLE(const TABLE_T val) {

		unsigned int _min = 0;
		unsigned int _max = TABLE_LEN;

		// better to always do 8 iterations
		// known at compile time
		#pragma unroll
		for(int i = 0; i < TBL_LOG2_LEN; i++) {
			// no need to take into account overflow here
			const unsigned int _mid = (_max + _min) >> 1;

			const unsigned int lt  = val < myTbl[_mid];

			_min = lt ? _min : _mid;
			_max = lt ? _mid : _max;
		}

		return _min;
	}
#endif
#if 1
	__device__ unsigned int getMaxLE(const TABLE_T val) {

		unsigned int mid = 0;

		// better to always do 8 iterations known at compile time
		#pragma unroll
		for(unsigned int i = 1u << (TBL_LOG2_LEN-1); i; i >>= 1) {
			const unsigned int ltmsk = (val < myTbl[mid|i])-1;
			mid |= i & ltmsk;
		}
		return mid;
	}

#endif
};

// this struct uses shuffles for the LUT
// so TBL_LOG2_LEN must be equal to 2 
template<int TBL_LOG2_LEN,
         typename TABLE_T>
struct LUT_SHUFFLE_T {

	static_assert(TBL_LOG2_LEN == 4);

	static constexpr int TABLE_LEN = 1 << TBL_LOG2_LEN;
	TABLE_T myTblVal = 0;

	__device__ LUT_SHUFFLE_T(const TABLE_T *bitTbl) {
		
		const int tid = threadIdx.y*blockDim.x + threadIdx.x;

		myTblVal = bitTbl[tid % TABLE_LEN];
		return;
	}

	__device__ unsigned int getMaxLE(const TABLE_T val) {

		unsigned int mid = 0;

		#pragma unroll
		for(unsigned int i = 1u << (TBL_LOG2_LEN-1); i; i >>= 1) {

			const unsigned int cmp = __shfl_sync(0xFFFFFFFF,
							     myTblVal,
							     mid|i,
							     TABLE_LEN); // == 16
			const unsigned int ltmsk = (val < cmp) - 1;
			mid |= i & ltmsk;
		}
		return mid;
	}
};

template<int BDIM_X,
         int BDIM_Y,
         int LOOP_X,
         int LOOP_Y,
         int COLOR,
         typename UINT2_T>
__device__ void loadTileReg_d(const int bidx,
			      const int bidy,
			      const long long begY,
                              const long long dimY,
                              const long long dimX,
                              const UINT2_T *__restrict__ v,
                                    UINT2_T (&ct)[LOOP_Y][LOOP_X],
                                    UINT2_T (&up)[LOOP_Y][LOOP_X],
                                    UINT2_T (&dw)[LOOP_Y][LOOP_X],
                                    UINT2_T (&sd)[LOOP_Y][LOOP_X]) {

        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;
        
	const int startX =        bidx*LOOP_X*BDIM_X + tidx;
        const int startY = begY + bidy*LOOP_Y*BDIM_Y + tidy;
        
	const int readBack = (COLOR == C_BLACK) ? !(tidy%2) : (tidy%2);
#if 0
	const int dimYm1 = dimY-1;
	const int dimXm1 = dimX-1;

        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {

                const int yoffct = startY + i*BDIM_Y;

                const int yoffup = (startY-1 + i*BDIM_Y)&dimYm1;
                const int yoffdw = (startY+1 + i*BDIM_Y)&dimYm1;

                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {

                        const int xoffct = startX + j*BDIM_X;

                        const int xoffsd = readBack ? (startX-1 + j*BDIM_X)&dimXm1:
                                                      (startX+1 + j*BDIM_X)&dimXm1;

                        up[i][j] = v[yoffup*dimX + xoffct];
                        sd[i][j] = v[yoffct*dimX + xoffsd];
                        ct[i][j] = v[yoffct*dimX + xoffct];
                        dw[i][j] = v[yoffdw*dimX + xoffct];
                }
        }

#else
#if 1
        v += startY*dimX + startX;

        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {

                const int yoffct = i*BDIM_Y;

                const int yoffup = (startY-1 + i*BDIM_Y) <     0 ? dimY-startY-1 : i*BDIM_Y-1;
                const int yoffdw = (startY+1 + i*BDIM_Y) >= dimY ?     -startY   : i*BDIM_Y+1;

                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {

                        const int xoffct = j*BDIM_X;

                        const int xoffsd = readBack ? ((startX-1 + j*BDIM_X) <     0 ? dimX-startX-1 : j*BDIM_X-1):
                                                      ((startX+1 + j*BDIM_X) >= dimX ?     -startX   : j*BDIM_X+1);

                        up[i][j] = v[yoffup*dimX + xoffct];
                        sd[i][j] = v[yoffct*dimX + xoffsd];
                        ct[i][j] = v[yoffct*dimX + xoffct];
                        dw[i][j] = v[yoffdw*dimX + xoffct];
                }
        }
#else
        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {

                const int yoff = (startY-1 + i*BDIM_Y) < 0 ? dimY-1 : (startY-1 + i*BDIM_Y);

                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {
                        const int xoff = startX + j*BDIM_X;

                        up[i][j] = v[yoff*dimX + xoff];
                }
        }

	#pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {

                const int yoff = startY + i*BDIM_Y;

                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {

                        const int xoff = readBack ? ((startX-1 + j*BDIM_X) <     0 ? dimX-1 : startX-1 + j*BDIM_X):
                                                    ((startX+1 + j*BDIM_X) >= dimX ?      0 : startX+1 + j*BDIM_X);

                        sd[i][j] = v[yoff*dimX + xoff];
                }
        }

        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {

                const int yoff = startY + i*BDIM_Y;

                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {
                        const int xoff = startX + j*BDIM_X;

                        ct[i][j] = v[yoff*dimX + xoff];
                }
        }

        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {

                const int yoff = (startY+1 + i*BDIM_Y) >= dimY ? 0 : (startY+1 + i*BDIM_Y);

                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {
                        const int xoff = startX + j*BDIM_X;

                        dw[i][j] = v[yoff*dimX + xoff];
                }
        }
#endif
#endif
        return;
} 
	
template<typename UINT_T>
__device__ void __swp(UINT_T &a, UINT_T &b) {

	const UINT_T x = a;
	a &= b;
	b |= x;

	return;
}

template<int BDIM_X,
         int BDIM_Y,
         int LOOP_X,
         int LOOP_Y,
         typename UINT2_T>
__device__ void sortBits_d(UINT2_T (&up)[LOOP_Y][LOOP_X],
                           UINT2_T (&dw)[LOOP_Y][LOOP_X],
                           UINT2_T (&sd)[LOOP_Y][LOOP_X],
                           UINT2_T (&ct)[LOOP_Y][LOOP_X]) {
        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {
                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {
#if 1
                        __swp(up[i][j].x, dw[i][j].x);
                        __swp(dw[i][j].x, sd[i][j].x);
                        __swp(sd[i][j].x, ct[i][j].x);
                        __swp(up[i][j].x, dw[i][j].x);
                        __swp(dw[i][j].x, sd[i][j].x);
                        __swp(up[i][j].x, dw[i][j].x);

                        __swp(up[i][j].y, dw[i][j].y);
                        __swp(dw[i][j].y, sd[i][j].y);
                        __swp(sd[i][j].y, ct[i][j].y);
                        __swp(up[i][j].y, dw[i][j].y);
                        __swp(dw[i][j].y, sd[i][j].y);
                        __swp(up[i][j].y, dw[i][j].y);
#else
			// shorter sorting networks
#if 1
                        __swp(sd[i][j].x, ct[i][j].x);
                        __swp(dw[i][j].x, sd[i][j].x);
                        __swp(up[i][j].x, dw[i][j].x);
                        __swp(sd[i][j].x, ct[i][j].x);
                        __swp(dw[i][j].x, sd[i][j].x);

                        __swp(sd[i][j].y, ct[i][j].y);
                        __swp(dw[i][j].y, sd[i][j].y);
                        __swp(up[i][j].y, dw[i][j].y);
                        __swp(sd[i][j].y, ct[i][j].y);
                        __swp(dw[i][j].y, sd[i][j].y);
#else
                        __swp(up[i][j].x, sd[i][j].x);
                        __swp(dw[i][j].x, ct[i][j].x);
                        __swp(up[i][j].x, dw[i][j].x);
                        __swp(sd[i][j].x, ct[i][j].x);
                        __swp(dw[i][j].x, sd[i][j].x);
                        
                        __swp(up[i][j].y, sd[i][j].y);
                        __swp(dw[i][j].y, ct[i][j].y);
                        __swp(up[i][j].y, dw[i][j].y);
                        __swp(sd[i][j].y, ct[i][j].y);
                        __swp(dw[i][j].y, sd[i][j].y);
#endif
#endif
		}
	}
	return;
}

template<int BDIM_X,
         int BDIM_Y,
         int LOOP_X, 
         int LOOP_Y,
         int BITXSP,
         int TBL_LOG2_LEN, // log2(len) of table bitTbl[]
         int COLOR,
         typename TABLE_T,
         typename UINT2_T>
__global__ 
__launch_bounds__(BDIM_X*BDIM_Y)
//__launch_bounds__(BDIM_X*BDIM_Y, TOT_REGS / (BDIM_X*BDIM_Y*MAX_RXTH))
void spinUpdateLUT_k(const int devid,
                     const unsigned long long seed,
                     const long long it,
                     const long long begY,
                     const long long dimY,
                     const long long dimX,  // ld 
                     const TABLE_T *bitTbl, // [16] or [256],
                     const UINT2_T *__restrict__ src,
                           UINT2_T *__restrict__ dst) {

        static_assert(0 == (BDIM_X & (BDIM_X-1)));
        static_assert(0 == (BDIM_Y & (BDIM_Y-1)));
        static_assert(0 == (BITXSP & (BITXSP-1)));

        using UINT_T = decltype(src[0].x);

        constexpr int BITXWD = 8*sizeof(UINT_T);
        //constexpr int SPIN_X_WORD = BITXWD/BITXSP;

        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;
        const int tid = tidy*BDIM_X + tidx;

	// kernel is launched with exaxtly:
	// * (dimX/(BDIM_X*LOOP_X)) * (dimY/((BDIM_Y*LOOP_Y)))
	// blocks
	const int virtGridDimX = dimX / (BDIM_X*LOOP_X);
	const int bidy = blockIdx.x / virtGridDimX;
	const int bidx = blockIdx.x - bidy*virtGridDimX;

#if KERNEL_LUT_IN_REGS == 0
	// this path uses shared memory for the LUT
        static_assert(TBL_LOG2_LEN == 4 || TBL_LOG2_LEN == 8);

	constexpr int TABLE_NREP = LUT_SHARED_T<BDIM_X, BDIM_Y, TBL_LOG2_LEN, TABLE_T>::TABLE_NREP;
	constexpr int TABLE_PAD  = LUT_SHARED_T<BDIM_X, BDIM_Y, TBL_LOG2_LEN, TABLE_T>::TABLE_PAD;
        constexpr int TABLE_LEN  = LUT_SHARED_T<BDIM_X, BDIM_Y, TBL_LOG2_LEN, TABLE_T>::TABLE_LEN;

        __shared__ TABLE_T sh[TABLE_NREP][TABLE_LEN+TABLE_PAD];

	LUT_SHARED_T<BDIM_X, BDIM_Y, TBL_LOG2_LEN, TABLE_T> lut_s;
	lut_s.init(bitTbl, sh);
#else
	// this path uses shuffles memory for the LUT
	// so TBL_LOG2_LEN must be equal to 4
	static_assert(TBL_LOG2_LEN == 4);
	static_assert(BDIM_X*BDIM_Y >= (1 << TBL_LOG2_LEN));

	LUT_SHUFFLE_T<TBL_LOG2_LEN, TABLE_T> lut_s(bitTbl);
#endif
        UINT2_T me[LOOP_Y][LOOP_X];

        const int startX =        bidx*LOOP_X*BDIM_X + tidx;
        const int startY = begY + bidy*LOOP_Y*BDIM_Y + tidy;

        dst += startY*dimX + startX;

        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {
                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {

                        me[i][j] = dst[i*BDIM_Y*dimX + j*BDIM_X];

                        me[i][j].x = ~me[i][j].x;
                        me[i][j].y = ~me[i][j].y;
                }
        }

        UINT2_T up[LOOP_Y][LOOP_X];
        UINT2_T ct[LOOP_Y][LOOP_X];
        UINT2_T dw[LOOP_Y][LOOP_X];
        UINT2_T sd[LOOP_Y][LOOP_X];
#if 0
        __shared__ UINT2_T shTile[BDIM_Y*LOOP_Y+2][BDIM_X*LOOP_X+2];
        loadTile_d<BDIM_X, BDIM_Y, BDIM_X*LOOP_X, BDIM_Y*LOOP_Y>(bidx, bidy, begY, dimY, dimX, src, shTile);
	__syncthreads();

        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {
                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {
                        up[i][j] = shTile[i*BDIM_Y +   tidy][j*BDIM_X + 1+tidx];
                        ct[i][j] = shTile[i*BDIM_Y + 1+tidy][j*BDIM_X + 1+tidx];
                        sd[i][j] = shTile[i*BDIM_Y + 1+tidy][j*BDIM_X + 2*(!readBack)+tidx];
                        dw[i][j] = shTile[i*BDIM_Y + 2+tidy][j*BDIM_X + 1+tidx];
                }
        }
#else
        loadTileReg_d<BDIM_X, BDIM_Y, LOOP_X, LOOP_Y, COLOR>(bidx, bidy, begY, dimY, dimX, src, ct, up, dw, sd);
#endif
        // BDIM_Y is power of two so row parity won't change across loops
        const int readBack = (COLOR == C_BLACK) ? !(tidy%2) : (tidy%2);

        if (readBack) {
                #pragma unroll
                for(int i = 0; i < LOOP_Y; i++) {
                        #pragma unroll
                        for(int j = 0; j < LOOP_X; j++) {
                                sd[i][j].x = (sd[i][j].y >> (BITXWD-BITXSP)) | (ct[i][j].x << BITXSP);
                                sd[i][j].y = (ct[i][j].x >> (BITXWD-BITXSP)) | (ct[i][j].y << BITXSP);
                        }
                }
        } else {
                #pragma unroll
                for(int i = 0; i < LOOP_Y; i++) {
                        #pragma unroll
                        for(int j = 0; j < LOOP_X; j++) {
                                sd[i][j].y = (ct[i][j].y >> BITXSP) | (sd[i][j].x << (BITXWD-BITXSP));
                                sd[i][j].x = (ct[i][j].x >> BITXSP) | (ct[i][j].y << (BITXWD-BITXSP));
                        }
                }
        }

        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {
                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {

                        up[i][j].x ^= me[i][j].x;
                        up[i][j].y ^= me[i][j].y;

                        dw[i][j].x ^= me[i][j].x;
                        dw[i][j].y ^= me[i][j].y;

                        sd[i][j].x ^= me[i][j].x;
                        sd[i][j].y ^= me[i][j].y;

                        ct[i][j].x ^= me[i][j].x;
                        ct[i][j].y ^= me[i][j].y;
                }
        }

        UINT2_T eq4[LOOP_Y][LOOP_X];
        UINT2_T eq3[LOOP_Y][LOOP_X];
#if 1
	// sorts bits in ascending order in <up, dw, sd, ct>
	sortBits_d<BDIM_X, BDIM_Y, LOOP_X, LOOP_Y>(up, dw, sd, ct);

        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {
                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {
                        eq4[i][j] = {up[i][j].x,
                                     up[i][j].y};

                        eq3[i][j] = {~up[i][j].x & dw[i][j].x,
                                     ~up[i][j].y & dw[i][j].y};
                }
        }
#else
        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {
                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {
                        eq4[i][j] = {up[i][j].x & dw[i][j].x & sd[i][j].x & ct[i][j].x,
                                     up[i][j].y & dw[i][j].y & sd[i][j].y & ct[i][j].y};

                        eq3[i][j] = {(~up[i][j].x &  dw[i][j].x &  sd[i][j].x &  ct[i][j].x) |
                                     ( up[i][j].x & ~dw[i][j].x &  sd[i][j].x &  ct[i][j].x) |
                                     ( up[i][j].x &  dw[i][j].x & ~sd[i][j].x &  ct[i][j].x) |
                                     ( up[i][j].x &  dw[i][j].x &  sd[i][j].x & ~ct[i][j].x),

                                     (~up[i][j].y &  dw[i][j].y &  sd[i][j].y &  ct[i][j].y) |
                                     ( up[i][j].y & ~dw[i][j].y &  sd[i][j].y &  ct[i][j].y) |
                                     ( up[i][j].y &  dw[i][j].y & ~sd[i][j].y &  ct[i][j].y) |
                                     ( up[i][j].y &  dw[i][j].y &  sd[i][j].y & ~ct[i][j].y)};
                }
        }
#endif

        static_assert(0 == (sizeof(TABLE_T) % sizeof(unsigned int)));

        constexpr int NRND_X_BSRC = sizeof(TABLE_T)/sizeof(unsigned int);
        constexpr int BSRC_X_BYTE = 8/TBL_LOG2_LEN;

        curandStatePhilox4_32_10_t st;

	const unsigned long long glbtid = 1ull*devid*gridDim.x*BDIM_X*BDIM_Y + blockIdx.x*BDIM_X*BDIM_Y + tid;
        const unsigned long long offset = 4ull*NRND_X_BSRC*sizeof(UINT_T)*BSRC_X_BYTE*LOOP_X*LOOP_Y*(2*it+COLOR);

        curand_init(seed, glbtid, offset, &st);

        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {
                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {

                        UINT2_T exp4 = {0, 0};
                        UINT2_T exp8 = {0, 0};

                        union {
                                TABLE_T t;
                                unsigned int u[NRND_X_BSRC];
                        } rnd;

                        #pragma unroll
                        for(int k = 0; k < BITXWD; k += TBL_LOG2_LEN) {
                                #pragma unroll
                                for(int r = 0; r < NRND_X_BSRC; r++) {
                                        rnd.u[r] = curand(&st);
                                }
                                exp4.x |= (UINT_T)lut_s.getMaxLE(rnd.t) << k;
                        }

                        #pragma unroll
                        for(int k = 0; k < BITXWD; k += TBL_LOG2_LEN) {
                                #pragma unroll
                                for(int r = 0; r < NRND_X_BSRC; r++) {
                                        rnd.u[r] = curand(&st);
                                }
                                exp4.y |= (UINT_T)lut_s.getMaxLE(rnd.t) << k;
                        }

                        #pragma unroll
                        for(int k = 0; k < BITXWD; k += TBL_LOG2_LEN) {
                                #pragma unroll
                                for(int r = 0; r < NRND_X_BSRC; r++) {
                                        rnd.u[r] = curand(&st);
                                }
                                exp8.x |= (UINT_T)lut_s.getMaxLE(rnd.t) << k;
                        }

                        #pragma unroll
                        for(int k = 0; k < BITXWD; k += TBL_LOG2_LEN) {
                                #pragma unroll
                                for(int r = 0; r < NRND_X_BSRC; r++) {
                                        rnd.u[r] = curand(&st);
                                }
                                exp8.y |= (UINT_T)lut_s.getMaxLE(rnd.t) << k;
                        }
#if 0
                        if (!bidx && !bidy && !tidx && !tidy) {
                                printf("exp4: %016llX, %016llX, exp8: %016llX, %016llX\n", exp4.x, exp4.y, exp8.x, exp8.y);
                        }
#endif                        
                        exp8.x &= exp4.x;
                        exp8.y &= exp4.y;
#if 0
                        if (eq4.x & eq3.x) {
                                printf("%s:%d: error, eq4.x & eq3.x: %016llX (eq4.x: %016llX, eq3.x: %016llX)\n",
                                        __func__, __LINE__, eq4.x & eq3.x, eq4.x, eq3.x);
                        }
                        if (eq4.y & eq3.y) {
                                printf("%s:%d: error, eq4.y & eq3.y: %016llX (eq4.y: %016llX, eq3.y: %016llX)\n",
                                        __func__, __LINE__, eq4.y & eq3.y, eq4.y, eq3.y);
                        }
#endif
                        UINT2_T flip = {0, 0};

                        flip.x = (eq4[i][j].x & exp8.x) | (eq3[i][j].x & exp4.x) | (~eq4[i][j].x & ~eq3[i][j].x);
                        flip.y = (eq4[i][j].y & exp8.y) | (eq3[i][j].y & exp4.y) | (~eq4[i][j].y & ~eq3[i][j].y);

                        me[i][j].x ^= flip.x;
                        me[i][j].y ^= flip.y;
                }
        }

        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {
                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {
                        me[i][j].x = ~me[i][j].x;
                        me[i][j].y = ~me[i][j].y;
                        dst[i*BDIM_Y*dimX + j*BDIM_X] = me[i][j];
                }
        }

        return;
}

// Meant to be called with 1 thread
__global__ void setFlag_k(const long long it, long long *ptr) {
	ptr[0] = it;
	return;
}

#if 1
__device__ __forceinline__ long long __ld_strong_glb(long long *ptr) {
        
        long long ret;
        asm volatile("ld.relaxed.sys.global.u64 %0, [%1];\n\t" : "=l"(ret) : "l"(ptr) : "memory");
        return ret;
}

__device__ __forceinline__ void __st_strong_glb(long long *ptr, long long val) {

        asm volatile("st.relaxed.sys.global.u64 [%0], %1;\n\t" :: "l"(ptr), "l"(val) : "memory");
        return;         
}

__global__ void setAndWaitFlag_k(long long value,
				 long long *__restrict__ setPtr,
				 long long *__restrict__ chkPtr) {

	__st_strong_glb(setPtr, value);

	long long v;
	do { v = __ld_strong_glb(chkPtr); } while (v != value);

	return;
}
#else
__global__ void setAndWaitFlag_k(long long value,
				 long long *__restrict__ setPtr,
				 long long *__restrict__ chkPtr) {

	__nv_atomic_store_n(setPtr,
			    value,
			    __NV_ATOMIC_RELAXED,
			    __NV_THREAD_SCOPE_SYSTEM);
	long long v;
	do {
		v = __nv_atomic_load_n(chkPtr,
				       __NV_ATOMIC_RELAXED,
				       __NV_THREAD_SCOPE_SYSTEM);
	} while (v != value);

	return;
}
#endif

template<int BDIM_X,
         int BDIM_Y,
         int WSIZE,
         typename T>
__device__ __forceinline__ T __block_sum_d(T v) {

        constexpr int NWARP = (BDIM_X*BDIM_Y)/WSIZE;

        __shared__ T sh[NWARP];

        const int tid = threadIdx.y*BDIM_X + threadIdx.x;

        const int lid = tid % WSIZE;
        const int wid = tid / WSIZE;

        #pragma unroll
        for(int i = WSIZE/2; i; i >>= 1) {
                v += __shfl_down_sync(0xFFFFFFFF, v, i);
        }
        if (lid == 0) sh[wid] = v;

        __syncthreads();
        if (wid == 0) {
                v = (lid < NWARP) ? sh[lid] : 0;

                #pragma unroll
                for(int i = NWARP/2; i; i >>= 1) {
                        v += __shfl_down_sync(0xFFFFFFFF, v, i);
                }
        }
        __syncthreads();
        return v;
}

template<int BDIM_X,
         int WSIZE,
         typename T>
__device__ __forceinline__ T __block_sum_d(T v) {

        return __block_sum_d<BDIM_X, 1, WSIZE>(v);
}

template<int BDIM_X,
         int BDIM_Y,
         int LOOP_X, 
         int LOOP_Y,
         int BITXSP,
         int COLOR,
         typename UINT2_T>
__global__ void computeSD_k(const long long begY,
                            const long long dimY,
                            const long long dimX, // ld
                            const double *exp,    //[9],
                            const UINT2_T *__restrict__ src,
                            const UINT2_T *__restrict__ dst,
                                  double *__restrict__ out) {

        static_assert(0 == (BDIM_X & (BDIM_X-1)));
        static_assert(0 == (BDIM_Y & (BDIM_Y-1)));
        static_assert(0 == (BITXSP & (BITXSP-1)));

        using UINT_T = decltype(src[0].x);

        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;
	
	// kernel il launched with exaxtly:
	// * (dimX/(BDIM_X*LOOP_X)) * (dimY/((BDIM_Y*LOOP_Y)))
	// blocks
	const int bidx = blockIdx.x % (dimX / (BDIM_X*LOOP_X));
	const int bidy = blockIdx.x / (dimX / (BDIM_X*LOOP_X));

        __shared__ UINT2_T shTile[BDIM_Y*LOOP_Y+2][BDIM_X*LOOP_X+2];

        loadTile_d<BDIM_X, BDIM_Y, BDIM_X*LOOP_X, BDIM_Y*LOOP_Y>(bidx, bidy, begY, dimY, dimX, src, shTile);

        __shared__ double shExp[5];

        for(int i = tidy*BDIM_X + tidx;
            i < sizeof(shExp)/sizeof(shExp[0]);
            i += BDIM_X*BDIM_Y) {

                shExp[i] = exp[i];
        }
        __syncthreads();

        const int __i = bidy*BDIM_Y*LOOP_Y + tidy;
        const int __j = bidx*BDIM_X*LOOP_X + tidx;

        UINT2_T me[LOOP_Y][LOOP_X];

        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {
                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {
                        me[i][j] = dst[(begY+__i+i*BDIM_Y)*dimX + __j+j*BDIM_X];
                }
        }

        UINT2_T up[LOOP_Y][LOOP_X];
        UINT2_T ct[LOOP_Y][LOOP_X];
        UINT2_T dw[LOOP_Y][LOOP_X];
        UINT2_T sd[LOOP_Y][LOOP_X];

        // BDIM_Y is power of two so row parity won't change across loops
        const int readBack = (COLOR == C_BLACK) ? !(__i%2) : (__i%2);

        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {
                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {
                        up[i][j] = shTile[i*BDIM_Y +   tidy][j*BDIM_X + 1+tidx];
                        ct[i][j] = shTile[i*BDIM_Y + 1+tidy][j*BDIM_X + 1+tidx];
                        sd[i][j] = shTile[i*BDIM_Y + 1+tidy][j*BDIM_X + 2*(!readBack)+tidx];
                        dw[i][j] = shTile[i*BDIM_Y + 2+tidy][j*BDIM_X + 1+tidx];
                }
        }

        if (readBack) {
                #pragma unroll
                for(int i = 0; i < LOOP_Y; i++) {
                        #pragma unroll
                        for(int j = 0; j < LOOP_X; j++) {
                                sd[i][j].x = (ct[i][j].x << BITXSP) | (sd[i][j].y >> (8*sizeof(sd[i][j].y)-BITXSP));
                                sd[i][j].y = (ct[i][j].y << BITXSP) | (ct[i][j].x >> (8*sizeof(ct[i][j].x)-BITXSP));
                        }
                }
        } else {
                #pragma unroll
                for(int i = 0; i < LOOP_Y; i++) {
                        #pragma unroll
                        for(int j = 0; j < LOOP_X; j++) {
                                sd[i][j].y = (ct[i][j].y >> BITXSP) | (sd[i][j].x << (8*sizeof(sd[i][j].x)-BITXSP));
                                sd[i][j].x = (ct[i][j].x >> BITXSP) | (ct[i][j].y << (8*sizeof(ct[i][j].y)-BITXSP));
                        }
                }
        }

        if constexpr(BITXSP == 4) {
                #pragma unroll
                for(int i = 0; i < LOOP_Y; i++) {
                        #pragma unroll
                        for(int j = 0; j < LOOP_X; j++) {
                                ct[i][j].x += up[i][j].x;
                                dw[i][j].x += sd[i][j].x;
                                ct[i][j].x += dw[i][j].x;

                                ct[i][j].y += up[i][j].y;
                                dw[i][j].y += sd[i][j].y;
                                ct[i][j].y += dw[i][j].y;
                        }
                }
        } else if constexpr(BITXSP == 2) {
                #pragma unroll
                for(int i = 0; i < LOOP_Y; i++) {
                        #pragma unroll
                        for(int j = 0; j < LOOP_X; j++) {
                                ct[i][j].x += up[i][j].x;
                                dw[i][j].x += sd[i][j].x;

                                ct[i][j].y += up[i][j].y;
                                dw[i][j].y += sd[i][j].y;
                        }
                }
        } // BITXSP == 1 do nothing

        double mysd = 0;

        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {
                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {
                        #pragma unroll
                        for(int z = 0; z < 8*sizeof(UINT_T); z += BITXSP) {

                                uint2 sum;
                                if constexpr(BITXSP == 4) {
                                        sum = make_uint2((ct[i][j].x >> z) & SPIN_MASK, 
                                                         (ct[i][j].y >> z) & SPIN_MASK);
                                }
                                if constexpr(BITXSP == 2) {
                                        sum = make_uint2(((ct[i][j].x >> z) & SPIN_MASK) +
                                                         ((dw[i][j].x >> z) & SPIN_MASK),

                                                         ((ct[i][j].y >> z) & SPIN_MASK) +
                                                         ((dw[i][j].y >> z) & SPIN_MASK));
                                }
                                if constexpr(BITXSP == 1) {
                                        sum = make_uint2(((ct[i][j].x >> z) & SPIN_MASK) +
                                                         ((up[i][j].x >> z) & SPIN_MASK) + 
                                                         ((dw[i][j].x >> z) & SPIN_MASK) +
                                                         ((sd[i][j].x >> z) & SPIN_MASK),

                                                         ((ct[i][j].y >> z) & SPIN_MASK) +
                                                         ((up[i][j].y >> z) & SPIN_MASK) + 
                                                         ((dw[i][j].y >> z) & SPIN_MASK) +
                                                         ((sd[i][j].y >> z) & SPIN_MASK)); 
                                }

                                int indx = (int(sum.x)*2 - 4) * (int((me[i][j].x >> z) & SPIN_MASK)*2 - 1);
                                int indy = (int(sum.y)*2 - 4) * (int((me[i][j].y >> z) & SPIN_MASK)*2 - 1);

                                mysd += shExp[(indx+4)/2] + shExp[(indy+4)/2];
                        }
                }
        }

        mysd = __block_sum_d<BDIM_X, BDIM_Y, WARP_SIZE>(mysd);

        if (!threadIdx.x && !threadIdx.y) {
                atomicAdd(out, mysd);
        }

        return;

}

double computeSD(const int ndev,
                 const dim3 grid,
                 const dim3 block,
                 const long long Y,
                 const long long lld,
                 const double *expSD_m,
                 const ulonglong2 *black_m,
                 const ulonglong2 *white_m) {

        constexpr int SPIN_X_WORD = (8*sizeof(*black_m)) / BIT_X_SPIN;

        int rank = 0;
	int ntask = 1;
#ifdef USE_MNNVL
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ntask);
#endif
	const long long dimY = ntask*ndev*Y;

        double *sum_d[MAX_GPU];
        for(int i = 0; i < ndev; i++) {

		const long long begY = (rank*ndev + i)*Y;

                CHECK_CUDA(cudaSetDevice(i));

                CHECK_CUDA(cudaMalloc(sum_d+i,     sizeof(**sum_d)));
                CHECK_CUDA(cudaMemset(sum_d[i], 0, sizeof(**sum_d)));

                computeSD_k<BLOCK_X, BLOCK_Y,
                            BREAD_X, BREAD_Y,
                            BIT_X_SPIN, C_BLACK>
                            <<<grid, block>>>(begY, dimY, lld, expSD_m, white_m, black_m, sum_d[i]);
                CHECK_ERROR("computeSD_k");

                computeSD_k<BLOCK_X, BLOCK_Y,
                            BREAD_X, BREAD_Y,
                            BIT_X_SPIN, C_WHITE>
                            <<<grid, block>>>(begY, dimY, lld, expSD_m, black_m, white_m, sum_d[i]);
                CHECK_ERROR("computeSD_k");
        }

        double ret = 0;

        for(int i = 0; i < ndev; i++) {
                double tmp;
                CHECK_CUDA(cudaSetDevice(i));
                CHECK_CUDA(cudaMemcpy(&tmp, sum_d[i], sizeof(tmp), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaFree(sum_d[i]));

                ret += tmp;
        }

#ifdef USE_MNNVL
	MPI_Allreduce(MPI_IN_PLACE, &ret, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

        return ret / (2ull*dimY*lld * SPIN_X_WORD);

}


template<int BDIM_X,
         int BDIM_Y,
         int LOOP_X, 
         int LOOP_Y,
         int BITXSP,
         int COLOR,
         typename UINT2_T>
__global__ void computeEne_k(const long long begY,
                             const long long dimY,
                             const long long dimX, // ld
                             const UINT2_T *__restrict__ src,
                             const UINT2_T *__restrict__ dst,
                                   double *__restrict__ out) {

        static_assert(0 == (BDIM_X & (BDIM_X-1)));
        static_assert(0 == (BDIM_Y & (BDIM_Y-1)));
        static_assert(0 == (BITXSP & (BITXSP-1)));

        using UINT_T = decltype(src[0].x);

        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;

	// kernel il launched with exaxtly:
	// * (dimX/(BDIM_X*LOOP_X)) * (dimY/((BDIM_Y*LOOP_Y)))
	// blocks
	const int bidx = blockIdx.x % (dimX / (BDIM_X*LOOP_X));
	const int bidy = blockIdx.x / (dimX / (BDIM_X*LOOP_X));

        __shared__ UINT2_T shTile[BDIM_Y*LOOP_Y+2][BDIM_X*LOOP_X+2];

        loadTile_d<BDIM_X, BDIM_Y, BDIM_X*LOOP_X, BDIM_Y*LOOP_Y>(bidx, bidy, begY, dimY, dimX, src, shTile);

        __syncthreads();

        const int __i = bidy*BDIM_Y*LOOP_Y + tidy;
        const int __j = bidx*BDIM_X*LOOP_X + tidx;

        UINT2_T me[LOOP_Y][LOOP_X];

        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {
                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {
                        me[i][j] = dst[(begY+__i+i*BDIM_Y)*dimX + __j+j*BDIM_X];
                }
        }

        UINT2_T up[LOOP_Y][LOOP_X];
        UINT2_T ct[LOOP_Y][LOOP_X];
        UINT2_T dw[LOOP_Y][LOOP_X];
        UINT2_T sd[LOOP_Y][LOOP_X];

        // BDIM_Y is power of two so row parity won't change across loops
        const int readBack = (COLOR == C_BLACK) ? !(__i%2) : (__i%2);

        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {
                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {
                        up[i][j] = shTile[i*BDIM_Y +   tidy][j*BDIM_X + 1+tidx];
                        ct[i][j] = shTile[i*BDIM_Y + 1+tidy][j*BDIM_X + 1+tidx];
                        sd[i][j] = shTile[i*BDIM_Y + 1+tidy][j*BDIM_X + 2*(!readBack)+tidx];
                        dw[i][j] = shTile[i*BDIM_Y + 2+tidy][j*BDIM_X + 1+tidx];
                }
        }

        if (readBack) {
                #pragma unroll
                for(int i = 0; i < LOOP_Y; i++) {
                        #pragma unroll
                        for(int j = 0; j < LOOP_X; j++) {
                                sd[i][j].x = (ct[i][j].x << BITXSP) | (sd[i][j].y >> (8*sizeof(sd[i][j].y)-BITXSP));
                                sd[i][j].y = (ct[i][j].y << BITXSP) | (ct[i][j].x >> (8*sizeof(ct[i][j].x)-BITXSP));
                        }
                }
        } else {
                #pragma unroll
                for(int i = 0; i < LOOP_Y; i++) {
                        #pragma unroll
                        for(int j = 0; j < LOOP_X; j++) {
                                sd[i][j].y = (ct[i][j].y >> BITXSP) | (sd[i][j].x << (8*sizeof(sd[i][j].x)-BITXSP));
                                sd[i][j].x = (ct[i][j].x >> BITXSP) | (ct[i][j].y << (8*sizeof(ct[i][j].y)-BITXSP));
                        }
                }
        }

        if constexpr(BITXSP == 4) {
                #pragma unroll
                for(int i = 0; i < LOOP_Y; i++) {
                        #pragma unroll
                        for(int j = 0; j < LOOP_X; j++) {
                                ct[i][j].x += up[i][j].x;
                                dw[i][j].x += sd[i][j].x;
                                ct[i][j].x += dw[i][j].x;

                                ct[i][j].y += up[i][j].y;
                                dw[i][j].y += sd[i][j].y;
                                ct[i][j].y += dw[i][j].y;
                        }
                }
        } else if constexpr(BITXSP == 2) {
                #pragma unroll
                for(int i = 0; i < LOOP_Y; i++) {
                        #pragma unroll
                        for(int j = 0; j < LOOP_X; j++) {
                                ct[i][j].x += up[i][j].x;
                                dw[i][j].x += sd[i][j].x;

                                ct[i][j].y += up[i][j].y;
                                dw[i][j].y += sd[i][j].y;
                        }
                }
        } // BITXSP == 1 do nothing

        double myene = 0;

        #pragma unroll
        for(int i = 0; i < LOOP_Y; i++) {
                #pragma unroll
                for(int j = 0; j < LOOP_X; j++) {
                        #pragma unroll
                        for(int z = 0; z < 8*sizeof(UINT_T); z += BITXSP) {

                                uint2 sum;
                                if constexpr(BITXSP == 4) {
                                        sum = make_uint2((ct[i][j].x >> z) & SPIN_MASK, 
                                                         (ct[i][j].y >> z) & SPIN_MASK);
                                }
                                if constexpr(BITXSP == 2) {
                                        sum = make_uint2(((ct[i][j].x >> z) & SPIN_MASK) +
                                                         ((dw[i][j].x >> z) & SPIN_MASK),

                                                         ((ct[i][j].y >> z) & SPIN_MASK) +
                                                         ((dw[i][j].y >> z) & SPIN_MASK));
                                }
                                if constexpr(BITXSP == 1) {
                                        sum = make_uint2(((ct[i][j].x >> z) & SPIN_MASK) +
                                                         ((up[i][j].x >> z) & SPIN_MASK) + 
                                                         ((dw[i][j].x >> z) & SPIN_MASK) +
                                                         ((sd[i][j].x >> z) & SPIN_MASK),

                                                         ((ct[i][j].y >> z) & SPIN_MASK) +
                                                         ((up[i][j].y >> z) & SPIN_MASK) + 
                                                         ((dw[i][j].y >> z) & SPIN_MASK) +
                                                         ((sd[i][j].y >> z) & SPIN_MASK)); 
                                }

                                int indx = (int(sum.x)*2 - 4) * (int((me[i][j].x >> z) & SPIN_MASK)*2 - 1);
                                int indy = (int(sum.y)*2 - 4) * (int((me[i][j].y >> z) & SPIN_MASK)*2 - 1);

                                myene += -(indx + indy);
                        }
                }
        }

        myene = __block_sum_d<BDIM_X, BDIM_Y, WARP_SIZE>(myene);

        if (!threadIdx.x && !threadIdx.y) {
                atomicAdd(out, myene);
        }

        return;

}

double computeEnergy(const int ndev,
                     const dim3 grid,
                     const dim3 block,
                     const size_t Y,
                     const size_t lld,
                     const ulonglong2 *black_m,
                     const ulonglong2 *white_m) {

        constexpr int SPIN_X_WORD = (8*sizeof(*black_m)) / BIT_X_SPIN;

        int rank = 0;
	int ntask = 1;
#ifdef USE_MNNVL
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ntask);
#endif
	const long long dimY = (ntask*ndev)*Y;

        double *sum_d[MAX_GPU];
        for(int i = 0; i < ndev; i++) {

		const long long begY = (rank*ndev + i)*Y;

                CHECK_CUDA(cudaSetDevice(i));

                CHECK_CUDA(cudaMalloc(sum_d+i,     sizeof(**sum_d)));
                CHECK_CUDA(cudaMemset(sum_d[i], 0, sizeof(**sum_d)));

                computeEne_k<BLOCK_X, BLOCK_Y,
                             BREAD_X, BREAD_Y,
                             BIT_X_SPIN, C_BLACK>
                             <<<grid, block>>>(begY, dimY, lld, white_m, black_m, sum_d[i]);
                CHECK_ERROR("computeEne_k");
        }

        double ret = 0;

        for(int i = 0; i < ndev; i++) {
                double tmp;
                CHECK_CUDA(cudaSetDevice(i));
                CHECK_CUDA(cudaMemcpy(&tmp, sum_d[i], sizeof(tmp), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaFree(sum_d[i]));

                ret += tmp;
        }

#ifdef USE_MNNVL
	MPI_Allreduce(MPI_IN_PLACE, &ret, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
        return ret / (2ull*dimY*lld * SPIN_X_WORD);

}

template<int BDIM_X,
         int BITXSP,
         typename UINT2_T,
         typename SUM_T>
__global__ void spinCount_k(const long long dimX,
                            const UINT2_T *__restrict__ v,
                                  SUM_T *__restrict__ sum) {

        //const long long nth = static_cast<long long>(blockDim.x)*gridDim.x;
        //const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

        constexpr int SPIN_X_WORD = 8*sizeof(UINT2_T)/BITXSP;

        SUM_T cntP = 0;
        SUM_T cntN = 0;

        v += blockIdx.x*dimX;

        for(long long i = threadIdx.x; i < dimX; i += BDIM_X) {
                UINT2_T spack = v[i];
#if 1
#if 0
                // assuming that the spin values in the arrays can only
                // be 0b00, 0b01 and 0b10, this could be done with less 
                // operations, as below, but we want to count explicitly 
                // all the spin values as a sanity check
                unsigned int p1 = __mypopc(spack.x & 0x1111111111111111ull) + 
                                  __mypopc(spack.y & 0x1111111111111111ull);  // +1 spins are represented as 0b10

                unsigned int n1 = __mypopc(~spack.x & 0x1111111111111111ull) +
                                  __mypopc(~spack.y & 0x1111111111111111ull); // -1 spins are represented as 0b00

                cntP += p1;
                cntN += n1;
#else

                unsigned int p1 = __mypopc(spack.x) + __mypopc(spack.y);
                cntP += p1;
                cntN += SPIN_X_WORD - p1;
#endif

#else
                // slow, for debug
                using UINT_T = decltype(v[0].x);

                #pragma unroll
                for(int z = 0; z < 8*sizeof(UINT_T); z += BITXSP) {

                        unsigned int sx = (spack.x >> z) & SPIN_MASK;
                        unsigned int sy = (spack.y >> z) & SPIN_MASK;
#if 0
                        if (sx > 2) { printf("%s:%d: error: sx > 2 (=%d)\n", __func__, __LINE__, sx); }
                        if (sy > 2) { printf("%s:%d: error: sy > 2 (=%d)\n", __func__, __LINE__, sy); }
#endif
                        switch(sx) {
                                case 0: cntN++; break;
                                case 1: cntZ++; break;
                                case 2: cntP++; break;
                                default: printf("Error!");
                        }
                        switch(sy) {
                                case 0: cntN++; break;
                                case 1: cntZ++; break;
                                case 2: cntP++; break;
                                default: printf("Error!");
                        }
                }
#endif
        }

        cntN = __block_sum_d<BDIM_X, WARP_SIZE>(cntN);
        cntP = __block_sum_d<BDIM_X, WARP_SIZE>(cntP);

        if (threadIdx.x == 0) {
                atomicAdd(sum+0, cntN);
                atomicAdd(sum+1, cntP);
        }
        return;
}

static void countSpins(const int ndev,
                       const long long Y,
                       const long long lld,
                       const ulonglong2 *black_m,
                       const ulonglong2 *white_m,
                             unsigned long long **sum_d,
                             unsigned long long  *sum_h) {
        int rank = 0;
	int ntask = 1;
#ifdef USE_MNNVL
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ntask);
#endif
        for(int i = 0; i < ndev; i++) {

		const long long begY = (rank*ndev + i)*Y;

                CHECK_CUDA(cudaSetDevice(i));
                CHECK_CUDA(cudaMemset(sum_d[i], 0, 2*sizeof(**sum_d)));
                spinCount_k<THREADS, BIT_X_SPIN><<<Y, THREADS>>>(lld, black_m + begY*lld, sum_d[i]);
                spinCount_k<THREADS, BIT_X_SPIN><<<Y, THREADS>>>(lld, white_m + begY*lld, sum_d[i]);
                CHECK_ERROR("spinCount_k");
        }

        sum_h[S_NEG1] = 0;
        sum_h[S_POS1] = 0;

        for(int i = 0; i < ndev; i++) {

                unsigned long long  cnt_h[2];

                CHECK_CUDA(cudaSetDevice(i));
                CHECK_CUDA(cudaMemcpy(cnt_h, sum_d[i], sizeof(cnt_h), cudaMemcpyDeviceToHost));

                sum_h[S_NEG1] += cnt_h[S_NEG1];
                sum_h[S_POS1] += cnt_h[S_POS1];
        }

        unsigned long long totalSpins = sizeof(*black_m)*8*2*Y*lld*ndev*ntask/BIT_X_SPIN;

#ifdef USE_MNNVL
	MPI_Allreduce(MPI_IN_PLACE, sum_h, 2, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
        assert(sum_h[S_NEG1]+sum_h[S_POS1] == totalSpins);

        return;
}

template<int BDIM_X,
         int BITXSP,
         typename UINT_T,
         typename SUM_T>
__global__ void corr2D_slow_k(const int nsamp,
                              const int *corr_samp,
                              const long long begY,
                              const long long dimY,
                              const long long dimX, // ld
                              const UINT_T *__restrict__ black,
                              const UINT_T *__restrict__ white,
                                    SUM_T  *__restrict__ corr) {

        constexpr int SPIN_X_WORD = 8*sizeof(UINT_T)/BITXSP;

        const int tid = threadIdx.x;

        const long long __i = begY + blockIdx.x;

        const long long X = dimX*2*SPIN_X_WORD;

        for(int j = 0; j < nsamp; j++) {

                const int r = corr_samp[j];

                SUM_T sum = 0;

                for(long long l = 0; l < X; l += BDIM_X) {


                        // threads beyond read last spin word but wthey won't use it
                        const int __j = l+tid < X ? l+tid : X-1; 

                        // read spin (__i, __j)
                        const long long myWrdX = (__j/2) / SPIN_X_WORD;
                        const long long myOffX = (__j/2) % SPIN_X_WORD;

                        UINT_T tmp = ((__i ^ __j) & 1) ? white[__i*dimX + myWrdX]:
                                                         black[__i*dimX + myWrdX];

                        const unsigned int mySpin = (tmp >> (myOffX*BITXSP)) & SPIN_MASK;

                        //if (mySpin != 1 && l+tid < X) {
                        if (l+tid < X) {

                                // read spin (__i, __j+r)
                                const long long nextX = __j+r < X ? __j+r : __j+r - X;

                                const long long nextWrdX = (nextX/2) / SPIN_X_WORD;
                                const long long nextOffX = (nextX/2) % SPIN_X_WORD;

                                tmp = ((__i ^ nextX) & 1) ? white[__i*dimX + nextWrdX]:
                                                            black[__i*dimX + nextWrdX];

                                const unsigned int nextSpin = (tmp >> (nextOffX*BITXSP)) & SPIN_MASK;

                                sum += (mySpin == nextSpin) ? SUM_T(1) : SUM_T(-1);

                                // read spin (__i+r, __j)
                                const long long nextY = (__i+r < dimY) ? __i+r : __i+r-dimY;

                                tmp = ((nextY ^ __j) & 1) ? white[nextY*dimX + myWrdX]:
                                                            black[nextY*dimX + myWrdX];

                                const unsigned int vertSpin = (tmp >> (myOffX*BITXSP)) & SPIN_MASK;

                                sum += (mySpin == vertSpin) ? SUM_T(1) : SUM_T(-1);
                        }

                }

                sum = __block_sum_d<BDIM_X, WARP_SIZE>(sum);
                if (!tid) {
                        atomicAdd(corr + j, sum);
                }
        }
        return;
}

// returns the 128-bit word from bit k of [a, b];
// k < 128 (sizeof(a.x)*8 * 2)
__device__ ulonglong2 __my_shift256_r(int k,
                                      ulonglong2 a,
                                      ulonglong2 b) {

        constexpr int BASEBITS = sizeof(a.x)*8;

        if (k/BASEBITS) {
                a.x = a.y;
                a.y = b.x;
                b.x = b.y;
                k = k % BASEBITS;
        }

        ulonglong2 ret;
        ret.x = (a.x >> k) | (a.y << (8*sizeof(a.y) - k));
        ret.y = (a.y >> k) | (b.x << (8*sizeof(b.x) - k));

        return ret;
}

template<int BDIM_X,
         int BITXSP,
         typename UINT2_T,
         typename SUM_T>
__global__ void corr2D_full_k(const int nsamp,
                              const int *corr_samp,
                              const long long begY,
                              const long long dimY,
                              const long long dimX, // ld
                              const UINT2_T *__restrict__ black,
                              const UINT2_T *__restrict__ white,
                                    SUM_T  *__restrict__ corr) {

        constexpr int SPIN_X_WORD = 8*sizeof(UINT2_T)/BITXSP;

        using UINT_T = decltype(black[0].x);

        const int tid = threadIdx.x;

        const long long __i = begY + blockIdx.x;

        const int rdispB =  (__i & 1);
        const int rdispW = !(__i & 1);

        for(int j = 0; j < nsamp; j++) {

                const int r = corr_samp[j];

                SUM_T sum = 0;

                const int rB = (r+rdispB)/2;
                const int rW = (r+rdispW)/2;

                const UINT_T wordOffB = rB/SPIN_X_WORD;
                const UINT_T wordOffW = rW/SPIN_X_WORD;

                for(long long l = tid; l < dimX; l += BDIM_X) {

                        const UINT2_T spinB2 = black[__i*dimX + l];
                        const UINT2_T spinW2 = white[__i*dimX + l];

//                        const UINT_T spinB = spinB2.x | (spinB2.y << (BITXSP/2));
//                        const UINT_T spinW = spinW2.x | (spinW2.y << (BITXSP/2));

                        // handle horizontal correlation

                        // get words containing spins at distance r from spins in spinB
                        UINT_T __j0 = (l+wordOffB  ) % dimX; //wordOff0 < dimX ? wordOff0 : wordOff0-dimX;
                        UINT_T __j1 = (l+wordOffB+1) % dimX; //wordOff1 < dimX ? wordOff1 : wordOff1-dimX;

                        const UINT2_T neigB_0 = r & 1 ? white[__i*dimX + __j0] : black[__i*dimX + __j0];
                        const UINT2_T neigB_1 = r & 1 ? white[__i*dimX + __j1] : black[__i*dimX + __j1];


                        // get words containing spins at distance r from spins in spinW
                        __j0 = (l+wordOffW  ) % dimX; //wordOff0 < dimX ? wordOff0 : wordOff0-dimX;
                        __j1 = (l+wordOffW+1) % dimX; //wordOff1 < dimX ? wordOff1 : wordOff1-dimX;

                        const UINT2_T neigW_0 = r & 1 ? black[__i*dimX + __j0] : white[__i*dimX + __j0];
                        const UINT2_T neigW_1 = r & 1 ? black[__i*dimX + __j1] : white[__i*dimX + __j1];

                        // align read spin words to spinB / spinW
                        const int spinOffB = (rB % SPIN_X_WORD)*BITXSP;
                        const int spinOffW = (rW % SPIN_X_WORD)*BITXSP;

                        // shift neighboring spins to align them to spinB/spinW
                        const UINT2_T neigB2 = __my_shift256_r(spinOffB, neigB_0, neigB_1);
                        const UINT2_T neigW2 = __my_shift256_r(spinOffW, neigW_0, neigW_1);

                        // pack them in a single UINT_T
//                        const UINT_T neigB = neigB2.x | (neigB2.y << (BITXSP/2));
//                        const UINT_T neigW = neigW2.x | (neigW2.y << (BITXSP/2));

                        // count same and different spins != 0 (value different from 1)
#if 1
//                        UINT_T nzmskB = ((spinB >> 1) | ~spinB) & ((neigB >> 1) | ~neigB);
//                        UINT_T nzmskW = ((spinW >> 1) | ~spinW) & ((neigW >> 1) | ~neigW);

                        UINT2_T neqB = {spinB2.x ^ neigB2.x, spinB2.y ^ neigB2.y};
                        UINT2_T neqW = {spinW2.x ^ neigW2.x, spinW2.y ^ neigW2.y};;

                        //neqB = (neqB >> 1) | neqB;
                        //neqW = (neqW >> 1) | neqW;

//                        sum -= __mypopc( neqB & nzmskB & 0x5555555555555555ull);
//                        sum += __mypopc(~neqB & nzmskB & 0x5555555555555555ull);

                        sum -= __mypopc( neqB.x) + __mypopc( neqB.y);
                        sum += __mypopc(~neqB.x) + __mypopc(~neqB.y);

//                        sum -= __mypopc( neqW & nzmskW & 0x5555555555555555ull);
//                        sum += __mypopc(~neqW & nzmskW & 0x5555555555555555ull);
                        
                        sum -= __mypopc( neqW.x) + __mypopc( neqW.y);
                        sum += __mypopc(~neqW.x) + __mypopc(~neqW.y);
#else
                        // simple loop
                        for(int z = 0; z < 8*sizeof(UINT_T); z += BITXSP) {
                                UINT_T bc = (spinB.x >> z) & SPIN_MASK;
                                UINT_T bh = (neigB.x >> z) & SPIN_MASK;

                                UINT_T wc = (spinW.x >> z) & SPIN_MASK;
                                UINT_T wh = (neigW.x >> z) & SPIN_MASK;

                                sum += (bc == bh) ? SUM_T(1) : SUM_T(-1);
                                sum += (wc == wh) ? SUM_T(1) : SUM_T(-1);
                                
                                UINT_T bc = (spinB.y >> z) & SPIN_MASK;
                                UINT_T bh = (neigB.y >> z) & SPIN_MASK;
                                                    
                                UINT_T wc = (spinW.y >> z) & SPIN_MASK;
                                UINT_T wh = (neigW.y >> z) & SPIN_MASK;

                                sum += (bc == bh) ? SUM_T(1) : SUM_T(-1);
                                sum += (wc == wh) ? SUM_T(1) : SUM_T(-1);
                        }
#endif

                        // handle vertical correlation

                        const long long nextY = (__i+r) % dimY; //(__i+r < dimY) ? __i+r : __i+r-dimY;

                        const UINT2_T vneigB2 = r & 1 ? white[nextY*dimX + l] : black[nextY*dimX + l];
                        const UINT2_T vneigW2 = r & 1 ? black[nextY*dimX + l] : white[nextY*dimX + l];

//                        const UINT_T vneigB = vneigB2.x | (vneigB2.y << (BITXSP/2));
//                        const UINT_T vneigW = vneigW2.x | (vneigW2.y << (BITXSP/2));

                        // count same and different spins != 0 (value different from 1)
#if 1
//                        nzmskB = ((spinB >> 1) | ~spinB) & ((vneigB >> 1) | ~vneigB);
//                        nzmskW = ((spinW >> 1) | ~spinW) & ((vneigW >> 1) | ~vneigW);

                        neqB = {spinB2.x ^ vneigB2.x, spinB2.y ^ vneigB2.y};
                        neqW = {spinW2.x ^ vneigW2.x, spinW2.y ^ vneigW2.y};

//                        neqB = (neqB >> 1) | neqB;
//                        neqW = (neqW >> 1) | neqW;

//                        sum -= __mypopc( neqB & nzmskB & 0x5555555555555555ull);
//                        sum += __mypopc(~neqB & nzmskB & 0x5555555555555555ull);
                        sum -= __mypopc( neqB.x) + __mypopc( neqB.y);
                        sum += __mypopc(~neqB.x) + __mypopc(~neqB.y);

//                        sum -= __mypopc( neqW & nzmskW & 0x5555555555555555ull);
//                        sum += __mypopc(~neqW & nzmskW & 0x5555555555555555ull);
                        sum -= __mypopc( neqW.x) + __mypopc( neqW.y);
                        sum += __mypopc(~neqW.x) + __mypopc(~neqW.y);
#else
                        // simple loop
                        for(int z = 0; z < 8*sizeof(UINT_T); z += BITXSP) {
                                UINT_T bc = ( spinB.x >> z) & SPIN_MASK;
                                UINT_T bv = (vneigB.x >> z) & SPIN_MASK;

                                UINT_T wc = ( spinW.x >> z) & SPIN_MASK;
                                UINT_T wv = (vneigW.x >> z) & SPIN_MASK;

                                sum += (bc == bv) ? SUM_T(1) : SUM_T(-1);
                                sum += (wc == wv) ? SUM_T(1) : SUM_T(-1);
                                
                                UINT_T bc = ( spinB.y >> z) & SPIN_MASK;
                                UINT_T bv = (vneigB.y >> z) & SPIN_MASK;
                                                     
                                UINT_T wc = ( spinW.y >> z) & SPIN_MASK;
                                UINT_T wv = (vneigW.y >> z) & SPIN_MASK;

                                sum += (bc == bv) ? SUM_T(1) : SUM_T(-1);
                                sum += (wc == wv) ? SUM_T(1) : SUM_T(-1);
                        }
#endif
                }

                sum = __block_sum_d<BDIM_X, WARP_SIZE>(sum);
                if (!tid) {
                        atomicAdd(corr + j, sum);
                }
        }
        return;
}

template<int BITXSP,
         typename UINT_T,
         typename SUM_T>
__global__ void corr2D_diag_k(const int nsamp,
                              const int *corr_samp,
                              const long long begY,
                              const long long dimY,
                              const long long dimX, // ld
                              const UINT_T *__restrict__ black,
                              const UINT_T *__restrict__ white,
                                    SUM_T  *__restrict__ corr) {

        constexpr int SPIN_X_WORD = 8*sizeof(UINT_T)/BITXSP;

        const int tid = threadIdx.x;

        const long long __i = begY + blockIdx.x;

        const long long X = dimX*2*SPIN_X_WORD;

        // read spin (__i, __j)
        const unsigned int dgWrdX = (__i/2) / SPIN_X_WORD;
        const unsigned int dgOffX = (__i/2) % SPIN_X_WORD;

        UINT_T tmp = black[__i*dimX + dgWrdX]; // top-left spin is black by 
                                               // convention so diagonal is too

        const unsigned int d_spin = (tmp >> (dgOffX*BITXSP)) & SPIN_MASK;

        // we could skip the loop if d_spin == 1...

        for(int j = tid; j < nsamp; j += blockDim.x) {

                const int r = corr_samp[j];

                // read spin (__i, __i+r)
                const unsigned int nextX = (__i+r) % X; //__i+r < X ? __i+r : __i+r - X;
                const unsigned int hzWrdX = (nextX/2) / SPIN_X_WORD;
                const unsigned int hzOffX = (nextX/2) % SPIN_X_WORD;
                tmp = (r & 1) ? white[__i*dimX + hzWrdX] : black[__i*dimX + hzWrdX];
                const unsigned int h_spin = (tmp >> (hzOffX*BITXSP)) & SPIN_MASK;

                // read spin (__i+r, __i)
                const unsigned int nextY = (__i+r) % dimY; // (__i+r < dimY) ? __i+r : __i+r-dimY;
                tmp = (r & 1) ? white[nextY*dimX + dgWrdX] : black[nextY*dimX + dgWrdX];
                const unsigned int v_spin = (tmp >> (dgOffX*BITXSP)) & SPIN_MASK;

                SUM_T sum = 0;

                sum += (d_spin == h_spin) ? SUM_T(1) : SUM_T(-1);
                sum += (d_spin == v_spin) ? SUM_T(1) : SUM_T(-1);

                atomicAdd(corr + j, sum);
        }

        return;
}

template<int BDIM_X,
         int BITXSP,
         typename UINT_T,
         typename SUM_T>
__global__ void corr2D_chkb_k(const int side,
                              const int nsamp,
                              const int *corr_samp,
                              const long long begY,
                              const long long dimY,
                              const long long dimX, // ld
                              const UINT_T *__restrict__ black,
                              const UINT_T *__restrict__ white,
                                    SUM_T  *__restrict__ corr) {

        constexpr int SPIN_X_WORD = 8*sizeof(UINT_T)/BITXSP;

        const long long tid = threadIdx.x;

        const long long __i = begY + side*blockIdx.x;

        const long long X = dimX*2*SPIN_X_WORD;

        // the &-s below (instead of the %-s) make
        // a difference (~30% on my RTX 6000 Ada
        // with 16x16 squares) so let's make sure
        // that X and Y are powers of 2
        // (no perf impact measured with device
        // side asserts)
        assert(!(   X&(   X-1)));
        assert(!(dimY&(dimY-1)));

        for(int j = 0; j < nsamp; j++) {

                const int r = corr_samp[j];

                SUM_T sum = 0;

                for(long long __j = tid*side; __j < X; __j += BDIM_X*side) {


                        // threads beyond read last spin word but wthey won't use it
                        //const int __j = l+tid < X ? l+tid : X-1; 

                        // read spin (__i, __j)
                        const long long myWrdX = (__j/2) / SPIN_X_WORD;
                        const long long myOffX = (__j/2) % SPIN_X_WORD;

                        UINT_T tmp = ((__i ^ __j) & 1) ? white[__i*dimX + myWrdX]:
                                                         black[__i*dimX + myWrdX];

                        const unsigned int mySpin = (tmp >> (myOffX*BITXSP)) & SPIN_MASK;

//                        if (mySpin != 1) {// && l+tid < X) {

                                // read spin (__i, __j+r)
                                const long long nextX = (__j+r) & (X-1); // __j+r < X ? __j+r : __j+r - X;

                                const long long nextWrdX = (nextX/2) / SPIN_X_WORD;
                                const long long nextOffX = (nextX/2) % SPIN_X_WORD;

                                tmp = ((__i ^ nextX) & 1) ? white[__i*dimX + nextWrdX]:
                                                            black[__i*dimX + nextWrdX];

                                const unsigned int nextSpin = (tmp >> (nextOffX*BITXSP)) & SPIN_MASK;

                                sum += (mySpin == nextSpin) ? SUM_T(1) : SUM_T(-1);

                                // read spin (__i+r, __j)
                                const long long nextY = (__i+r) & (dimY-1); //(__i+r < dimY) ? __i+r : __i+r-dimY;

                                tmp = ((nextY ^ __j) & 1) ? white[nextY*dimX + myWrdX]:
                                                            black[nextY*dimX + myWrdX];

                                const unsigned int vertSpin = (tmp >> (myOffX*BITXSP)) & SPIN_MASK;

                                sum += (mySpin == vertSpin) ? SUM_T(1) : SUM_T(-1);
//                        }

                }

                sum = __block_sum_d<BDIM_X, WARP_SIZE>(sum);
                if (!tid) {
                        atomicAdd(corr + j, sum);
                }
        }
        return;
}

// g(L) = 6*sqrt((log(L) - 16*log(2))/(x*x) + 1)
// where L is the linear system size. The actual value of x is
// x=3.4482758620689657, giving g(2**20) \approx 6.66. However, we opted for a
// safer choice of x \approx 3.3, to make sure that the cutoff would include
// everything important
double get_gL(int L) {

        const double numer = log(L) - 16.0*log(2);
        const double denom = 3.3*3.3;
        return 6.0*sqrt(numer/denom  + 1.0);
}

// r_cut_off = min(256, (int)(g(L) * sqrt(it)+0.5) )
// g(L): 6.00 for L=2^16
//     : 6.72 for L=2^20
int get_r_cutoff(int it, int L) {

        const double gL = get_gL(L);
#if 0
        printf("%s: g(L=%d): %lf\n", __func__, L, gL);
#endif
        return max(256, int(gL*sqrt(double(it)) + 0.5));
}

int *getRValues_m(long long it, int L, int *n) {

        // get r-cutoff value
        const int rcoff = get_r_cutoff(it, L);

        //printf("rcoff: %d\n", rcoff);

        // get the number of "exponentials" r | r-cutoffr-s < r <= L/2 

        int nsamp_exp = 0;
        int exp_start = 0;
        //for(int i = exp_start; ; i++) {
        for(int i = 0; ; i++) {

                int r = (int)pow(2., double(i)/CORR_EXPR_PP2);
                if (r > L/2) {
                        break;
                }

                if (r > rcoff) {
                        nsamp_exp++;
                        if (!exp_start) {
                                exp_start = i;
                        }
                }
        }
        //printf("nsamp_exp: %d, exp_start: %d\n", nsamp_exp, exp_start);

        int nsamp = rcoff+1 + nsamp_exp;

        int *buf_m = NULL;
        CHECK_CUDA(cudaMallocManaged(&buf_m, sizeof(*buf_m)*nsamp));

        for(int i = 0; i <= rcoff; i++) {
                buf_m[i] = i;
        }
        for(int i = 0; i < nsamp_exp; i++) {

                int r = (int)pow(2., double(exp_start + i)/CORR_EXPR_PP2);
                buf_m[rcoff+1 + i] = r;
        }

        CHECK_CUDA(cudaMemAdvise(buf_m,
                                 sizeof(*buf_m)*nsamp,
                                 cudaMemAdviseSetReadMostly, 0));

        n[0] = nsamp;
#if 0
        printf("%s: total r samples: %d\n", __func__, nsamp);
        printf("\tcorr full from corr_samp[%4d]=%4d to corr_samp[             %4d]=%4d (%d samples)\n",
                0, buf_m[0], 
                2*CORR_MIXD_THRS, buf_m[2*CORR_MIXD_THRS],
                2*CORR_MIXD_THRS+1);
        printf("\tcorr chkb from corr_samp[%4d]=%4d to corr_samp[r_cut_off(t)=%4d]=%4d (%d samples), then up to corr_samp[%4d]=%4d (%d sampless)\n",
                2*CORR_MIXD_THRS+1, buf_m[2*CORR_MIXD_THRS+1],
                rcoff, buf_m[rcoff], 
                rcoff-(2*CORR_MIXD_THRS+1)+1,
                nsamp-1, 
                buf_m[nsamp-1],
                nsamp - (2*CORR_MIXD_THRS+1));
#endif
        return buf_m;
}

static void computeCorr(const int corrType,
                        const char *fname,
                        const int ndev,
                        const int it,
                        const int lld,
                        const int Y,    // per-GPU full lattice (B+W) Y
                        const int dimX,    // per-GPU full lattice (B+W) X
                        const ulonglong2 *black_m,
                        const ulonglong2 *white_m) {

        if (corrType == CORR_MIXD) {        
                if (CORR_MIXD_THRS*2 > CORR_ALLR_MAX) {
                        fprintf(stderr,
                                "%s:%d: error, CORR_MIXD_THRS must be <= %d!\n",
                                __func__, __LINE__, CORR_ALLR_MAX/2);
                        exit(EXIT_FAILURE);
                }
        }

        int rank = 0;
        int ntask = 1;
#ifdef USE_MNNVL
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ntask);
#endif 

        // get all r-values in a managed buffer
        int nsamp;
        int *corr_samp_m = getRValues_m(it, dimX, &nsamp);

        double *csum_d[MAX_GPU];
		
	const long long dimY = ntask*ndev*Y;
#if 0
#ifdef USE_MNNVL
	MPI_Barrier(MPI_COMM_WORLD);
#endif
        double t = Wtime();
#endif
        for(int i = 0; i < ndev; i++) {

		const long long begY = (rank*ndev + i)*Y;

                CHECK_CUDA(cudaSetDevice(i));

                CHECK_CUDA(cudaMalloc(csum_d+i,     sizeof(**csum_d)*nsamp));
                CHECK_CUDA(cudaMemset(csum_d[i], 0, sizeof(**csum_d)*nsamp));

                switch(corrType) {
                        case CORR_FULL:
#if 0
                                // L*L naive
                                corr2D_slow_k<THREADS, BIT_X_SPIN>
                                              <<<Y, THREADS>>>(nsamp, 
                                                               corr_samp_m,
                                                               begY,
                                                               dimY,
                                                               lld*2,
                                                               reinterpret_cast<const unsigned long long *>(black_m),
                                                               reinterpret_cast<const unsigned long long *>(white_m),
                                                               csum_d[i]);
                                CHECK_ERROR("corr2D_slow_k");
#else
                                // L*L fast
                                corr2D_full_k<THREADS, BIT_X_SPIN>
                                              <<<Y, THREADS>>>(nsamp, 
                                                               corr_samp_m,
                                                               begY,
                                                               dimY,
                                                               lld,
                                                               black_m,
                                                               white_m,
                                                               csum_d[i]);
                                CHECK_ERROR("corr2D_full_k");
#endif
                                break;
                        case CORR_DIAG:
                                // diagonal naive
                                corr2D_diag_k<BIT_X_SPIN>
                                              <<<Y, THREADS>>>(nsamp, 
                                                               corr_samp_m,
                                                               begY,
                                                               dimY,
                                                               lld*2,
                                                               reinterpret_cast<const unsigned long long *>(black_m),
                                                               reinterpret_cast<const unsigned long long *>(white_m),
                                                               csum_d[i]);
                                CHECK_ERROR("corr2D_diag_k");
                                break;
                        case CORR_CHKB:
                                // checkerboard naive
                                corr2D_chkb_k<THREADS, BIT_X_SPIN>
                                              <<<Y/CORR_CHKB_SIDE, THREADS>>>(CORR_CHKB_SIDE,
                                                                              nsamp, 
                                                                              corr_samp_m,
                                                                              begY,
                                                                              dimY,
                                                                              lld*2,
                                                                              reinterpret_cast<const unsigned long long *>(black_m),
                                                                              reinterpret_cast<const unsigned long long *>(white_m),
                                                                              csum_d[i]);
                                CHECK_ERROR("corr2D_chkb_k");
                                break;
                        case CORR_MIXD:
                                // L*L fast with r <= rcoff (== corr_samp_m[rcoff])
                                corr2D_full_k<THREADS, BIT_X_SPIN>
                                              <<<Y, THREADS>>>(2*CORR_MIXD_THRS+1,
                                                               corr_samp_m,
                                                               begY,
                                                               dimY,
                                                               lld,
                                                               black_m,
                                                               white_m,
                                                               csum_d[i]);
                                CHECK_ERROR("corr2D_full_k");

                                // checkerboard naive with side CORR_MIXD_THRS
                                // and r > rcoff (== corr_samp_m[rcoff])
                                corr2D_chkb_k<THREADS, BIT_X_SPIN>
                                              <<<Y/CORR_MIXD_THRS, THREADS>>>(CORR_MIXD_THRS,
                                                                              nsamp       - 2*CORR_MIXD_THRS-1, 
                                                                              corr_samp_m + 2*CORR_MIXD_THRS+1,
                                                                              begY,
                                                                              dimY,
                                                                              lld*2,
                                                                              reinterpret_cast<const unsigned long long *>(black_m),
                                                                              reinterpret_cast<const unsigned long long *>(white_m),
                                                                              csum_d[i] + 2*CORR_MIXD_THRS+1);
                                CHECK_ERROR("corr2D_chkb_k");
                                break;
                        default:
                                fprintf(stderr,
                                        "%s:%d: error, unepected correlation type: %d!\n",
                                        __func__, __LINE__, corrType);
                                exit(EXIT_FAILURE);
                }
        }
#if 0
        for(int i = 0; i < ndev; i++) {
                CHECK_CUDA(cudaSetDevice(i));
                CHECK_CUDA(cudaDeviceSynchronize());
        }
#ifdef USE_MNNVL
	MPI_Barrier(MPI_COMM_WORLD);
#endif
        t = Wtime()-t;
        PRINTF1(" (corr time: %lf sec) ", t);
#endif
        double *csum_h[MAX_GPU];

        for(int i = 0; i < ndev; i++) {

                csum_h[i] = (double *)Malloc(sizeof(**csum_h)*nsamp);

                CHECK_CUDA(cudaSetDevice(i));
                CHECK_CUDA(cudaMemcpy(csum_h[i],
                                      csum_d[i],
                                      sizeof(**csum_h)*nsamp,
                                      cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaFree(csum_d[i]));
        }
        for(int i = 1; i < ndev; i++) {
                for(int r = 0; r < nsamp; r++) {
                        csum_h[0][r] += csum_h[i][r];
                }
        }
        
	double *corr_h = csum_h[0];
	
#ifdef USE_MNNVL
        MPI_Reduce(rank?corr_h:MPI_IN_PLACE, corr_h, nsamp, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
	if (!rank) {
		switch(corrType) {
			case CORR_FULL: 
				for(int r = 0; r < nsamp; r++) {
					corr_h[r] /= 2.0*dimY*dimX;
				}
				break;
			case CORR_DIAG:
				for(int r = 0; r < nsamp; r++) {
					corr_h[r] /= 2.0*dimY;
				}
				break;
			case CORR_CHKB:
				for(int r = 0; r < nsamp; r++) {
					corr_h[r] /= 2.0*(dimY*dimX)/(CORR_CHKB_SIDE*CORR_CHKB_SIDE);
				}
				break;
			case CORR_MIXD:
				for(int r = 0; r < nsamp; r++) {
					corr_h[r] /= (2.0*dimY*dimX) / (r <= CORR_MIXD_THRS*2 ? 1 : CORR_MIXD_THRS*CORR_MIXD_THRS);
				}
				break;
		}

		FILE *fp = Fopen(fname, "a");

		// write r-values line
		fprintf(fp, "%-25s", "#");
		for(int i = 0; i < nsamp; i++) {
			fprintf(fp, " %25d", corr_samp_m[i]);
		}
		fprintf(fp,"\n");

		// write correlation values
		fprintf(fp, "%25d", it);
		for(int r = 0; r < nsamp; r++) {
			  fprintf(fp," %25.*G", DBL_DECIMAL_DIG, corr_h[r]);
		}
		fprintf(fp,"\n");
		fclose(fp);
	}

        for(int i = 0; i < ndev; i++) {
		free(csum_h[i]);
	}
        CHECK_CUDA(cudaFree(corr_samp_m));

        return;
}

static void dumpSpins(const char *fprefix,
                      const int ndev,
                      const int Y,
                      const size_t lld,
                      const size_t llenGpu,
                      const ulonglong2 *black_m,
                      const ulonglong2 *white_m) {

        char fname[256];

        int rank = 0;
#ifdef USE_MNNVL
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	ulonglong2 *black_h = (ulonglong2 *)Malloc(sizeof(*black_h)*llenGpu);
	ulonglong2 *white_h = (ulonglong2 *)Malloc(sizeof(*white_h)*llenGpu);
#endif
        //#pragma omp parallel for schedule(static)
        for(int d = 0; d < ndev; d++) {
#ifdef USE_MNNVL
		CHECK_CUDA(cudaSetDevice(d));

		CHECK_CUDA(cudaMemcpy(black_h, black_m + rank*ndev*llenGpu + d*llenGpu, sizeof(*black_h)*llenGpu, cudaMemcpyDeviceToHost));
		CHECK_CUDA(cudaMemcpy(white_h, white_m + rank*ndev*llenGpu + d*llenGpu, sizeof(*white_h)*llenGpu, cudaMemcpyDeviceToHost));
#else
                const ulonglong2 *black_h = black_m + rank*ndev*llenGpu + d*llenGpu;
                const ulonglong2 *white_h = white_m + rank*ndev*llenGpu + d*llenGpu;
#endif
                snprintf(fname, sizeof(fname), "%s_rank%d_dev%d.txt", fprefix, rank, d);
                FILE *fp = Fopen(fname, "w");

                for(int i = 0; i < Y; i++) {
                        for(int j = 0; j < lld; j++) {

                                ulonglong2 __1st = (i%2) ? white_h[i*lld + j] : black_h[i*lld + j];
                                ulonglong2 __2nd = (i%2) ? black_h[i*lld + j] : white_h[i*lld + j];

                                for(int k = 0; k < 8*sizeof(__1st.x); k += BIT_X_SPIN) {
                                        fprintf(fp, "%1llX",  (__1st.x >> k) & SPIN_MASK);
                                        fprintf(fp, "%1llX",  (__2nd.x >> k) & SPIN_MASK);
                                }
                                for(int k = 0; k < 8*sizeof(__1st.y); k += BIT_X_SPIN) {
                                        fprintf(fp, "%1llX",  (__1st.y >> k) & SPIN_MASK);
                                        fprintf(fp, "%1llX",  (__2nd.y >> k) & SPIN_MASK);
                                }
                        }
                        fprintf(fp, "\n");
                }
                fclose(fp);
        }
#ifdef USE_MNNVL
	free(black_h);
	free(white_h);
#endif
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
                "\t\tSpecifies the horizontal dimension of the entire  lattice  (black+white  spins).\n"
                "\t\tThis dimension must be a multiple of %2$d.\n"
                "\n"
                "\t-y|--y <VERT_DIM>\n"
                "\t\tSpecifies the vertical dimension of the per-GPU lattice.  This dimension must be\n"
                "\t\ta multiple of %3$d.\n"
                "\n"
                "\t-n|--n <NSTEPS>\n"
                "\t\tSpecifies the number of iteration to run.\n"
                "\t\tDefualt: %4$d\n"
                "\n"
                "\t-g|--gpus <NUM_DEVICES>\n"
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
                "\t-t|--temp <TEMP_0>[[,<IT_1>:<TEMP_1>]...]\n"
                "\t\tSpecifies the temperature(s), in absolute units. If both this option and '-a' are\n"
                "\t\tspecified then this option is used.\n"
                "\t\tDefault: %7$f\n"
                "\n"
                "\t-p|--print <STAT_FREQ>\n"
                "\t\tSpecifies the frequency, in no.  of  iteration,  with  which  the  magnetization\n"
                "\t\tstatistics is printed. If this option is used with --pexp, this option is ignored.\n"
                "\t\tDefault: only at the beginning and at end of the simulation\n"
                "\n"
                "\t--pexp\n"
                "\t\tPrints statistics every power-of-2 time steps.  This  option  overrides  the  -p\n"
                "\t\toption.\n"
                "\t\tDefault: disabled\n"
                "\n"
                "\t-c|--corr <CORR_FILE_PATH>\n"
                "\t\tEnables correlation and writes to file CORR_FILE_PATH  the  correlation of  each\n"
                "\t\tpoint with the vertical and  orizontal  neighbors at distance r <= %8$d.   Beyond\n"
                "\t\tthat, distance as chosen according to an  exponential rule, with %9$d  values  per\n"
                "\t\tpower of 2.  The  correlation  is  computed  every  time  the  magnetization  is\n"
                "\t\tprinted on screen (based  on  either  the  '-p'  or  '-e'  options)  and  it  is\n"
                "\t\twritten in the  file one line per measure.\n"
                "\t\tDefault: full correlation (see --corrfull option)\n"
                "\n"
                "\t--corrfull\n"
                "\t\tCompute the correlation for each spin in the system.\n"
                "\n"
                "\t--corrdiag\n"
                "\t\tCompute the correlation only for diagonal spins.\n"
                "\n"
                "\t--corrchkb\n"
                "\t\tComputes the correlation for only one spin (the top-left one)  for each block of\n"
                "\t\t%10$dx%10$d spins (checkerboard pattern).\n"
                "\n"
                "\t--corrmixd\n"
                "\t\tComputes the correlation using a mix of full and checkerboard modes.   The  full\n"
                "\t\tcorrelation is used for  all distances  r <= %12$d. Then,  for each spin in a %11$dx%11$d\n"
                "\t\tsquare, it is computed for each r > %12$d.\n"
                "\n"
                "\t--writechkp <CHECKPOINT_FILE_PATH>\n"
                "\t\tEnables write of checkpoint file at the end of the simulation.  The file can  be\n"
                "\t\tlater used to resume the simulation with the '-r' option.  This option and  '-r'\n"
                "\t\tcan be used together to break down a  large  run  into  multiple  smaller  runs.\n"
                "\n"
                "\t--readchkp <CHECKPOINT_FILE_PATH>\n"
                "\t\tEnables the restart of a simulation from the state in a checkpoint file.  Please\n"
                "\t\tnote that in order for that to work, the non-checkpoint  command  lines  options\n"
                "\t\tused in the run where the checkpoint file was created must match with those used\n"
                "\t\tin the run where the checkpoint file is read.  This option and '-r' can be  used\n"
                "\t\ttogether  to  break   down   a   large   run   into   multiple   smaller   runs.\n"
                "\n"
                "\t-o|--o\n"
                "\t\tEnables the file dump of  the lattice  every time  the magnetization is printed.\n"
                "\t\tDefault: off\n\n",
                bname,
                SPIN_X_WORD*2*BLOCK_X*BREAD_X,
                BLOCK_Y*BREAD_Y,
                NUMIT_DEF,
                SEED_DEF,
                ALPHA_DEF,
                ALPHA_DEF*CRIT_TEMP,
                CORR_ALLR_MAX,
                CORR_EXPR_PP2,
                CORR_CHKB_SIDE,
                CORR_MIXD_THRS,
                CORR_MIXD_THRS*2);
        exit(EXIT_SUCCESS);
}

#ifdef __ARM_ARCH

using Float128 = long double;
#define powq            powl
#define roundq          roundl
#define __F128(x)       (x##L)
#define __F128_SNPRINTF snprintf
#define __F128_SPEC     "%.*Lf"
#define __F128_DIG      __LDBL_DIG__

#else

#include <quadmath.h>
using Float128 = __float128;
#define __F128(x)       (x##Q)
#define __F128_SNPRINTF quadmath_snprintf
#define __F128_SPEC     "%.*Qf"
#define __F128_DIG      FLT128_DIG

#endif

//#define RNDTBL_DEBUG

template<typename TABLE_T>
void createBitRndTable(double p1, const int TBL_LOG2_LEN, TABLE_T *table) {

        static_assert(std::is_integral<TABLE_T>::value &&
                      std::is_unsigned<TABLE_T>::value);

#if defined(RNDTBL_DEBUG)
        char qstr[256];
#endif
        const Float128 prob1 = p1;
        const Float128 prob0 = Float128(1.0)-prob1;

#ifdef RNDTBL_DEBUG
        printf("\n");
        printf("prob0: %lf\n", 1.0-p1);
        printf("prob1: %lf\n", p1);
#endif
        if (TBL_LOG2_LEN != 4 && TBL_LOG2_LEN != 8) {
                fprintf(stderr,
                        "%s:%d: error, TBL_LOG2_LEN must be either 4 or 8 (%d)!\n",
                        __func__, __LINE__, TBL_LOG2_LEN);
                exit(EXIT_FAILURE);

        }
        
        const Float128 MAX_VAL = (sizeof(TABLE_T) == 4) ? __F128(4294967296.) : __F128(18446744073709551616.);

#ifdef RNDTBL_DEBUG
        __F128_SNPRINTF(qstr, sizeof(qstr), __F128_SPEC, __F128_DIG, MAX_VAL);
        printf("\nMAX_VAL: %s\n\n", qstr);
#endif
        const int TABLE_LEN = 1 << TBL_LOG2_LEN;

        Float128 table_q[TABLE_LEN];
        
        //printf("QDBL_DIG: %d\n", FLT128_DIG);

        Float128 ptot = 0;
        for(unsigned int i = 0; i < TABLE_LEN; i++) {

                const int n1 = __builtin_popcount(i);
                const int n0 = TBL_LOG2_LEN - n1;

                const Float128 p1 = powq(prob1, n1);
                const Float128 p0 = powq(prob0, n0);

                table_q[i] = p0*p1;

#ifdef RNDTBL_DEBUG
                __F128_SNPRINTF(qstr, sizeof(qstr), __F128_SPEC, __F128_DIG, table_q[i]);
                printf("table[%3d]: %s\n", i, qstr);
#endif
                ptot += p0*p1;
        }

        Float128 last = table_q[0];
        table_q[0] = 0;
        for(unsigned int i = 1; i < TABLE_LEN; i++) {
                Float128 save = table_q[i];
                table_q[i] = table_q[i-1] + last;
                last = save;
        }

        for(unsigned int i = 0; i < TABLE_LEN; i++) {

                table[i] = roundq(table_q[i]*MAX_VAL);

                if (i && table[i] == table[i-1]) {
                        unsigned long long table_i   = table[i];
                        printf("%s:%d: WARNING, quad precision not enough with current temp: table[%d] = table[%d] = %10llu\n",
                                __func__, __LINE__, i-1, i, table_i);
                }
        }
#ifdef RNDTBL_DEBUG
        __F128_SNPRINTF(qstr, sizeof(qstr), __F128_SPEC, __F128_DIG, ptot);
        printf("\nptot: %s\n\n", qstr);
        
        for(int i = 0; i < TABLE_LEN; i++) {

                unsigned long long table_i   = table[i];
                unsigned long long table_ip1 = (i < TABLE_LEN-1) ? table[i+1] : TABLE_T(-1);

                __F128_SNPRINTF(qstr, sizeof(qstr), __F128_SPEC, __F128_DIG, table_q[i]);
                printf("table[%3d]: %s, %20llu [nvals: %20llu]\n",
                        i, qstr, table_i, table_ip1 - table_i + (i==TABLE_LEN-1));
        }
        printf("\n");
#endif
}

void printTime(int width, double secs) {

        const char *unit[] = {"s",  "m",    "h",     "d",       "M",        "y"};
        double       div[] = {1.0, 60.0, 3600.0, 86400.0, 2678400.0, 32140800.0};

        int i;
        for(i = 0; i < sizeof(div)/sizeof(div[0]); i++) {
                if (secs < div[i]) break;
        }

        i = i ? i-1 : i;

        int rank = 0;
#ifdef USE_MNNVL
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        PRINTF1("%*.2lf%s\n", width-1, secs/div[i], unit[i]);

        return;
}

void writeConfig(const char *fname,
                 dim3 grid,
                 dim3 block,
                 unsigned long long seed,
                 long long printExp,
                 double printExpCand,
                 long long it,
                 int ndev,
                 size_t lld,
                 int Y,
                 int X,
                 ulonglong2 *black_m,
		 ulonglong2 *white_m) {

        //char fname[256];
        //snprintf(fname, sizeof(fname), "%s/dump_it%lld_seed%llu.bin", dirname, it, seed);

        FILE *fp = Fopen(fname, "w");

        Fwrite(&grid.x, sizeof(grid.x), 1, fp);
        Fwrite(&grid.y, sizeof(grid.y), 1, fp);

        Fwrite(&block.x, sizeof(block.x), 1, fp);
        Fwrite(&block.y, sizeof(block.y), 1, fp);
        
        Fwrite(&seed, sizeof(seed), 1, fp);
        
        Fwrite(&printExp, sizeof(printExp), 1, fp);         // probably unnecessary
        Fwrite(&printExpCand, sizeof(printExpCand), 1, fp);

        Fwrite(&it, sizeof(it), 1, fp);

        Fwrite(&lld, sizeof(lld), 1, fp);
        Fwrite(&Y, sizeof(Y), 1, fp);
        Fwrite(&X, sizeof(X), 1, fp); // unnecessary
#ifdef USE_MNNVL
	int rank  = 0;
	int ntask = 1;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ntask);

	const long long llenGpu = lld*Y;

	ulonglong2 *black_h = (ulonglong2 *)Malloc(sizeof(*black_h)*llenGpu);
        ulonglong2 *white_h = (ulonglong2 *)Malloc(sizeof(*white_h)*llenGpu);

	for(int d = 0; d < ndev; d++) {
		CHECK_CUDA(cudaSetDevice(d));

		CHECK_CUDA(cudaMemcpy(black_h, black_m + rank*ndev*llenGpu + d*llenGpu, sizeof(*black_h)*llenGpu, cudaMemcpyDeviceToHost));
		CHECK_CUDA(cudaMemcpy(white_h, white_m + rank*ndev*llenGpu + d*llenGpu, sizeof(*white_h)*llenGpu, cudaMemcpyDeviceToHost));
        	
		Fwrite(black_h, sizeof(*black_h)*llenGpu, 1, fp);
	        Fwrite(white_h, sizeof(*white_h)*llenGpu, 1, fp);
	}
	free(black_h);
	free(white_h);
#else
        Fwrite(black_m, sizeof(*black_m)*ndev*Y*lld, 1, fp);
        Fwrite(white_m, sizeof(*white_m)*ndev*Y*lld, 1, fp);
#endif
        fclose(fp);
        return;
}
                
void readConfig(const char *fname,
                dim3 grid,
                dim3 block,
                long long printExp,
                int ndev,
                size_t lld,
                int Y,
                int X,
                double *printExpCand,
                unsigned long long *seed,
                long long *it,
                ulonglong2 *black_m,
		ulonglong2 *white_m) {

        FILE *fp = Fopen(fname, "r");

        dim3 gridRead;
        Fread(&gridRead.x, sizeof(gridRead.x), 1, fp);
        Fread(&gridRead.y, sizeof(gridRead.y), 1, fp);
        if (gridRead.x != grid.x || gridRead.y != grid.y) {
                fprintf(stderr,
                        "%s:%d: error, config in file %s requires a different grid "
                        "(%u, %u) than that specified for the current run (%u, %u)!\n",
                        __func__, __LINE__, fname, gridRead.x, gridRead.y, grid.x, grid.y);
                exit(EXIT_FAILURE);
        }

        dim3 blockRead;
        Fread(&blockRead.x, sizeof(blockRead.x), 1, fp);
        Fread(&blockRead.y, sizeof(blockRead.y), 1, fp);
        if (blockRead.x != block.x || blockRead.y != block.y) {
                fprintf(stderr,
                        "%s:%d: error, config in file %s requires a different block "
                        "(%u, %u) than that specified for the current run (%u, %u)!\n",
                        __func__, __LINE__, fname, blockRead.x, blockRead.y, block.x, block.y);
                exit(EXIT_FAILURE);
        }
        
        Fread(seed, sizeof(*seed), 1, fp);
        
        long long dummyPrintExp;
        Fread(&dummyPrintExp, sizeof(dummyPrintExp), 1, fp);

        Fread(printExpCand, sizeof(*printExpCand), 1, fp);
        
        Fread(it, sizeof(*it), 1, fp);

        size_t lldRead;
        Fread(&lldRead, sizeof(lldRead), 1, fp);
        if (lldRead != lld) {
                fprintf(stderr,
                        "%s:%d: error, config in file %s requires a different lld "
                        "(%zu) than that specified for the current run (%zu)!\n",
                        __func__, __LINE__, fname, lldRead, lld);
                exit(EXIT_FAILURE);
        }
        
        int YRead;
        int XRead;
        Fread(&YRead, sizeof(YRead), 1, fp);
        Fread(&XRead, sizeof(XRead), 1, fp); // unnecessary
        if (YRead != Y || XRead != X) {
                fprintf(stderr,
                        "%s:%d: error, config in file %s requires a different system size "
                        "per GPU (Y: %d, X: %d) than that specified for the current run (Y: %d, X: %d)!\n",
                        __func__, __LINE__, fname, YRead, XRead, Y, X);
                exit(EXIT_FAILURE);
        }

#ifdef USE_MNNVL
	int rank  = 0;
	int ntask = 1;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ntask);

	const long long llenGpu = lld*Y;

	ulonglong2 *black_h = (ulonglong2 *)Malloc(sizeof(*black_h)*llenGpu);
        ulonglong2 *white_h = (ulonglong2 *)Malloc(sizeof(*white_h)*llenGpu);

	for(int d = 0; d < ndev; d++) {
		CHECK_CUDA(cudaSetDevice(d));
        
		Fread(black_h, sizeof(*black_h)*llenGpu, 1, fp);
      		Fread(white_h, sizeof(*white_h)*llenGpu, 1, fp);

		CHECK_CUDA(cudaMemcpy(black_m + rank*ndev*llenGpu + d*llenGpu, black_h, sizeof(*black_h)*llenGpu, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(white_m + rank*ndev*llenGpu + d*llenGpu, white_h, sizeof(*white_h)*llenGpu, cudaMemcpyHostToDevice));
	}
	free(black_h);
	free(white_h);
#else
        Fread(black_m, sizeof(*black_m)*ndev*Y*lld, 1, fp);
        Fread(white_m, sizeof(*white_m)*ndev*Y*lld, 1, fp);
#endif

        fclose(fp);
        return;
}

int parseMutliTempOpt(char *optarg,
		      long long *tempStart,
		      double *temps) {

	if (!optarg || !tempStart || !temps) {
		fprintf(stderr,
			"%s:%d: error, one ore more void parameters!\n",
			__func__, __LINE__);
		exit(EXIT_FAILURE);
	}

	int ntemps = 0;
	tempStart[0] = 1ll;

	while(ntemps < MAX_TEMPS) {

		char *tok = NULL;

		// first temperature has no starting 
		// iteration (implicitly starts from it 1)
		if (ntemps > 0) {
			tok = strtok(NULL, ":");

			if (!tok) {
			       break;
			} else if (strstr(tok, ",")) {
				fprintf(stderr, "Error, wrong multi-temperature format!\n");
				exit(EXIT_FAILURE);
			}

			tempStart[ntemps] = atoll(tok);
		}
		tok = strtok(ntemps ? NULL : optarg, ",");

		if (!tok || strstr(tok, ":")) {
			fprintf(stderr, "Error, wrong multi-temperature format!\n");
			exit(EXIT_FAILURE);
		}

		temps[ntemps++] = atof(tok);
	}
	return ntemps;
}

int main(int argc, char **argv) {

        ulonglong2 *black_m=NULL;
        ulonglong2 *white_m=NULL;

        constexpr int SPIN_X_WORD = (8*sizeof(*black_m)) / BIT_X_SPIN;

        int X = 0;
        int Y = 0;

        int dumpOut = 0;

        //char cname[256];

        int corrType = CORR_FULL;

        int corrOut = 0;
        char *corrFpath = NULL;

        long long it = 0;
        long long lastIt = 0;
        long long numIt = NUMIT_DEF;

        unsigned long long seed = SEED_DEF;

        int ndevTot = 1;
        int ndevLoc = 1; // no. of devices per process; when not using MNNVL it's total number of devices

        double alpha = ALPHA_DEF;
	int ntemps = 1;
	long long tempStart[MAX_TEMPS] = {1};
        double temps[MAX_TEMPS] = {-1.0f};

        long long printFreq = 0;
        int printExp = 0;
        double printExpCand = 1;
        double printExpFact = pow(2., 0.125);
        long long printExpLast = 0;

        char *readChkpFpath = NULL;
        char *writeChkpFpath = NULL;

	int rank  = 0;
	int ntask = 1;
#ifdef USE_MNNVL
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ntask);
#endif
        int och;
        while(1) {
                int option_index = 0;
                static struct option long_options[] = {
                        {        "x", required_argument, 0, 'x'},
                        {        "y", required_argument, 0, 'y'},
                        {      "nit", required_argument, 0, 'n'},
                        {     "seed", required_argument, 0, 's'},
                        {      "out",       no_argument, 0, 'o'},
                        {     "gpus", required_argument, 0, 'g'},
                        {    "alpha", required_argument, 0, 'a'}, // remove?
                        {     "temp", required_argument, 0, 't'},
                        {    "print", required_argument, 0, 'p'},
                        {     "pexp",       no_argument, 0,   1},
                        {     "corr", required_argument, 0, 'c'},
                        { "corrfull",       no_argument, 0,   2},
                        { "corrdiag",       no_argument, 0,   3},
                        { "corrchkb",       no_argument, 0,   4},
                        { "corrmixd",       no_argument, 0,   5},
                        { "readchkp", required_argument, 0, 'r'},
                        {"writechkp", required_argument, 0, 'w'},
                        {     "help",       no_argument, 0, 'h'},
                        {          0,                 0, 0,   0}
                };

                och = getopt_long(argc, argv, "x:y:n:ohs:d:a:t:p:c:r:g:S:w:r:", long_options, &option_index);
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
                                numIt = atoll(optarg);
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
                        case 'g':
                                ndevLoc = atoi(optarg);
                                break;
                        case 'a':
                                alpha = atof(optarg);
                                break;
                        case 't':
                                if (!strstr(optarg, ",")) {
					ntemps = 1;
					temps[0] = atof(optarg);
				} else {
#if defined(USE_UPDATE_LUT_K)
					ntemps = parseMutliTempOpt(optarg, tempStart, temps);
#else
					fprintf(stderr, "Error, multi-temperature prototocol only available for LUT kernel!\n");
					exit(EXIT_FAILURE);
#endif
				}
                                break;
                        case 'p':
                                printFreq = atoll(optarg);
                                break;
                        case 'c':
                                corrOut = 1;
				corrFpath = strndup(optarg, 256);
                                break;
                        case 1:
                                printExp = 1;
                                break;
                        case 2:
                                corrType = CORR_FULL;
                                break;
                        case 3:
                                corrType = CORR_DIAG;
                                break;
                        case 4:
                                corrType = CORR_CHKB;
                                break;
                        case 5:
                                corrType = CORR_MIXD;
                                break;
                        case 'r':
                                readChkpFpath = strndup(optarg, 256);
                                break;
                        case 'w':
                                writeChkpFpath = strndup(optarg, 256);
                                break;
                        case '?':
                                exit(EXIT_FAILURE);
                        default:
                                fprintf(stderr, "unknown option: %c\n", och);
                                exit(EXIT_FAILURE);
                }
        }

	ndevTot = ndevLoc*ntask;

	if (ndevLoc < 1) {
                printf("Error, number of GPUs must be positive\n");
                exit(EXIT_FAILURE);
        }

	int totDevLoc;
	CHECK_CUDA(cudaGetDeviceCount(&totDevLoc));

	assert(totDevLoc <= MAX_GPU);

        if (ndevLoc > totDevLoc) {
                printf("Error, too many GPUs specified. Only %d found\n", totDevLoc);
                exit(EXIT_FAILURE);
        }

#ifdef USE_MNNVL
	if (readChkpFpath) {

		char *p = strstr(readChkpFpath, "%");
		char *lastc = readChkpFpath + strlen(readChkpFpath)-1;

		if (!p                           ||
                    p >= lastc                   ||
                    (p[1] != 'd' && p[1] != 'i') || 
                    strstr(p+1, "%")) {

			fprintf(stderr,
				"Error, file name for checkpoint read must "
				"contain one of either '%%d' of '%%i' as "
				"rank specifier!\n");
			exit(EXIT_FAILURE);
		}
		char *newname = (char *)Malloc(sizeof(*newname)*256);
		snprintf(newname, 256, readChkpFpath, rank);

		free(readChkpFpath);
		readChkpFpath = newname;
	}
	if (writeChkpFpath) {

		char *p = strstr(writeChkpFpath, "%");
		char *lastc = writeChkpFpath + strlen(writeChkpFpath)-1;

		if (!p                           ||
                    p >= lastc                   ||
                    (p[1] != 'd' && p[1] != 'i') || 
                    strstr(p+1, "%")) {

			fprintf(stderr,
				"Error, file name for checkpoint write must "
				"contain one of either '%%d' of '%%i' as "
				"rank specifier!\n");
			exit(EXIT_FAILURE);
		}
		char *newname = (char *)Malloc(sizeof(*newname)*256);
		snprintf(newname, 256, writeChkpFpath, rank);

		free(writeChkpFpath);
		writeChkpFpath = newname;
	}
#endif

        if (!X || !Y) {
                if (!X) {
                        if (Y && !(Y % (2*SPIN_X_WORD*2*BLOCK_X*BREAD_X))) {
                                X = Y;
                        } else {
                                X = 2*SPIN_X_WORD*2*BLOCK_X*BREAD_X;
                        }
                }
                if (!Y) {
                        if (!(X%(BLOCK_Y*BREAD_Y))) {
                                Y = X;
                        } else {
                                Y = BLOCK_Y*BREAD_Y;
                        }
                }
        }

        if (!X || (X%2) || ((X/2)%(SPIN_X_WORD*BLOCK_X*BREAD_X))) {
                fprintf(stderr, "\nPlease specify an X dim multiple of %d\n\n", 2*SPIN_X_WORD*BLOCK_X*BREAD_X);
                usage(SPIN_X_WORD, argv[0]);
                exit(EXIT_FAILURE);
        }
        if (!Y || (Y%(BLOCK_Y*BREAD_Y))) {
                fprintf(stderr, "\nPlease specify a Y dim multiple of %d\n\n", BLOCK_Y*BREAD_Y);
                usage(SPIN_X_WORD, argv[0]);
                exit(EXIT_FAILURE);
        }

	if ( (X & (X-1)) || (Y & (Y-1)) ) {
		fprintf(stderr, "\nPlease specify power-of-2 values for X and Y!\n");
		exit(EXIT_FAILURE);
	}

        if (temps[0] == -1.0) {
                temps[0] = alpha*CRIT_TEMP;
        }

        if (printExp) {
                printFreq = 1;
        }

	// It's clearer to replicate the print loop code rather than
	// use many ifdefs around the parts only relevant to USE_MNNVL
#ifdef USE_MNNVL
	// gather nodes data
	char (*procNames)[MPI_MAX_PROCESSOR_NAME] = (char (*)[MPI_MAX_PROCESSOR_NAME])Malloc(sizeof(*procNames)*ntask);
	int nameLen;
	MPI_Get_processor_name(procNames[rank], &nameLen);
	MPI_Gather(procNames[rank],
		   MPI_MAX_PROCESSOR_NAME,
		   MPI_CHAR,
		   procNames,
		   MPI_MAX_PROCESSOR_NAME,
		   MPI_CHAR,
		   0,
		   MPI_COMM_WORLD);

	cudaDeviceProp *props = (cudaDeviceProp *)Malloc(sizeof(*props)*ndevLoc);
	for(int i = 0; i < ndevLoc; i++) {
		CHECK_CUDA(cudaGetDeviceProperties(props+i, i));
	}

	cudaDeviceProp *props_all = NULL;
	if (!rank) {
		props_all = (cudaDeviceProp *)Malloc(sizeof(*props_all)*ntask*ndevLoc);
	}

	MPI_Datatype MPI_DEV_PROP;
	MPI_Type_contiguous(sizeof(cudaDeviceProp), MPI_BYTE, &MPI_DEV_PROP);
	MPI_Type_commit(&MPI_DEV_PROP);

	MPI_Gather(props, ndevLoc, MPI_DEV_PROP, props_all, ndevLoc, MPI_DEV_PROP, 0, MPI_COMM_WORLD);
        
	if (!rank) {
		printf("\nUsing GPUs:\n");
		for(int i = 0; i < ndevTot; i++) {
			printf("\t%2d (%s, %zu GB, %d SMs, %d th/SM max, CC %d.%d, ECC %s) on %-22s rank %2d\n",
					i, props_all[i].name,
					props_all[i].totalGlobalMem / (1<<30),
					props_all[i].multiProcessorCount,
					props_all[i].maxThreadsPerMultiProcessor,
					props_all[i].major, props_all[i].minor,
					props_all[i].ECCEnabled?"on":"off",
					procNames[i/ndevLoc], i/ndevLoc);
		}
		printf("\n");
	}
	free(props);
	free(props_all);
	free(procNames);
#else
        cudaDeviceProp props;

        printf("\nUsing GPUs:\n");
        for(int i = 0; i < ndevLoc; i++) {
                CHECK_CUDA(cudaGetDeviceProperties(&props, i));
                printf("\t%2d (%s, %zu GB, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n",
                        i, props.name, 
                        DIV_UP(props.totalGlobalMem, ONE_GB),
                        props.multiProcessorCount,
                        props.maxThreadsPerMultiProcessor,
                        props.major, props.minor,
                        props.ECCEnabled?"on":"off");
        }
        printf("\n");
#endif
        fflush(stdout);

#ifndef USE_MNNVL
        for(int i = 0; i < ndevLoc; i++) {
                int attrMM = 0;;
                CHECK_CUDA(cudaDeviceGetAttribute(&attrMM, cudaDevAttrManagedMemory, i));
                if (!attrMM) {
                        printf("Error, GPU %d does not support managed memory!\n", i);
                        exit(EXIT_FAILURE);
                }
        }
#endif

        //const char *corr_str[CORR_NUM] = {"full", "diag", "chkb", "mixd"};
#ifdef USE_MNNVL
	if (!rank) {
#endif
        if(corrOut) {
                //snprintf(cname,
                //         sizeof(cname),
                //         "corr_%s_%dx%d_temp%.15G_seed%llu",
                //         corr_str[corrType],
                //        ndev*Y, X, temps[0], seed);

                // On Mon 15/7/24 we changed from this rule:
                //
                // * for all points compute corr with r <= 32 (all r) then for one spin per 16x16
                //   square compute corr with r > 32 (all r up to 256, then log rule 2^{x/32} kicks in)
                //
                // to this rule:
                //
                // * for all points, compute corr with r=0,1,...,32 then for one spin per 16x16 square
                //   compute corr with r=33, 34,..., r_cut_off(t) and after that using the log rule 2^{x/32}
                //
                // with    r_cut_off(t) = max{256, (int) (g(L)*sqrt(t)+0.5)} } // with g(2**16) = 6 and g(2**20) = 6.72
                // 
                // so we'll compute nsamp_corr/corr_samp_m[] inside computeCorr() based on the iteration number
                if (!readChkpFpath) {
                        Remove(corrFpath);
                }
        }
#ifdef USE_MNNVL
	}
#endif
        const size_t lld = (X/2)/SPIN_X_WORD;         // leading dimension of the spin lattice
        //const size_t llenLoc = size_t(Y)*lld;       // length of a single buffer per GPU
        const size_t llenGpu = size_t(Y)*lld;         // length of a single buffer per GPU
        const size_t llenLoc = 2ull*ndevLoc*llenGpu;  // length of the buffer held by all the GPUs local to the process
        const size_t llenTot = llenLoc*ntask;         // total length of the buffer held by all the GPUs of to the process

        size_t nspinTot = llenTot*SPIN_X_WORD;

	assert(0 == (lld % (BLOCK_X*BREAD_X)));
	assert(0 == (  Y % (BLOCK_Y*BREAD_Y)));

	// moved to a 1D grid because with very large
	// systems, the 2nd dimension can be > 65535,
	// the maximum value allowed for a grid
	dim3 grid((lld / (BLOCK_X*BREAD_X))*
		  (  Y / (BLOCK_Y*BREAD_Y)));

        dim3 block(BLOCK_X, BLOCK_Y);

#ifndef USE_MNNVL
        // allocate global buffer
        CHECK_CUDA(cudaMallocManaged(&black_m, sizeof(*black_m)*llenLoc, cudaMemAttachGlobal));

        white_m = black_m + llenLoc/2;
#else
	vmmAllocCtx_t *vmmctx_w = vmmFabricMalloc((void **)&black_m, sizeof(*black_m) * llenGpu);
	vmmAllocCtx_t *vmmctx_b = vmmFabricMalloc((void **)&white_m, sizeof(*white_m) * llenGpu);
#ifdef USE_MEMBRARIER

	size_t minFabAlloc = vmmFabricGranularity(0); // = 262144

	long long *flagB_large_m = NULL;
	long long *flagW_large_m = NULL;

	// add call to get granularity...

	vmmAllocCtx_t *vmmctx_flagB = vmmFabricMalloc((void **)&flagB_large_m, sizeof(*flagB_large_m) * minFabAlloc); // hackish for now, size must be mult. of granularity, i.e. 2097152 bytes
	vmmAllocCtx_t *vmmctx_flagW = vmmFabricMalloc((void **)&flagW_large_m, sizeof(*flagW_large_m) * minFabAlloc);

	// only the memories of the first and
	// last local device are used per node

	long long **flagPrevB_m = (long long **)Malloc(sizeof(*flagPrevB_m)*ndevLoc);
	long long **flagCurrB_m = (long long **)Malloc(sizeof(*flagCurrB_m)*ndevLoc);
	long long **flagNextB_m = (long long **)Malloc(sizeof(*flagNextB_m)*ndevLoc);

	long long **flagPrevW_m = (long long **)Malloc(sizeof(*flagPrevW_m)*ndevLoc);
	long long **flagCurrW_m = (long long **)Malloc(sizeof(*flagCurrW_m)*ndevLoc);
	long long **flagNextW_m = (long long **)Malloc(sizeof(*flagNextW_m)*ndevLoc);

	for(int i = 0; i < ndevLoc; i++) {

		int prevDev = (rank*ndevLoc + i-1 + ndevTot) % ndevTot;
		int currDev =  rank*ndevLoc + i;
		int nextDev = (rank*ndevLoc + i+1 + ndevTot) % ndevTot;

		flagPrevB_m[i] = flagB_large_m + prevDev*minFabAlloc;
		flagCurrB_m[i] = flagB_large_m + currDev*minFabAlloc;
		flagNextB_m[i] = flagB_large_m + nextDev*minFabAlloc;

		flagPrevW_m[i] = flagW_large_m + prevDev*minFabAlloc;
		flagCurrW_m[i] = flagW_large_m + currDev*minFabAlloc;
		flagNextW_m[i] = flagW_large_m + nextDev*minFabAlloc;
	}
#endif

        MPI_Barrier(MPI_COMM_WORLD);
#endif
        // read checkpoint configuration, if necessary
        if (readChkpFpath) {
                PRINTF1("Reading checkpoint from file(s) %s... ", readChkpFpath);
#ifdef USE_MNNVL
		MPI_Barrier(MPI_COMM_WORLD);
#endif
                double wt = Wtime();
                readConfig(readChkpFpath, grid, block, printExp, ndevLoc, lld, Y, X, &printExpCand, &seed, &lastIt, black_m, white_m);
#ifdef USE_MNNVL
		MPI_Barrier(MPI_COMM_WORLD);
#endif
                wt = Wtime()-wt;
                PRINTF1("done in %lf secs\n\n", wt);

                // "lastIt" contains the last executed iteration, so we'll start from lastIt+1

                if (printExp) {
                        printFreq = max(lastIt+1, (long long)(0.5 + printExpCand));

                        // this to compute printExpCand based on lastIt
                        //printExpCand = pow(printExpFact, round(log(lastIt) / log(printExpFact)) + 1.0);
                        //printf("printExpCand: %lf\n", printExpCand);
                        //printFreq = max(lastIt+1, (long long)(0.5 + printExpCand));
                }
        }
        assert(lastIt >= 0);

	if (!rank) {
		printf("Run configuration:\n");
		printf("\tword size: %zu\n", sizeof(black_m[0]));
		printf("\tbits per spin: %d (mask: 0x%1X)\n", BIT_X_SPIN, SPIN_MASK);
		printf("\tspins/word: %d\n", SPIN_X_WORD);
		printf("\tspins: %zu (~%.2E)\n", nspinTot, double(nspinTot));
		printf("\tseed: %llu\n", seed);
		printf("\tblock size (X, Y): %d, %d\n", block.x, block.y);
		printf("\ttile  size (X, Y): %d, %d\n", BLOCK_X*BREAD_X, BLOCK_Y*BREAD_Y);
		printf("\tgrid size 1D: %u\n", grid.x);
		printf("\tvirtual grid size 2D (X, Y): %lu, %u\n", lld / (BLOCK_X*BREAD_X), Y / (BLOCK_Y*BREAD_Y));
		printf("\tspins per tile (X, Y): %d, %d\n", BLOCK_X*BREAD_X*SPIN_X_WORD, BLOCK_Y*BREAD_Y*SPIN_X_WORD);
		
		printf("\n");
		printf("\titerations:\n");
		printf("\t\tbeg: %lld\n", lastIt+1);
		printf("\t\tend: %lld\n", lastIt+numIt);
		printf("\t\ttot: %lld\n", numIt);

		printf("\n");
		if (printExp) { printf("\tprint stats according to an exp with factor %lf\n", printExpFact);  }
		else          { printf("\tprint stats every %lld steps\n", !printFreq ? numIt : printFreq); }
		if (corrOut) {
			printf("\tcompute corr. at every stats output from ");
			switch(corrType) {
					case CORR_FULL: printf("each spin\n"); break;
					case CORR_DIAG: printf("diagonal spins only\n"); break;
					case CORR_CHKB: printf("one spin per %1$dx%1$d square\n", CORR_CHKB_SIDE); break;
					case CORR_MIXD: printf("each spin up to r <= %1$d, and one spin per %2$dx%2$d square for r > %1$d\n", CORR_MIXD_THRS*2, CORR_MIXD_THRS); break;
			}
			printf("\tcorrelation output written to file: %s\n", corrFpath);
		}
		if (ntemps == 1) {
			printf("\ttemperature: %.15G (%.15G*T_crit)\n", temps[0], temps[0]/CRIT_TEMP);
		} else {
			printf("\tmixed temperature protocol enabled:\n");
			for(int i = 0; i < ntemps-1; i++) {
				printf("\t\tfrom iter %10lld to %10lld: temperature %.15G\n", tempStart[i], tempStart[i+1]-1, temps[i]);
			}
			printf("\t\tfrom iter %10lld to %10c: temperature %.15G\n", tempStart[ntemps-1], '*', temps[ntemps-1]);
		}

		printf("\n");
		printf("\tno. of  processes: %d\n", ntask);
		printf("\tGPUs  per process: %d\n", ndevLoc);
		printf("\ttotal no. of GPUs: %d\n", ndevTot);
		printf("\tGPUs  memory type: %s\n",
#ifndef USE_MNNVL
			"managed");
#else
			"fabric");
#endif
		printf("\n");
		printf("\tper-GPU lattice size:      %8d x %8d spins\n", Y, X);
		printf("\tper-GPU lattice shape: 2 x %8d x %8zu ull2s (%12zu total)\n", Y, lld, llenGpu*2);
		printf("\n");
#ifdef USE_MNNVL
		printf("\tper-proc lattice size:      %8d x %8d spins\n", ndevLoc*Y, X);
		printf("\tper-proc lattice shape: 2 x %8d x %8zu ull2s (%12zu total)\n", ndevLoc*Y, lld, llenLoc);
		printf("\n");
#endif
		printf("\ttotal lattice size:      %8d x %8d spins\n", ndevTot*Y, X);
		printf("\ttotal lattice shape: 2 x %8d x %8zu ull2s (%12zu total)\n", ndevTot*Y, lld, llenTot);
		printf("\n");
		printf("\ttotal memory: %.2lf GB (%.2lf GB per GPU)\n", double(llenTot*sizeof(*black_m))/ONE_GB, double(llenGpu*2*sizeof(*black_m))/ONE_GB);

#ifdef USE_UPDATE_LUT_K
		printf("\n\trandom-bit table:\n");
		printf("\t\tsize of element: %zu-bit\n", sizeof(RND_TBL_T)*8);
		printf("\t\tno. of elements: %d\n", 1<<RNDBITS_TBL_LOG2_SIZE);
		printf("\t\tbits per lookup: %d\n", RNDBITS_TBL_LOG2_SIZE);
		printf("\t\t   storage type: %s\n", KERNEL_LUT_IN_REGS ? "registers" : "shared mem");
#endif
	}

        unsigned long long sum_h[2];
        unsigned long long *sum_d[MAX_GPU];

        double et;

#ifndef USE_MNNVL
        printf("\nSetting up GPUs:\n"); fflush(stdout);
#endif

        //#pragma omp parallel for schedule(static)
        for(int i = 0; i < ndevLoc; i++) {

                et = Wtime();

                CHECK_CUDA(cudaSetDevice(i));

                CHECK_CUDA(cudaMalloc(sum_d+i,     sizeof(**sum_d)*2));
                CHECK_CUDA(cudaMemset(sum_d[i], 0, sizeof(**sum_d)*2));
#ifndef USE_MNNVL
                // set preferred loc for black/white
                CHECK_CUDA(cudaMemAdvise(black_m +  i*llenGpu, sizeof(*black_m)*llenGpu, cudaMemAdviseSetPreferredLocation, i));
                CHECK_CUDA(cudaMemAdvise(white_m +  i*llenGpu, sizeof(*white_m)*llenGpu, cudaMemAdviseSetPreferredLocation, i));

                // black boundaries up/down
                //fprintf(stderr, "v_m + %12zu + %12zu, %12zu, ..., %2d)\n", i*llenGpu,  (Y-1)*lld, lld*sizeof(*v_m), (i+ndevLoc+1)%ndevLoc);
                CHECK_CUDA(cudaMemAdvise(black_m + i*llenGpu,             sizeof(*black_m)*lld, cudaMemAdviseSetAccessedBy, (i+ndevLoc-1)%ndevLoc));
                CHECK_CUDA(cudaMemAdvise(black_m + i*llenGpu + (Y-1)*lld, sizeof(*black_m)*lld, cudaMemAdviseSetAccessedBy, (i+ndevLoc+1)%ndevLoc));

                // white boundaries up/down
                CHECK_CUDA(cudaMemAdvise(white_m + i*llenGpu,             sizeof(*white_m)*lld, cudaMemAdviseSetAccessedBy, (i+ndevLoc-1)%ndevLoc));
                CHECK_CUDA(cudaMemAdvise(white_m + i*llenGpu + (Y-1)*lld, sizeof(*white_m)*lld, cudaMemAdviseSetAccessedBy, (i+ndevLoc+1)%ndevLoc));

                if (readChkpFpath) {
                        CHECK_CUDA(cudaMemPrefetchAsync(black_m + i*llenGpu, sizeof(*black_m)*llenGpu, i, 0));
                        CHECK_CUDA(cudaMemPrefetchAsync(white_m + i*llenGpu, sizeof(*white_m)*llenGpu, i, 0));
                } else {
                        CHECK_CUDA(cudaMemset(black_m + i*llenGpu, 0, sizeof(*black_m)*llenGpu));
                        CHECK_CUDA(cudaMemset(white_m + i*llenGpu, 0, sizeof(*white_m)*llenGpu));
                }
                
		et = Wtime()-et;
                printf("\tGPU %2d done in %lf secs\n", i, et); fflush(stdout);
#endif
        }

#if !defined(USE_UPDATE_LUT_K)
        unsigned int *exp_m;
        CHECK_CUDA(cudaMallocManaged(&exp_m, sizeof(*exp_m)*2*5));

        unsigned int (*exp_h)[5] = reinterpret_cast<unsigned int (*)[5]>(exp_m);

        for(int i = 0; i < 2; i++) {
                for(int j = 0; j < 5; j++) {
                        double beta = 1.0/temps[0];
                        double de = (i?1.0:-1.0)*(j*2-4);
                        double e = exp(-2.0*beta * max(de, 0.0));
                        exp_h[i][j] = (unsigned int)(e * __2POW32M1U);
                        //printf("exp_h[%d][%d]: %10u (e: %12G)\n", i, j, exp_h[i][j], e);
                }
        }

        CHECK_CUDA(cudaMemAdvise(exp_m, sizeof(*exp_m)*2*5, cudaMemAdviseSetReadMostly, 0));
#endif

        // compute Schwinger-Dyson equation
        double *expSD_m[MAX_TEMPS];

	for(int i = 0; i < ntemps; i++) {

		CHECK_CUDA(cudaMallocManaged(&expSD_m[i], sizeof(*expSD_m[i])*5));

		for(int j = 0; j < 5; j++) {
			//expSD_m[j] = exp(-2.0*(j-4.0)*(1.0/temps[0]));
			expSD_m[i][j] = exp(-2.0*(2.0*j-4.0)*(1.0/temps[i]));
			//printf("expSD_m[%d]: %lf\n", j, expSD_m[j]);
		}
		CHECK_CUDA(cudaMemAdvise(expSD_m[i], sizeof(*expSD_m[i])*5, cudaMemAdviseSetReadMostly, 0));
	}

#if defined(USE_UPDATE_LUT_K)
        RND_TBL_T *bitTbl_m[MAX_TEMPS];

	for(int i = 0; i < ntemps; i++) {
		CHECK_CUDA(cudaMallocManaged(&bitTbl_m[i], sizeof(*bitTbl_m[i])*(1<<RNDBITS_TBL_LOG2_SIZE)));
		createBitRndTable(exp(-4.0/temps[i]), RNDBITS_TBL_LOG2_SIZE, bitTbl_m[i]);
	}
#endif
	
	long long *begYLoc = (long long *)Malloc(sizeof(*begYLoc)*ndevLoc);
	for(int i = 0; i < ndevLoc; i++) {
		begYLoc[i] = (rank*ndevLoc + i)*Y;
	}

        if (!readChkpFpath) {
                PRINTF1("\nInitializing spin lattice... ");
#ifdef USE_MNNVL
		MPI_Barrier(MPI_COMM_WORLD);
#endif
                et = Wtime();
                for(int i = 0; i < ndevLoc; i++) {

			const int devId = rank*ndevLoc + i;
			const long long begY = begYLoc[i];

                        CHECK_CUDA(cudaSetDevice(i));
                        spinInit_k<BLOCK_X, BLOCK_Y,
                                   BREAD_X, BREAD_Y,
                                   BIT_X_SPIN, C_BLACK>
                                   <<<grid, block>>>(devId, seed, 0, begY, lld, black_m);
                        CHECK_ERROR("spinInit_k");

                        spinInit_k<BLOCK_X, BLOCK_Y,
                                   BREAD_X, BREAD_Y,
                                   BIT_X_SPIN, C_WHITE>
                                   <<<grid, block>>>(devId, seed, 0, begY, lld, white_m);
                        CHECK_ERROR("spinInit_k");
                }
                for(int i = 0; i < ndevLoc; i++) {
                        CHECK_CUDA(cudaSetDevice(i));
                        CHECK_CUDA(cudaDeviceSynchronize());
                }
#ifdef USE_MNNVL
		MPI_Barrier(MPI_COMM_WORLD);
#endif
                et = Wtime()-et;
                PRINTF1("done in %lf secs\n", et);
        }
#if 0
        CHECK_CUDA(cudaDeviceSynchronize());
        dumpSpins("initial_dump", ndevLoc, Y, lld, llenGpu, black_m, white_m);
        exit(EXIT_FAILURE);
#endif
	int tempIdx = 0;
	// adjust tempIdx in case we are 
	// resarting a checkpointed run
	for(int i = 0; i < ntemps; i++) {
		if (tempStart[i] <= lastIt) {
			tempIdx = i;
		}
	}
	
	PRINTF1("\n[Switching to temperature: %.15G]\n", temps[tempIdx]);


        PRINTF1("\nRunning simulation...\n");
        PRINTF1("\n%12s %14s %14s %14s %12s %12s %12s %12s\n\n", "Step", "Magn.", "N(-1)", "N(1)", "SD value", "flips/ns", "GB/s", "ERT");

        countSpins(ndevLoc, Y, lld, black_m, white_m, sum_d, sum_h);
        double magn = abs(double(sum_h[S_POS1])-double(sum_h[S_NEG1])) / nspinTot;

        double sdv = computeSD(ndevLoc, grid, block, Y, lld, expSD_m[tempIdx], black_m, white_m);

        PRINTF1("%12lld %14E %14llu %14llu %12lf\n", lastIt, magn, sum_h[S_NEG1], sum_h[S_POS1], sdv);

        // total number of bytes read/written to
        // global mem to update one color of spins;
        // i.e. src color read, dst color read, dst color write
        unsigned long long rwbytes_upd = sizeof(*black_m)*(llenTot/2)*3 +
#if !defined(USE_UPDATE_LUT_K)
                sizeof(exp_m)*grid.x;
#else
		sizeof(*bitTbl_m[0])*(1<<RNDBITS_TBL_LOG2_SIZE)*grid.x;
#endif

        double mc_tot = 0;
        double upd_t = 0;

        long long statIt = 0;

#if defined(USE_MNNVL) && defined(USE_MEMBRARIER)
	for(int i = 0; i < ndevLoc; i++) {
		CHECK_CUDA(cudaSetDevice(i));
		setFlag_k<<<1,1>>>(lastIt, flagCurrB_m[i]);
		setFlag_k<<<1,1>>>(lastIt, flagCurrW_m[i]);
		CHECK_ERROR("setFlag_k");
		CHECK_CUDA(cudaDeviceSynchronize());
	}
	MPI_Barrier(MPI_COMM_WORLD);
#endif
        upd_t = et = Wtime();

        for(it = lastIt+1; it <= lastIt+numIt; it++) {

#if defined(USE_UPDATE_LUT_K)
		if (tempIdx < ntemps-1 && it == tempStart[tempIdx+1]) {
			tempIdx++;
			if (tempIdx > 0) { // don't print for initial tempterature
				PRINTF1("[Iteration %lld, swtitching to temperature: %.15G]\n",
					tempStart[tempIdx], temps[tempIdx]);
			}
		}
		//printf("it: %lld, currTemp = %.15G\n", it, temps[tempIdx]);
#endif
                double mc_et = Wtime();

                for(int i = 0; i < ndevLoc; i++) {
			
			const int devId = rank*ndevLoc + i;
			const long long begY = begYLoc[i];

                        CHECK_CUDA(cudaSetDevice(i));
#if !defined(USE_UPDATE_LUT_K)
                        spinUpdate_k<BLOCK_X, BLOCK_Y,
                                     BREAD_X, BREAD_Y,
                                     BIT_X_SPIN, C_BLACK>
                                     <<<grid, block>>>(devId, seed, it-1, begY, ndevTot*Y, lld, exp_m, white_m, black_m);
#else
                        spinUpdateLUT_k<BLOCK_X, BLOCK_Y,
                                        BREAD_X, BREAD_Y,
                                        BIT_X_SPIN, RNDBITS_TBL_LOG2_SIZE,
                                        C_BLACK>
                                        <<<grid, block>>>(devId, seed, it-1, begY, ndevTot*Y, lld, bitTbl_m[tempIdx], white_m, black_m);
#endif
                        CHECK_ERROR("spinUpdate_k");
#if defined(USE_MNNVL) && defined(USE_MEMBRARIER)
			if (ntask > 1) {
				if (i ==         0) setAndWaitFlag_k<<<1,1>>>(it, flagCurrB_m[i], flagPrevB_m[i]);
				if (i == ndevLoc-1) setAndWaitFlag_k<<<1,1>>>(it, flagCurrB_m[i], flagNextB_m[i]);
				CHECK_ERROR("setAndWaitFlag_k");
			}
#endif
                }
		for(int i = 0; i < ndevLoc; i++) {
			CHECK_CUDA(cudaSetDevice(i));
			CHECK_CUDA(cudaDeviceSynchronize());
		}
#if defined(USE_MNNVL) && !defined(USE_MEMBRARIER)
		if (ntask > 1) {
			MPI_Barrier(MPI_COMM_WORLD);
		}
#endif

                for(int i = 0; i < ndevLoc; i++) {
			
			const int devId = rank*ndevLoc + i;
			const long long begY = begYLoc[i];

                        CHECK_CUDA(cudaSetDevice(i));
#if !defined(USE_UPDATE_LUT_K)
                        spinUpdate_k<BLOCK_X, BLOCK_Y,
                                     BREAD_X, BREAD_Y,
                                     BIT_X_SPIN, C_WHITE>
                                     <<<grid, block>>>(devId, seed, it-1, begY, ndevTot*Y, lld, exp_m, black_m, white_m);
#else
                        spinUpdateLUT_k<BLOCK_X, BLOCK_Y,
                                        BREAD_X, BREAD_Y,
                                        BIT_X_SPIN, RNDBITS_TBL_LOG2_SIZE,
                                        C_WHITE>
                                        <<<grid, block>>>(devId, seed, it-1, begY, ndevTot*Y, lld, bitTbl_m[tempIdx], black_m, white_m);
#endif
                        CHECK_ERROR("spinUpdate_k");
#if defined(USE_MNNVL) && defined(USE_MEMBRARIER)
			if (ntask > 1) {
				if (i ==         0) setAndWaitFlag_k<<<1,1>>>(it, flagCurrW_m[i], flagPrevW_m[i]);
				if (i == ndevLoc-1) setAndWaitFlag_k<<<1,1>>>(it, flagCurrW_m[i], flagNextW_m[i]);
				CHECK_ERROR("setAndWaitFlag_k");
			}
#endif
                }
		for(int i = 0; i < ndevLoc; i++) {
			CHECK_CUDA(cudaSetDevice(i));
			CHECK_CUDA(cudaDeviceSynchronize());
		}
#if defined(USE_MNNVL) && !defined(USE_MEMBRARIER)
		if (ntask > 1) {
			MPI_Barrier(MPI_COMM_WORLD);
		}
#endif
		mc_et = Wtime()-mc_et;
                mc_tot += mc_et;

                statIt++;

                if (printFreq && (it % printFreq) == 0) {
                        upd_t = Wtime()-upd_t;

#if defined(USE_MNNVL) && defined(USE_MEMBRARIER)
			MPI_Barrier(MPI_COMM_WORLD);
#endif
                        countSpins(ndevLoc, Y, lld, black_m, white_m, sum_d, sum_h);
                        magn = abs(double(sum_h[S_POS1])-double(sum_h[S_NEG1])) / nspinTot;

                        sdv = computeSD(ndevLoc, grid, block, Y, lld, expSD_m[tempIdx], black_m, white_m);

                        PRINTF1("%12lld %14E %14llu %14llu %12lf %12.2lf %12.2lf ",
                               it, magn, sum_h[S_NEG1], sum_h[S_POS1], sdv, (nspinTot*statIt)/(upd_t*1.0E+9),
                               (2ull*rwbytes_upd*statIt / 1.0E+9) / upd_t);

                        if (corrOut) {
                                computeCorr(corrType, corrFpath, ndevLoc, it, lld, Y, X, black_m, white_m);
                        }
			
			//PRINTF1("(mc time: %lf secs)", mc_et);

                        if (dumpOut) {
                                char fname[256];
                                snprintf(fname, sizeof(fname), "lattice_%dx%d_it%08lld_temp%.15G", Y, X, it, temps[tempIdx]);
                                dumpSpins(fname, ndevLoc, Y, lld, llenGpu, black_m, white_m);
                        }
                        if (printExp) {
                                printExpLast = printFreq;
                                printExpCand *= printExpFact;
                                printFreq = max(it+1, (long long)(0.5 + printExpCand));
                        }
                        statIt = 0;

                        printTime(12, (Wtime()-et)*numIt/(it-lastIt));
                        fflush(stdout);

                        upd_t = Wtime();
#ifdef USE_MNNVL
			MPI_Barrier(MPI_COMM_WORLD);
#endif
                }
        }
        for(int i = 0; i < ndevLoc; i++) {
                CHECK_CUDA(cudaSetDevice(i));
                CHECK_CUDA(cudaDeviceSynchronize());
        }
#ifdef USE_MNNVL
	MPI_Barrier(MPI_COMM_WORLD);
#endif
        et = Wtime()-et;

        // fix it and printFreq to print stats for the last step, if necessary
        it--;
        if (printExp) printFreq = printExpLast;
        if (!printFreq || (it % printFreq)) {

                upd_t = Wtime()-upd_t;

                countSpins(ndevLoc, Y, lld, black_m, white_m, sum_d, sum_h);
                magn = abs(double(sum_h[S_POS1])-double(sum_h[S_NEG1])) / nspinTot;

                sdv = computeSD(ndevLoc, grid, block, Y, lld, expSD_m[tempIdx], black_m, white_m);

                PRINTF1("%12lld %14E %14llu %14llu %12lf %12.2lf %12.2lf\n",
                       it, magn, sum_h[S_NEG1], sum_h[S_POS1], sdv, (nspinTot*statIt)/(upd_t*1.0E+9),
                       (2ull*rwbytes_upd*statIt / 1.0E+9) / upd_t);
#if 0
                if (corrOut) {
                        computeCorr(corrType, corrFpath, ndevLoc, it, lld, Y, X, black_m, white_m);
                }
#endif
                if (dumpOut) {
                        char fname[256];
                        snprintf(fname, sizeof(fname), "lattice_%dx%d_T_%.15G_IT_%08lld", Y, X, temps[tempIdx], it);
                        dumpSpins(fname, ndevLoc, Y, lld, llenGpu, black_m, white_m);
                }
        }

        double ene = computeEnergy(ndevLoc, grid, block, Y, lld, black_m, white_m);
        PRINTF1("\nFinal energy: %.6lf\n", ene);

        const double bw = (2ull*numIt*rwbytes_upd / 1.0E+9) / et;

        PRINTF1("\nDone in %E ms (stats overhead: %.2lf%%, spins/ns: %.2lf, BW: %.2lf GB/s)\n",
                et*1.0E+3,
                1.E+2*(et/mc_tot - 1.0),
                nspinTot*numIt / (et*1.0E+9),
                bw);

        if (writeChkpFpath) {
		PRINTF1("\nWriting checkpoint to file(s) %s... ", writeChkpFpath);
#ifdef USE_MNNVL
		MPI_Barrier(MPI_COMM_WORLD);
#endif
                double wt = Wtime();
                writeConfig(writeChkpFpath, grid, block, seed, printExp, printExpCand, it, ndevLoc, lld, Y, X, black_m, white_m);
#ifdef USE_MNNVL
		MPI_Barrier(MPI_COMM_WORLD);
#endif
                wt = Wtime()-wt;
                PRINTF1("done in %lf secs\n", wt);
        }
#ifdef USE_MNNVL
	// wait for all processes before starting to clean up
	MPI_Barrier(MPI_COMM_WORLD);
#endif
        if (corrFpath) {
                free(corrFpath);
        }
        if (readChkpFpath) {
                free(readChkpFpath);
        }
        if (writeChkpFpath) {
                free(writeChkpFpath);
        }

	free(begYLoc);

#ifndef USE_MNNVL
        CHECK_CUDA(cudaFree(black_m));
#else
	vmmFabricFree(vmmctx_b);
	vmmFabricFree(vmmctx_w);
#ifdef USE_MEMBRARIER
	vmmFabricFree(vmmctx_flagB);
	vmmFabricFree(vmmctx_flagW);

	free(flagPrevB_m);
	free(flagCurrB_m);
	free(flagNextB_m);

	free(flagPrevW_m);
	free(flagCurrW_m);
	free(flagNextW_m);
#endif
#endif
#if !defined(USE_UPDATE_LUT_K)
        CHECK_CUDA(cudaFree(exp_m));
#else
	for(int i = 0; i < ntemps; i++) {
		CHECK_CUDA(cudaFree(bitTbl_m[i]));
	}
#endif
	for(int i = 0; i < ntemps; i++) {
		CHECK_CUDA(cudaFree(expSD_m[i]));
	}

        for(int i = 0; i < ndevLoc; i++) {
                CHECK_CUDA(cudaSetDevice(i));
                CHECK_CUDA(cudaFree(sum_d[i]));
        }

        for(int i = 0; i < ndevLoc; i++) {
                CHECK_CUDA(cudaSetDevice(i));
                CHECK_CUDA(cudaDeviceReset());
        }
#ifdef USE_MNNVL
	MPI_Finalize();
#endif
        return 0;
}

