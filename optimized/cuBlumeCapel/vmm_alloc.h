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
#ifndef __VMM_ALLOC_H__
#define __VMM_ALLOC_H__

typedef struct {

	CUdeviceptr cuptr;
	size_t virtAddrRangeSize;

	CUmemGenericAllocationHandle *handles;

} vmmAllocCtx_t;

#ifdef __cplusplus
extern "C" {
#endif

// helper to obtain the minimum size for fabric allocations
size_t vmmFabricGranularity(int device);

// Allocates sizePerGPU bytes on each device 
// visible to each MPI rank and return to
// each caller the starting address of a 
// Virtual Address range to which all the 
// allocations are mapped. Mappings are
// performed in Rank,DeviceId order:
//
// <Rank   0, Device 0><Rank   0, Device 1>, <Rank   0, Device N-1>, 
// <Rank   1, Device 0><Rank   1, Device 1>, <Rank   1, Device N-1>,
// ...
// <Rank M-1, Device 0><Rank M-1, Device 1>, <Rank M-1, Device N-1>,
//
// Remote memories are accessed via FABRIC handles.
//
// Requirements:
//   * all ranks must have access to the same number of GPUs;
//   * all the GPUs must be the same type;
vmmAllocCtx_t *vmmFabricMalloc(void **devPtr, size_t sizePerGpu);

void vmmFabricFree(vmmAllocCtx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif
