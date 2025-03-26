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
#include <cuda.h>
#include <mpi.h>
#include "vmm_alloc.h"

#define MIN(x,y) (((x)<(y))?(x):(y))
#define MAX(x,y) (((x)>(y))?(x):(y))

#define DIV_UP(a,b) (((a)+((b)-1))/(b))

#define MAX_DEVICE_NAME (256)

#define CHECK_CUDA(call) {                                                   \
    cudaError_t err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    }}

#define CHECK_CU(call) {                                                        \
    CUresult res = call;                                                        \
    if(CUDA_SUCCESS != res) {                                                   \
	const char *errstr=NULL;                                                \
	cuGetErrorName(res, &errstr);                                           \
        fprintf(stderr, "Cuda driver API error in file '%s' in line %d: %s.\n", \
                __FILE__, __LINE__, errstr);                                    \
        exit(EXIT_FAILURE);                                                     \
    }}

static void *Malloc(size_t sz) {

	void *ptr;

	ptr = (void *)malloc(sz);
	if (!ptr) {
		fprintf(stderr, "Cannot allocate %zu bytes...\n", sz);
		exit(EXIT_FAILURE);
	}
	return ptr;
}

size_t vmmFabricGranularity(int device) {

	CUmemAllocationProp prop = {};

	prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	prop.location.id = device;

	// necessary to export the handle for remote memory access via NVLink
	prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

	size_t granularity = 0;
	CHECK_CU(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

	return granularity;
}

// call to "allocate" physical memory (cuMemCreate() handle) on GPU "device"
// On entry size contains de desired size of the allocation; on exit the actual
// size, which must be a multiple of the granularity
static CUmemGenericAllocationHandle allocatePhysicalMemory(int device, size_t size) {

	CUmemAllocationProp prop = {};

	prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	prop.location.id = device;

	// necessary to export the handle for remote memory access via NVLink
	prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

	size_t granularity = 0;
	CHECK_CU(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

	if (size % granularity) {
	
		cudaDeviceProp props;
		CHECK_CUDA(cudaGetDeviceProperties(&props, device));

		int nameLen;
		char procName[MPI_MAX_PROCESSOR_NAME];
		MPI_Get_processor_name(procName, &nameLen);	
	
		fprintf(stderr,
			"%s:%d: error, requested allocation size (%zu bytes) is "
			"not a multiple of minimum supported granularity (%zu bytes) "
			"for device %d (%s) on node %s!\n",
			__func__, __LINE__, size, granularity, device, props.name, procName);
		MPI_Abort(MPI_COMM_WORLD, 0);
	}

	// Ensure size matches granularity requirements for the allocation
	//size_t padded_size = DIV_UP(size, granularity)*granularity;
#if 0
	printf("%s:%d: device %d, padded_size: %zu\n", __func__, __LINE__, device, padded_size);
#endif
	// Allocate physical memory
	CUmemGenericAllocationHandle allocHandle;

	//printf("device: %d, size: %zu\n", device, size);
	CHECK_CU(cuMemCreate(&allocHandle, size, &prop, 0));

	return allocHandle;
}

static void setAccessOnDevice(int device, CUdeviceptr ptr, size_t size) {

	CUmemAccessDesc accessDesc = {};

	accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	accessDesc.location.id = device;
	accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

	//printf("device: %d\n", device);

	// Make the address accessible
	CHECK_CU(cuMemSetAccess(ptr, size, &accessDesc, 1));

	return;
}

vmmAllocCtx_t *vmmFabricMalloc(void **devPtr, size_t sizePerGpu) {

	int inited = 0;
	MPI_Initialized(&inited);

	if (!inited) {
		fprintf(stderr,
			"%s:%d: error, MPI must be initialized  before calling this function!\n",
			__func__, __LINE__);
		exit(EXIT_FAILURE);
	}

	int rank, ntask;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ntask);

	char (*procNames)[MPI_MAX_PROCESSOR_NAME] = (char (*)[MPI_MAX_PROCESSOR_NAME])Malloc(sizeof(*procNames)*ntask);
	int nameLen;
	MPI_Get_processor_name(procNames[rank], &nameLen);
	MPI_Gather(procNames[rank], MPI_MAX_PROCESSOR_NAME, MPI_CHAR, procNames, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);

	int ndev = 0;
	CHECK_CUDA(cudaGetDeviceCount(&ndev));

	int ndev_or;
	int ndev_and;
	MPI_Allreduce(&ndev, &ndev_or,  1, MPI_INT, MPI_BOR,  MPI_COMM_WORLD);
	MPI_Allreduce(&ndev, &ndev_and, 1, MPI_INT, MPI_BAND, MPI_COMM_WORLD);
	if (ndev_or != ndev_and) {
		if (!rank) {
			fprintf(stderr,
				"%s:%d: error, not all processes have the same number of GPUs!\n",
				__func__, __LINE__);
		}
		MPI_Abort(MPI_COMM_WORLD, 0);
	}
	
	// local GPUs
	cudaDeviceProp *props = (cudaDeviceProp *)Malloc(sizeof(*props)*ndev);
	for(int i = 0; i < ndev; i++) {
		CHECK_CUDA(cudaGetDeviceProperties(props+i, i));
	}

	// check local GPUs support
	for(int i = 0; i < ndev; i++) {

		int deviceSupportsVmm;
		CHECK_CU(cuDeviceGetAttribute(&deviceSupportsVmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, i));
		if (!deviceSupportsVmm) {
			fprintf(stderr,
				"%s:%d: error, device %d (%s) on node %s does NOT support Virtual Memory Management!\n",
				__func__, __LINE__, i, props[i].name, procNames[rank]);
			MPI_Abort(MPI_COMM_WORLD, 0);
		}

		// FOR FABRIC
		int deviceSupportsFabricMem;
		CHECK_CU(cuDeviceGetAttribute(&deviceSupportsFabricMem, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, i));
		if (deviceSupportsFabricMem == 0) {
			fprintf(stderr,
				"%s:%d: error, device %d (%s) on node %s does NOT support Fabric Handles!\n",
				__func__, __LINE__, i, props[i].name, procNames[rank]);
			MPI_Abort(MPI_COMM_WORLD, 0);
		}
	}

	// check that all GPUs are of the same kind (this may be relaxed)
	cudaDeviceProp *props_all = NULL;
	if (!rank) {
		props_all = (cudaDeviceProp *)Malloc(sizeof(*props)*ntask*ndev);
	}

	MPI_Datatype MPI_DEV_PROP;
	MPI_Type_contiguous(sizeof(cudaDeviceProp), MPI_BYTE, &MPI_DEV_PROP);
	MPI_Type_commit(&MPI_DEV_PROP);

	MPI_Gather(props, ndev, MPI_DEV_PROP, props_all, ndev, MPI_DEV_PROP, 0, MPI_COMM_WORLD);

	if (!rank) {
		for(int i = 1; i < ntask*ndev; i++) {
			if (strncmp(props_all[i-1].name, props_all[i].name, MAX_DEVICE_NAME)) {
				fprintf(stderr,
					"%s:%d: error, device %d from proc %d (%s) and "
					"device %d from proc %d (%s) are different:\n"
					"\t%s\n\t%s\n",
					__func__, __LINE__,
					(i-1)%ndev, (i-1)/ndev, procNames[(i-1)/ndev],
					 i   %ndev,  i   /ndev, procNames[ i   /ndev],
					props_all[i-1].name, props_all[i].name);
				MPI_Abort(MPI_COMM_WORLD, 0);
			}
		}
	}
	free(props);
	free(props_all);

	// allocate local handles
	CUmemGenericAllocationHandle *handles = (CUmemGenericAllocationHandle *)Malloc(sizeof(*handles)*ntask*ndev);
	memset(handles, 0, sizeof(*handles)*ntask*ndev);

	for(int i = 0; i < ndev; i++) {
		handles[rank*ndev + i] = allocatePhysicalMemory(i, sizePerGpu);
	}

	// export local handles
	CUmemFabricHandle *fabricHandles = (CUmemFabricHandle *)Malloc(sizeof(*fabricHandles)*ntask*ndev);
	memset(fabricHandles, 0, sizeof(*fabricHandles)*ntask*ndev);
	for(int i = 0; i < ndev; i++) {
		//printf("CU_MEM_HANDLE_TYPE_FABRIC: %d, CU_MEM_HANDLE_TYPE_MAX: %d\n", CU_MEM_HANDLE_TYPE_FABRIC, CU_MEM_HANDLE_TYPE_MAX);
		CHECK_CU(cuMemExportToShareableHandle(&fabricHandles[ndev*rank + i],
						      handles[ndev*rank + i],
						      CU_MEM_HANDLE_TYPE_FABRIC, 0));
	}

	// distribute local handles
	MPI_Datatype MPI_FABRIC_HANDLE;
	MPI_Type_contiguous(sizeof(CUmemFabricHandle), MPI_BYTE, &MPI_FABRIC_HANDLE);
	MPI_Type_commit(&MPI_FABRIC_HANDLE);

	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, fabricHandles, ndev, MPI_FABRIC_HANDLE, MPI_COMM_WORLD);

	// import remote handles
	for(int i = 0; i < ntask; i++) {
		if (i == rank) {
			continue;
		}
		for(int d = 0; d < ndev; d++) {
			CHECK_CU(cuMemImportFromShareableHandle(&handles[i*ndev + d],
							        &fabricHandles[i*ndev + d],
								CU_MEM_HANDLE_TYPE_FABRIC));
		}
	}
	// this can now be removed?
	free(fabricHandles);

	// create a (large) Virtual Address range and map local and remote handles
	const size_t totalSize = sizePerGpu*size_t(ntask)*size_t(ndev);

	CUdeviceptr cuptr;
	CHECK_CU(cuMemAddressReserve(&cuptr, totalSize, 0, 0, 0));

	for(size_t i = 0; i < ntask; i++) {
		for(size_t d = 0; d < ndev; d++) {
			CHECK_CU(cuMemMap(cuptr + i*sizePerGpu*ndev + d*sizePerGpu,
					  sizePerGpu, 0, handles[i*ndev + d], 0));
		}
	}

	for(int d = 0; d < ndev; d++) {
		setAccessOnDevice(d, cuptr, totalSize); //sizePerGpu*ntask*ndev);
	}


	free(procNames);

	vmmAllocCtx_t *ctx = (vmmAllocCtx_t *)Malloc(sizeof(*ctx));
	
	ctx->cuptr = cuptr;
	ctx->virtAddrRangeSize = totalSize;

	ctx->handles = handles;

	*devPtr = (void *)cuptr;

	return ctx;
}

void vmmFabricFree(vmmAllocCtx_t *ctx) {

	int ndev = 0;
	CHECK_CUDA(cudaGetDeviceCount(&ndev));
	
	int rank, ntask;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ntask);

	CHECK_CU(cuMemUnmap(ctx->cuptr, ctx->virtAddrRangeSize));

	for(int i = 0; i < ntask; i++) {
		for(int d = 0; d < ndev; d++) {
			CHECK_CU(cuMemRelease(ctx->handles[i*ndev + d]));
		}
	}
	CHECK_CU(cuMemAddressFree(ctx->cuptr, ctx->virtAddrRangeSize));

	free(ctx->handles);
	free(ctx);

	return;
}
