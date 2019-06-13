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
#ifndef __CUDA_MACRO_H__
#define __CUDA_MACRO_H__

#define CHECK_CUDA(call) {                                                   \
    cudaError_t err = call;                                                  \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    }}

#define CHECK_CUBLAS(call) {                                                 \
    cublasStatus_t status = call;                                            \
    if( CUBLAS_STATUS_SUCCESS != status) {                                   \
        fprintf(stderr, "CUBLAS error: %s = %d at (%s:%d)\n", #call,         \
                status, __FILE__, __LINE__);                                 \
        exit(EXIT_FAILURE);                                                  \
    }}

#define CHECK_CURAND(call) {                                                 \
    curandStatus_t status = call;                                            \
    if( CURAND_STATUS_SUCCESS != status) {                                   \
        fprintf(stderr, "CURAND error: %s = %d at (%s:%d)\n", #call,         \
                status, __FILE__, __LINE__);                                 \
        exit(EXIT_FAILURE);                                                  \
    }}

#define CHECK_ERROR(errorMessage) {                                          \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }}
#endif
