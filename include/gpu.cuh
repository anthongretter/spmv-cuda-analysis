#ifndef SPMV_GPU_H
#define SPMV_GPU_H

/**
 * GPU implementations related functions and macros
 */

#include "commons.cuh"
#include "matrix.cuh"

#define THREADS_PER_BLOCK 256

#define ALLOC(ptr, size)                  cudaMallocManaged(&ptr, size)
#define FREE(ptr)                         cudaFree(ptr)
#define SETUP()                           cudaEventCreate(&start); cudaEventCreate(&stop)
#define TIMER_START(clk)                  cudaEventRecord(clk)
#define TIMER_STOP(clk)                   cudaEventRecord(clk); cudaEventSynchronize(clk)
#define TIMER_DIFF(start, stop, elapsed)  cudaEventElapsedTime(elapsed, start, stop)
#define TIMER_T                           cudaEvent_t

void SPMV(int row, int col, int n, void* ptr_matrix, VEC_T vec, VEC_T result);

__global__ void SPMV_kernel(int row, int col, int n, void* ptr_matrix, VEC_T vec, VEC_T result);

void* SPMV_setup(int row, int col, int n, MATRIX_T matrix);

void SPMV_free(void* ptr_matrix);

size_t SPMV_overall_accesses(int row, int col, int n);

#endif /* SPMV_GPU_H */
