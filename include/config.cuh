#ifndef SPMV_CONFIG_H
#define SPMV_CONFIG_H

/**
 * Configuration header for SPMV program
 * 
 * Includes the appropriate implementation header (GPU or CPU)
 * based on compilation flag (-DGPU)
 */

#include <time.h>

#if defined GPU

#include "gpu.cuh"

#define THREADS_PER_BLOCK 256

#define ALLOC(ptr, size)                  cudaMallocManaged(&ptr, size)
#define FREE(ptr)                         cudaFree(ptr)
#define SETUP()                           cudaEventCreate(&start); cudaEventCreate(&stop)
#define TIMER_START(clk)                  cudaEventRecord(clk)
#define TIMER_STOP(clk)                   cudaEventRecord(clk); cudaEventSynchronize(clk)
#define TIMER_DIFF(start, stop, elapsed)  cudaEventElapsedTime(elapsed, start, stop); *elapsed /= 1000.0;
#define TIMER_T                           cudaEvent_t

#else

#include "cpu.cuh"

#define ALLOC(ptr, size)                  ptr = static_cast<decltype(ptr)>(malloc(size))
#define FREE(ptr)                         free(ptr)
#define SETUP()
#define TIMER_START(clk)                  clk = clock()
#define TIMER_STOP(clk)                   clk = clock()
#define TIMER_DIFF(start, stop, elapsed)  *elapsed = ((double) (stop - start) / CLOCKS_PER_SEC)
#define TIMER_T                           clock_t

#endif

#endif /* SPMV_CONFIG_H */
