#ifndef SPMV_CPU_H
#define SPMV_CPU_H

/**
 * CPU implementations related functions and macros
 */

#include <time.h>

#include "commons.cuh"
#include "matrix.cuh"

#define ALLOC(ptr, size)                  ptr = static_cast<decltype(ptr)>(malloc(size))
#define FREE(ptr)                         free(ptr)
#define SETUP()
#define TIMER_START(clk)                  clk = clock()
#define TIMER_STOP(clk)                   clk = clock()
#define TIMER_DIFF(start, stop, elapsed)  *elapsed = ((double) (stop - start) / CLOCKS_PER_SEC)
#define TIMER_T                           clock_t

void SPMV(int row, int col, int n, void* ptr_matrix, VEC_T vec, VEC_T result);

void* SPMV_setup(int row, int col, int n, MATRIX_T matrix);

void SPMV_free(void* ptr_matrix);

size_t SPMV_overall_accesses(int row, int col, int n);

#endif /* SPMV_CPU_H */
