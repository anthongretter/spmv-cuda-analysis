#ifndef SPMV_CPU_H
#define SPMV_CPU_H

/**
 * CPU implementations related functions and macros
 */

#include "commons.cuh"
#include "matrix.cuh"
#include "config.cuh"

void SPMV(int row, int col, int n, void* ptr_matrix, VEC_T vec, VEC_T result);

void* SPMV_setup(int row, int col, int n, MATRIX_T matrix);

void SPMV_free(void* ptr_matrix);

size_t SPMV_overall_accesses(int row, int col, int n);

#endif /* SPMV_CPU_H */
