#ifndef SPMV_GPU_H
#define SPMV_GPU_H

/**
 * GPU implementations related functions and macros
 */

#include "commons.h"
#include "matrix.h"

#ifdef GPU_IMP

#define ALLOC(ptr, size)                cudaMalloc(ptr, size)
#define FREE(ptr)                       cudaFree(ptr)
#define ROUTINE(csr, vec, result, ops)

/**
 * Performs sparse matrix-vector multiplication using CSR format
 * @param m_csr Pointer to CSR matrix structure
 * @param vec Input vector
 * @param result Output vector storing the multiplication result
 */
__global__ static void spmv(struct CSR *m_csr, const VEC_T vec, VEC_T result, long *ops) {
    // Thread ID corresponds to a row index
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Check row index is within bounds
    if (row < row_length) {
        PRIM_T sum = 0;

        // Process all non-zero elements in this row
        for (int j = csr_row[row]; j < csr_row[row + 1]; j++) {
            sum += csr_val[j] * vec[csr_col[j]];
        }

        // Store the result for this row
        result[row] = sum;
    }

}
#endif

#endif /* SPMV_GPU_H */
