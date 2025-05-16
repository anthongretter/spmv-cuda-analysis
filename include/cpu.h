#ifndef SPMV_CPU_H
#define SPMV_CPU_H

/**
 * CPU implementations related functions and macros
 */

#include <time.h>

#include "commons.h"
#include "matrix.h"

#ifndef GPU_IMP

#define ALLOC(ptr, size)               ptr = malloc(size)
#define FREE(ptr)                      free(ptr)
#define TIMER_START()                  clock()
#define TIMER_STOP()                   clock()
#define SPMV(csr, vec, result, ops)    spmv(csr, vec, result, ops)

/**
 * Performs sparse matrix-vector multiplication using CSR format
 * @param m_csr Pointer to CSR matrix structure
 * @param vec Input vector
 * @param result Output vector storing the multiplication result
 */
static void spmv_csr(struct CSR *m_csr, const VEC_T vec, VEC_T result, long *ops) {
    for (int i = 0; i < m_csr->row_length; i++) {
        result[i] = 0;
        for (int j = m_csr->row[i]; j < m_csr->row[i + 1]; j++) {
            (*ops) += 2;
            result[i] += m_csr->val[j] * vec[m_csr->col[j]];
        }
    }
}

/**
 * Performs sparse matrix-vector multiplication using naive algorithm
 * @param row Number of rows in the matrix
 * @param col Number of columns in the matrix
 * @param matrix Pointer to the matrix
 * @param vec Input vector
 * @param result Output vector storing the multiplication result
 * @param ops Pointer to a long variable storing the number of operations performed
 */
static void spmv_naive(int row, int col, MATRIX_T matrix, VEC_T vec, VEC_T result, long *ops) {
    for (int i = 0; i < row; i++) {
        result[i] = 0;
        for (int j = 0; j < col; j++) {
            (*ops) += 2;
            result[i] += matrix[i][j] * vec[j];
        }
    }
}

#endif

#endif /* SPMV_CPU_H */
