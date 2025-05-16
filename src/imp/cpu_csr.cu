#include "cpu.cuh"
#include "matrix.cuh"

void SPMV(int row, int col, int n, void* ptr_matrix, VEC_T vec, VEC_T result) {
    struct CSR *m_csr = (struct CSR *)ptr_matrix;
    for (int i = 0; i < row; i++) {
        result[i] = 0;
        for (int j = m_csr->row[i]; j < m_csr->row[i + 1]; j++) {
            result[i] += m_csr->val[j] * vec[m_csr->col[j]];
        }
    }
}

void* setup_matrix(int row, int col, int n, MATRIX_T matrix) {
    return (void*) matrix_csr_format(row, col, n, matrix);
}
