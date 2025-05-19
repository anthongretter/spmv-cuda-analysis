#include "cpu.cuh"

void SPMV(int row, int col, int n, void* ptr_matrix, VEC_T vec, VEC_T result) {
    MATRIX_T matrix = (MATRIX_T) ptr_matrix;
    for (int i = 0; i < row; i++) {
        result[i] = 0;
        for (int j = 0; j < col; j++) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
}

void* SPMV_setup(int row, int col, int n, MATRIX_T matrix) {
    return (void*) matrix;
}

void SPMV_free(void* ptr_matrix) {
    MATRIX_T matrix = (MATRIX_T) ptr_matrix;
    matrix_free(matrix);
}

size_t SPMV_overall_accesses(int row, int col, int n) {
    return (row * col) * sizeof(PRIM_T) +  // matrix
           col * sizeof(PRIM_T) +          // result[i] = 0
           col * sizeof(PRIM_T);           // vec
}
