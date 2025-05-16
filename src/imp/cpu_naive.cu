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

void* setup_matrix(int row, int col, int n, MATRIX_T matrix) {
    return (void*) matrix;
}
