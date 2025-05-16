#include "gpu.cuh"

__global__ void SPMV_kernel(int row, int col, int n, void* ptr_matrix, VEC_T vec, VEC_T result) {
    struct CSR *csr = (struct CSR *)ptr_matrix;

    // Thread ID corresponds to a row index
    int thread_row = blockIdx.x * blockDim.x + threadIdx.x;

    // Check row index is within bounds
    if (thread_row < row) {
        PRIM_T sum = 0;

        // Process all non-zero elements in this row
        for (int j = csr->row[thread_row]; j < csr->row[thread_row + 1]; j++) {
            sum += csr->val[j] * vec[csr->col[j]];
        }

        // Store the result for this row
        result[thread_row] = sum;
    }
}

void SPMV(int row, int col, int n, void* ptr_matrix, VEC_T vec, VEC_T result) {
    SPMV_kernel<<<1, THREADS_PER_BLOCK>>>(row, col, n, ptr_matrix, vec, result);
}

void* setup_matrix(int row, int col, int n, MATRIX_T matrix) {
    return (void*) matrix_csr_format(row, col, n, matrix);
}

