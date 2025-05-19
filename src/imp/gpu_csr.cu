#include "gpu.cuh"

__global__ void SPMV_kernel(int row, int col, int n, void* ptr_matrix, VEC_T vec, VEC_T result) {
    struct CSR *csr = (struct CSR *)ptr_matrix;

    int thread_row = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_row < row) {
        PRIM_T sum = 0;
        for (int j = csr->row[thread_row]; j < csr->row[thread_row + 1]; j++) {
            sum += csr->val[j] * vec[csr->col[j]];
        }
        result[thread_row] = sum;
    }
}

void SPMV(int row, int col, int n, void* ptr_matrix, VEC_T vec, VEC_T result) {
    int BLOCKS_PER_GRID = (row + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    SPMV_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(row, col, n, ptr_matrix, vec, result);
}

void* SPMV_setup(int row, int col, int n, MATRIX_T matrix) {
    struct CSR *csr = matrix_csr_format(row, col, n, matrix);
    return (void*)csr;
}

void SPMV_free(void* ptr_matrix) {
    struct CSR *device_csr = (struct CSR*)ptr_matrix;
    struct CSR host_csr_copy;

    cudaMemcpy(&host_csr_copy, device_csr, sizeof(struct CSR), cudaMemcpyDeviceToHost);

    cudaFree(host_csr_copy.row);
    cudaFree(host_csr_copy.col);
    cudaFree(host_csr_copy.val);
    cudaFree(device_csr);
}

size_t SPMV_overall_accesses(int row, int col, int n) {
    return (row + 1) * sizeof(int) +  // csr->row
           n * sizeof(int) +          // csr->col
           col * sizeof(PRIM_T) +     // vec
           col * sizeof(PRIM_T) +     // result
           n * sizeof(PRIM_T);        // csr->val
}