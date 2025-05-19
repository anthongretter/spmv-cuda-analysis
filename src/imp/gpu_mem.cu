#include "gpu.cuh"
#include <math.h>

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
    struct CSR *host_csr = matrix_csr_format(row, col, n, matrix);

    struct CSR *device_csr;
    cudaMalloc(&device_csr, sizeof(struct CSR));

    int *d_row, *d_col;
    PRIM_T *d_val;
    cudaMalloc(&d_row, (row + 1) * sizeof(int));
    cudaMalloc(&d_col, n * sizeof(int));
    cudaMalloc(&d_val, n * sizeof(PRIM_T));

    cudaMemcpy(d_row, host_csr->row, (row + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, host_csr->col, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, host_csr->val, n * sizeof(PRIM_T), cudaMemcpyHostToDevice);

    struct CSR host_csr_copy = *host_csr;
    host_csr_copy.row = d_row;
    host_csr_copy.col = d_col;
    host_csr_copy.val = d_val;

    cudaMemcpy(device_csr, &host_csr_copy, sizeof(struct CSR), cudaMemcpyHostToDevice);

    cudaFree(host_csr->row);
    cudaFree(host_csr->col);
    cudaFree(host_csr->val);
    cudaFree(host_csr);

    return (void*)device_csr;
}

void SPMV_free(void* ptr_matrix) {
    struct CSR host_csr;
    struct CSR *device_csr = (struct CSR *)ptr_matrix;

    cudaMemcpy(&host_csr, device_csr, sizeof(struct CSR), cudaMemcpyDeviceToHost);

    cudaFree(host_csr.row);
    cudaFree(host_csr.col);
    cudaFree(host_csr.val);
    cudaFree(device_csr);
}


size_t SPMV_overall_accesses(int row, int col, int n) {
    return (row + 1) * sizeof(int) +  // csr->row
           n * sizeof(int) +          // csr->col
           col * sizeof(PRIM_T) +     // vec
           col * sizeof(PRIM_T) +     // result
           n * sizeof(PRIM_T);        // csr->val
}
