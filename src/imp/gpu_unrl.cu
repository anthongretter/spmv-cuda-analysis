#include "gpu.cuh"
#include <math.h>

template<unsigned int unroll_factor>
__device__ __forceinline__ PRIM_T compute_unrolled_sum(volatile struct CSR *csr, VEC_T vec, int &j) {
    PRIM_T sum = 0;
    
    if constexpr (unroll_factor >= 22) sum += csr->val[j + 21] * vec[csr->col[j + 21]];
    if constexpr (unroll_factor >= 21) sum += csr->val[j + 20] * vec[csr->col[j + 20]];
    if constexpr (unroll_factor >= 20) sum += csr->val[j + 19] * vec[csr->col[j + 19]];
    if constexpr (unroll_factor >= 19) sum += csr->val[j + 18] * vec[csr->col[j + 18]];
    if constexpr (unroll_factor >= 18) sum += csr->val[j + 17] * vec[csr->col[j + 17]];
    if constexpr (unroll_factor >= 17) sum += csr->val[j + 16] * vec[csr->col[j + 16]];
    if constexpr (unroll_factor >= 16) sum += csr->val[j + 15] * vec[csr->col[j + 15]];
    if constexpr (unroll_factor >= 15) sum += csr->val[j + 14] * vec[csr->col[j + 14]];
    if constexpr (unroll_factor >= 14) sum += csr->val[j + 13] * vec[csr->col[j + 13]];
    if constexpr (unroll_factor >= 13) sum += csr->val[j + 12] * vec[csr->col[j + 12]];
    if constexpr (unroll_factor >= 12) sum += csr->val[j + 11] * vec[csr->col[j + 11]];
    if constexpr (unroll_factor >= 11) sum += csr->val[j + 10] * vec[csr->col[j + 10]];
    if constexpr (unroll_factor >= 10) sum += csr->val[j + 9] * vec[csr->col[j + 9]];
    if constexpr (unroll_factor >= 9)  sum += csr->val[j + 8] * vec[csr->col[j + 8]];
    if constexpr (unroll_factor >= 8)  sum += csr->val[j + 7] * vec[csr->col[j + 7]];
    if constexpr (unroll_factor >= 7)  sum += csr->val[j + 6] * vec[csr->col[j + 6]];
    if constexpr (unroll_factor >= 6)  sum += csr->val[j + 5] * vec[csr->col[j + 5]];
    if constexpr (unroll_factor >= 5)  sum += csr->val[j + 4] * vec[csr->col[j + 4]];
    if constexpr (unroll_factor >= 4)  sum += csr->val[j + 3] * vec[csr->col[j + 3]];
    if constexpr (unroll_factor >= 3)  sum += csr->val[j + 2] * vec[csr->col[j + 2]];
    if constexpr (unroll_factor >= 2)  sum += csr->val[j + 1] * vec[csr->col[j + 1]];
    if constexpr (unroll_factor >= 1)  sum += csr->val[j + 0] * vec[csr->col[j + 0]];

    return sum;
}


__global__ void SPMV_kernel(int row, int col, int n, void* ptr_matrix, VEC_T vec, VEC_T result) {
    struct CSR *csr = (struct CSR *)ptr_matrix;

    int thread_row = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_row < row) {
        PRIM_T sum = 0;
        int row_start = csr->row[thread_row];
        int row_end = csr->row[thread_row + 1];
        int j = row_start;

        while (j < row_end) {
            int remaining = row_end - j;

            // Pick the largest unroll factor that fits
            switch (remaining) {
                default: // >=22
                    sum += compute_unrolled_sum<22>(csr, vec, j);
                    j += 22;
                    break;
                case 21:
                    sum += compute_unrolled_sum<21>(csr, vec, j);
                    j += 21;
                    break;
                case 20:
                    sum += compute_unrolled_sum<20>(csr, vec, j);
                    j += 20;
                    break;
                case 19:
                    sum += compute_unrolled_sum<19>(csr, vec, j);
                    j += 19;
                    break;
                case 18:
                    sum += compute_unrolled_sum<18>(csr, vec, j);
                    j += 18;
                    break;
                case 17:
                    sum += compute_unrolled_sum<17>(csr, vec, j);
                    j += 17;
                    break;
                case 16:
                    sum += compute_unrolled_sum<16>(csr, vec, j);
                    j += 16;
                    break;
                case 15:
                    sum += compute_unrolled_sum<15>(csr, vec, j);
                    j += 15;
                    break;
                case 14:
                    sum += compute_unrolled_sum<14>(csr, vec, j);
                    j += 14;
                    break;
                case 13:
                    sum += compute_unrolled_sum<13>(csr, vec, j);
                    j += 13;
                    break;
                case 12:
                    sum += compute_unrolled_sum<12>(csr, vec, j);
                    j += 12;
                    break;
                case 11:
                    sum += compute_unrolled_sum<11>(csr, vec, j);
                    j += 11;
                    break;
                case 10:
                    sum += compute_unrolled_sum<10>(csr, vec, j);
                    j += 10;
                    break;
                case 9:
                    sum += compute_unrolled_sum<9>(csr, vec, j);
                    j += 9;
                    break;
                case 8:
                    sum += compute_unrolled_sum<8>(csr, vec, j);
                    j += 8;
                    break;
                case 7:
                    sum += compute_unrolled_sum<7>(csr, vec, j);
                    j += 7;
                    break;
                case 6:
                    sum += compute_unrolled_sum<6>(csr, vec, j);
                    j += 6;
                    break;
                case 5:
                    sum += compute_unrolled_sum<5>(csr, vec, j);
                    j += 5;
                    break;
                case 4:
                    sum += compute_unrolled_sum<4>(csr, vec, j);
                    j += 4;
                    break;
                case 3:
                    sum += compute_unrolled_sum<3>(csr, vec, j);
                    j += 3;
                    break;
                case 2:
                    sum += compute_unrolled_sum<2>(csr, vec, j);
                    j += 2;
                    break;
                case 1:
                    sum += compute_unrolled_sum<1>(csr, vec, j);
                    j += 1;
                    break;
                case 0:
                    break;
            }
        }

        // Handle remaining elements
        // for (; j < row_end; j++) {
        //     sum += csr->val[j] * vec[csr->col[j]];
        // }

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

    cudaMemcpyAsync(d_row, host_csr->row, (row + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_col, host_csr->col, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_val, host_csr->val, n * sizeof(PRIM_T), cudaMemcpyHostToDevice);

    struct CSR host_csr_copy = *host_csr;
    host_csr_copy.row = d_row;
    host_csr_copy.col = d_col;
    host_csr_copy.val = d_val;

    cudaMemcpyAsync(device_csr, &host_csr_copy, sizeof(struct CSR), cudaMemcpyHostToDevice);

    cudaFree(host_csr->row);
    cudaFree(host_csr->col);
    cudaFree(host_csr->val);
    cudaFree(host_csr);

    return (void*)device_csr;
}

void SPMV_free(void* ptr_matrix) {
    struct CSR host_csr;
    struct CSR *device_csr = (struct CSR *)ptr_matrix;

    cudaMemcpyAsync(&host_csr, device_csr, sizeof(struct CSR), cudaMemcpyDeviceToHost);

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
