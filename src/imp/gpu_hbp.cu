#include "gpu.cuh"
#include <math.h>

#define HBP_HASH_BUCKETS 256
#define HBP_BLOCK_SIZE 32
#define MAX_PARTITION_SIZE 1024

struct HBP_CSR {
    int* row;
    int* col;
    PRIM_T* val;
    int* partition_map;    // Maps rows to partitions
    int* partition_offsets; // Starting row for each partition
    int* bucket_sizes;     // Size of each hash bucket
    int num_partitions;    // Total number of partitions
    int* row_lengths;      // Length of each row (for load balancing)
};

__device__ __host__ __forceinline__ int hash_function_1(int row, int col, int num_buckets) {
    // Polynomial hash with good distribution properties
    return ((row * 73 + col * 37) ^ (row >> 2)) % num_buckets;
}

__device__ __host__ __forceinline__ int hash_function_2(int row, int nnz_count, int num_buckets) {
    // Hash based on row and its sparsity pattern
    return ((row * 101 + nnz_count * 53) ^ (nnz_count << 3)) % num_buckets;
}

__device__ __host__ __forceinline__ int competitive_hash(int row, int workload, int num_partitions) {
    // Competitive method for load balancing
    int base_partition = row % num_partitions;
    int workload_factor = (workload > 16) ? (workload / 8) : 1;
    return (base_partition + workload_factor) % num_partitions;
}

template<unsigned int unroll_factor>
__device__ PRIM_T compute_unrolled_sum(volatile struct HBP_CSR *csr, volatile VEC_T vec, int &j) {
    PRIM_T sum = 0;

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
    struct HBP_CSR *hbp_csr = (struct HBP_CSR *)ptr_matrix;
    
    int thread_row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (thread_row < row) {
        PRIM_T sum = 0;
        int row_start = hbp_csr->row[thread_row];
        int row_end = hbp_csr->row[thread_row + 1];
        int j = row_start;

        while (j < row_end) {
            int remaining = row_end - j;

            // Pick the largest unroll factor that fits
            switch (remaining) {
                default: // >=16
                    sum += compute_unrolled_sum<16>(hbp_csr, vec, j);
                    j += 16;
                    break;
                case 15:
                    sum += compute_unrolled_sum<15>(hbp_csr, vec, j);
                    j += 15;
                    break;
                case 14:
                    sum += compute_unrolled_sum<14>(hbp_csr, vec, j);
                    j += 14;
                    break;
                case 13:
                    sum += compute_unrolled_sum<13>(hbp_csr, vec, j);
                    j += 13;
                    break;
                case 12:
                    sum += compute_unrolled_sum<12>(hbp_csr, vec, j);
                    j += 12;
                    break;
                case 11:
                    sum += compute_unrolled_sum<11>(hbp_csr, vec, j);
                    j += 11;
                    break;
                case 10:
                    sum += compute_unrolled_sum<10>(hbp_csr, vec, j);
                    j += 10;
                    break;
                case 9:
                    sum += compute_unrolled_sum<9>(hbp_csr, vec, j);
                    j += 9;
                    break;
                case 8:
                    sum += compute_unrolled_sum<8>(hbp_csr, vec, j);
                    j += 8;
                    break;
                case 7:
                    sum += compute_unrolled_sum<7>(hbp_csr, vec, j);
                    j += 7;
                    break;
                case 6:
                    sum += compute_unrolled_sum<6>(hbp_csr, vec, j);
                    j += 6;
                    break;
                case 5:
                    sum += compute_unrolled_sum<5>(hbp_csr, vec, j);
                    j += 5;
                    break;
                case 4:
                    sum += compute_unrolled_sum<4>(hbp_csr, vec, j);
                    j += 4;
                    break;
                case 3:
                    sum += compute_unrolled_sum<3>(hbp_csr, vec, j);
                    j += 3;
                    break;
                case 2:
                    sum += compute_unrolled_sum<2>(hbp_csr, vec, j);
                    j += 2;
                    break;
                case 1:
                    sum += compute_unrolled_sum<1>(hbp_csr, vec, j);
                    j += 1;
                    break;
                case 0:
                    break;
            }
        }

        // Remaining elements
        for (; j < row_end; j++) {
            sum += hbp_csr->val[j] * vec[hbp_csr->col[j]];
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
    
    struct HBP_CSR *host_hbp = (struct HBP_CSR*)malloc(sizeof(struct HBP_CSR));
    
    float avg_nnz_per_row = (float)n / row;
    int num_partitions = min(HBP_HASH_BUCKETS, max(8, (int)(row / MAX_PARTITION_SIZE)));
    host_hbp->num_partitions = num_partitions;
    
    int *partition_map = (int*)malloc(row * sizeof(int));
    int *partition_offsets = (int*)malloc((num_partitions + 1) * sizeof(int));
    int *bucket_sizes = (int*)malloc(HBP_HASH_BUCKETS * sizeof(int));
    int *row_lengths = (int*)malloc(row * sizeof(int));
    
    memset(bucket_sizes, 0, HBP_HASH_BUCKETS * sizeof(int));
    memset(partition_offsets, 0, (num_partitions + 1) * sizeof(int));
    
    // Calculate row lengths and hash-based partitioning
    for (int i = 0; i < row; i++) {
        row_lengths[i] = host_csr->row[i + 1] - host_csr->row[i];
        int bucket = hash_function_1(i, row_lengths[i], HBP_HASH_BUCKETS);
        int partition = competitive_hash(i, row_lengths[i], num_partitions);
        
        partition_map[i] = partition;
        bucket_sizes[bucket]++;
        partition_offsets[partition + 1]++;
    }
    
    // Convert partition counts to offsets
    for (int i = 1; i <= num_partitions; i++) {
        partition_offsets[i] += partition_offsets[i - 1];
    }

    struct HBP_CSR *device_hbp;
    cudaMalloc(&device_hbp, sizeof(struct HBP_CSR));

    int *d_row, *d_col, *d_partition_map, *d_partition_offsets, *d_bucket_sizes, *d_row_lengths;
    PRIM_T *d_val;
    
    cudaMalloc(&d_row, (row + 1) * sizeof(int));
    cudaMalloc(&d_col, n * sizeof(int));
    cudaMalloc(&d_val, n * sizeof(PRIM_T));
    cudaMalloc(&d_partition_map, row * sizeof(int));
    cudaMalloc(&d_partition_offsets, (num_partitions + 1) * sizeof(int));
    cudaMalloc(&d_bucket_sizes, HBP_HASH_BUCKETS * sizeof(int));
    cudaMalloc(&d_row_lengths, row * sizeof(int));

    cudaMemcpyAsync(d_row, host_csr->row, (row + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_col, host_csr->col, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_val, host_csr->val, n * sizeof(PRIM_T), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_partition_map, partition_map, row * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_partition_offsets, partition_offsets, (num_partitions + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_bucket_sizes, bucket_sizes, HBP_HASH_BUCKETS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_row_lengths, row_lengths, row * sizeof(int), cudaMemcpyHostToDevice);

    struct HBP_CSR host_hbp_copy = *host_hbp;
    host_hbp_copy.row = d_row;
    host_hbp_copy.col = d_col;
    host_hbp_copy.val = d_val;
    host_hbp_copy.partition_map = d_partition_map;
    host_hbp_copy.partition_offsets = d_partition_offsets;
    host_hbp_copy.bucket_sizes = d_bucket_sizes;
    host_hbp_copy.row_lengths = d_row_lengths;

    cudaMemcpyAsync(device_hbp, &host_hbp_copy, sizeof(struct HBP_CSR), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize(); // Ensure all transfers complete before freeing
    FREE(host_csr->row);
    FREE(host_csr->col);
    FREE(host_csr->val);
    FREE(host_csr);
    free(host_hbp);
    free(partition_map);
    free(partition_offsets);
    free(bucket_sizes);
    free(row_lengths);

    return (void*)device_hbp;
}

void SPMV_free(void* ptr_matrix) {
    struct HBP_CSR host_hbp;
    struct HBP_CSR *device_hbp = (struct HBP_CSR *)ptr_matrix;

    cudaMemcpy(&host_hbp, device_hbp, sizeof(struct HBP_CSR), cudaMemcpyDeviceToHost);

    cudaFree(host_hbp.row);
    cudaFree(host_hbp.col);
    cudaFree(host_hbp.val);
    cudaFree(host_hbp.partition_map);
    cudaFree(host_hbp.partition_offsets);
    cudaFree(host_hbp.bucket_sizes);
    cudaFree(host_hbp.row_lengths);
    cudaFree(device_hbp);
}


size_t SPMV_overall_accesses(int row, int col, int n) {
    int num_partitions = min(HBP_HASH_BUCKETS, max(8, row / MAX_PARTITION_SIZE));
    
    return (row + 1) * sizeof(int) +           // hbp_csr->row
           n * sizeof(int) +                   // hbp_csr->col
           col * sizeof(PRIM_T) +              // vec
           row * sizeof(PRIM_T) +              // result
           n * sizeof(PRIM_T) +                // hbp_csr->val
           row * sizeof(int) +                 // partition_map
           (num_partitions + 1) * sizeof(int) + // partition_offsets
           HBP_HASH_BUCKETS * sizeof(int) +    // bucket_sizes
           row * sizeof(int);                  // row_lengths
}