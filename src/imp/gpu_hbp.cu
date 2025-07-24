#include "gpu.cuh"
#include <math.h>

// Hash-Based Partitioning (HBP) structures and constants
#define HBP_HASH_BUCKETS 256
#define HBP_BLOCK_SIZE 32
#define MAX_PARTITION_SIZE 1024

// HBP matrix structure with hash-based partitioning
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

// Nonlinear hash functions for element grouping
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

// Inline function for loop unrolling with powers of 2 (HBP-optimized)
__device__ __forceinline__ PRIM_T compute_unrolled_sum_hbp(struct HBP_CSR *hbp_csr, VEC_T vec, int &j, int row_end, int unroll_factor) {
    PRIM_T sum = 0;
    
    if (unroll_factor == 16) {
        for (; j <= row_end - 16; j += 16) {
            sum += hbp_csr->val[j] * vec[hbp_csr->col[j]];
            sum += hbp_csr->val[j + 1] * vec[hbp_csr->col[j + 1]];
            sum += hbp_csr->val[j + 2] * vec[hbp_csr->col[j + 2]];
            sum += hbp_csr->val[j + 3] * vec[hbp_csr->col[j + 3]];
            sum += hbp_csr->val[j + 4] * vec[hbp_csr->col[j + 4]];
            sum += hbp_csr->val[j + 5] * vec[hbp_csr->col[j + 5]];
            sum += hbp_csr->val[j + 6] * vec[hbp_csr->col[j + 6]];
            sum += hbp_csr->val[j + 7] * vec[hbp_csr->col[j + 7]];
            sum += hbp_csr->val[j + 8] * vec[hbp_csr->col[j + 8]];
            sum += hbp_csr->val[j + 9] * vec[hbp_csr->col[j + 9]];
            sum += hbp_csr->val[j + 10] * vec[hbp_csr->col[j + 10]];
            sum += hbp_csr->val[j + 11] * vec[hbp_csr->col[j + 11]];
            sum += hbp_csr->val[j + 12] * vec[hbp_csr->col[j + 12]];
            sum += hbp_csr->val[j + 13] * vec[hbp_csr->col[j + 13]];
            sum += hbp_csr->val[j + 14] * vec[hbp_csr->col[j + 14]];
            sum += hbp_csr->val[j + 15] * vec[hbp_csr->col[j + 15]];
        }
    } else if (unroll_factor == 8) {
        for (; j <= row_end - 8; j += 8) {
            sum += hbp_csr->val[j] * vec[hbp_csr->col[j]];
            sum += hbp_csr->val[j + 1] * vec[hbp_csr->col[j + 1]];
            sum += hbp_csr->val[j + 2] * vec[hbp_csr->col[j + 2]];
            sum += hbp_csr->val[j + 3] * vec[hbp_csr->col[j + 3]];
            sum += hbp_csr->val[j + 4] * vec[hbp_csr->col[j + 4]];
            sum += hbp_csr->val[j + 5] * vec[hbp_csr->col[j + 5]];
            sum += hbp_csr->val[j + 6] * vec[hbp_csr->col[j + 6]];
            sum += hbp_csr->val[j + 7] * vec[hbp_csr->col[j + 7]];
        }
    } else if (unroll_factor == 4) {
        for (; j <= row_end - 4; j += 4) {
            sum += hbp_csr->val[j] * vec[hbp_csr->col[j]];
            sum += hbp_csr->val[j + 1] * vec[hbp_csr->col[j + 1]];
            sum += hbp_csr->val[j + 2] * vec[hbp_csr->col[j + 2]];
            sum += hbp_csr->val[j + 3] * vec[hbp_csr->col[j + 3]];
        }
    }
    
    return sum;
}

// Simplified HBP SpMV kernel
__global__ void SPMV_kernel(int row, int col, int n, void* ptr_matrix, VEC_T vec, VEC_T result) {
    struct HBP_CSR *hbp_csr = (struct HBP_CSR *)ptr_matrix;
    
    int thread_row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (thread_row < row) {
        PRIM_T sum = 0;
        int row_start = hbp_csr->row[thread_row];
        int row_end = hbp_csr->row[thread_row + 1];
        int j = row_start;

        // Apply hierarchical unrolling
        sum += compute_unrolled_sum_hbp(hbp_csr, vec, j, row_end, 16);
        sum += compute_unrolled_sum_hbp(hbp_csr, vec, j, row_end, 8);
        sum += compute_unrolled_sum_hbp(hbp_csr, vec, j, row_end, 4);
        
        // Handle remaining elements
        for (; j < row_end; j++) {
            sum += hbp_csr->val[j] * vec[hbp_csr->col[j]];
        }

        result[thread_row] = sum;
    }
}

void SPMV(int row, int col, int n, void* ptr_matrix, VEC_T vec, VEC_T result) {
    // Standard grid configuration
    int BLOCKS_PER_GRID = (row + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Launch kernel
    SPMV_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(row, col, n, ptr_matrix, vec, result);
}

void* SPMV_setup(int row, int col, int n, MATRIX_T matrix) {
    struct CSR *host_csr = matrix_csr_format(row, col, n, matrix);
    
    // Create HBP structure on host
    struct HBP_CSR *host_hbp = (struct HBP_CSR*)malloc(sizeof(struct HBP_CSR));
    
    // Calculate partitions based on matrix characteristics
    float avg_nnz_per_row = (float)n / row;
    int num_partitions = min(HBP_HASH_BUCKETS, max(8, (int)(row / MAX_PARTITION_SIZE)));
    host_hbp->num_partitions = num_partitions;
    
    // Allocate host arrays for HBP data
    int *partition_map = (int*)malloc(row * sizeof(int));
    int *partition_offsets = (int*)malloc((num_partitions + 1) * sizeof(int));
    int *bucket_sizes = (int*)malloc(HBP_HASH_BUCKETS * sizeof(int));
    int *row_lengths = (int*)malloc(row * sizeof(int));
    
    // Initialize arrays
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

    // Allocate device memory for HBP structure
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

    // Copy data to device
    cudaMemcpy(d_row, host_csr->row, (row + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, host_csr->col, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, host_csr->val, n * sizeof(PRIM_T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_partition_map, partition_map, row * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_partition_offsets, partition_offsets, (num_partitions + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bucket_sizes, bucket_sizes, HBP_HASH_BUCKETS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_lengths, row_lengths, row * sizeof(int), cudaMemcpyHostToDevice);

    // Setup device HBP structure
    struct HBP_CSR host_hbp_copy = *host_hbp;
    host_hbp_copy.row = d_row;
    host_hbp_copy.col = d_col;
    host_hbp_copy.val = d_val;
    host_hbp_copy.partition_map = d_partition_map;
    host_hbp_copy.partition_offsets = d_partition_offsets;
    host_hbp_copy.bucket_sizes = d_bucket_sizes;
    host_hbp_copy.row_lengths = d_row_lengths;

    cudaMemcpy(device_hbp, &host_hbp_copy, sizeof(struct HBP_CSR), cudaMemcpyHostToDevice);

    // Free host memory using proper macros and sequence
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

    // First copy the structure to get device pointers
    cudaMemcpy(&host_hbp, device_hbp, sizeof(struct HBP_CSR), cudaMemcpyDeviceToHost);
    
    // Then free all device memory
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
    // Calculate memory access for HBP format
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