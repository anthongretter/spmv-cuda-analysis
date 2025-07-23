#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mtx.cuh"
#include "config.cuh"
#include "matrix.cuh"
#include "commons.cuh"

FILE* mtx_fopen_from_cli(int argc, const char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <file>\n\n\t<file>: A .mtx matrix file format\n", argv[0]);
        exit(1);
    }

    char filename[strlen(argv[1]) + 1];
    strcpy(filename, argv[1]);

    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Error: File '%s' not found\n", filename);
        exit(1);
    }
    printf("Using file: %s\n", filename);
    return f;
}

void mtx_fclose(FILE* file) {
    fclose(file);
}

MATRIX_T mtx_parse(FILE* file, int* row, int* col, int* n, matrix_geometry_t* geometry) {
    printf("Reading matrix...\n");

    char line[256];
    while (fgets(line, sizeof(line), file) && line[0] == '%'); // Skip headers
    sscanf(line, "%d %d %d", row, col, n);

    MATRIX_T matrix = matrix_alloc(*row, *col);
    int r, c;
    PRIM_T value;
	
    for (int i = 0; i < *n; i++) {
        fscanf(file, "%d %d %f", &r, &c, &value);
        matrix[r - 1][c - 1] = value;
    }

    printf("Loaded matrix (%dx%d) of %d non-zero elements\n", *row, *col, *n);
    
    // Analyze geometry if requested
    if (geometry != NULL) {
        mtx_analyze_geometry(matrix, *row, *col, *n, geometry);
    }
    
    return matrix;
}

void mtx_analyze_geometry(MATRIX_T matrix, int row, int col, int n, matrix_geometry_t* geometry) {
    printf("Analyzing matrix geometry...\n");
    
    // Initialize geometry structure
    geometry->rows = row;
    geometry->cols = col;
    geometry->nnz = n;
    geometry->density = (double)n / (double)(row * col);
    
    // Analyze row distribution
    int* row_nnz = (int*)calloc(row, sizeof(int));
    geometry->min_nnz_per_row = col + 1; // Initialize to impossible value
    geometry->max_nnz_per_row = 0;
    geometry->empty_rows = 0;
    geometry->full_rows = 0;
    
    // Count non-zeros per row and analyze sparsity pattern
    int min_col = col, max_col = 0;
    double sum_nnz = 0.0;
    
    for (int i = 0; i < row; i++) {
        int row_min_col = col, row_max_col = -1;
        for (int j = 0; j < col; j++) {
            if (matrix[i][j] != 0.0) {
                row_nnz[i]++;
                if (j < row_min_col) row_min_col = j;
                if (j > row_max_col) row_max_col = j;
                if (j < min_col) min_col = j;
                if (j > max_col) max_col = j;
            }
        }
        
        sum_nnz += row_nnz[i];
        
        if (row_nnz[i] == 0) geometry->empty_rows++;
        if (row_nnz[i] == col) geometry->full_rows++;
        if (row_nnz[i] < geometry->min_nnz_per_row) geometry->min_nnz_per_row = row_nnz[i];
        if (row_nnz[i] > geometry->max_nnz_per_row) geometry->max_nnz_per_row = row_nnz[i];
    }
    
    geometry->avg_nnz_per_row = sum_nnz / row;
    geometry->max_column_span = max_col - min_col + 1;
    
    // Calculate row variance
    double variance_sum = 0.0;
    for (int i = 0; i < row; i++) {
        double diff = row_nnz[i] - geometry->avg_nnz_per_row;
        variance_sum += diff * diff;
    }
    geometry->row_variance = variance_sum / row;
    
    // Calculate row imbalance factor
    geometry->row_imbalance_factor = (geometry->max_nnz_per_row > 0) ? 
        (double)geometry->max_nnz_per_row / geometry->avg_nnz_per_row : 0.0;
    
    // Analyze bandwidth
    int max_lower = 0, max_upper = 0;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (matrix[i][j] != 0.0) {
                int lower_dist = i - j;
                int upper_dist = j - i;
                if (lower_dist > max_lower) max_lower = lower_dist;
                if (upper_dist > max_upper) max_upper = upper_dist;
            }
        }
    }
    geometry->lower_bandwidth = max_lower;
    geometry->upper_bandwidth = max_upper;
    geometry->bandwidth = max_lower + max_upper + 1;
    
    // Calculate diagonal dominance
    double diag_sum = 0.0, off_diag_sum = 0.0;
    int diag_elements = 0;
    for (int i = 0; i < row && i < col; i++) {
        if (matrix[i][i] != 0.0) {
            diag_sum += fabs(matrix[i][i]);
            diag_elements++;
        }
        for (int j = 0; j < col; j++) {
            if (i != j && matrix[i][j] != 0.0) {
                off_diag_sum += fabs(matrix[i][j]);
            }
        }
    }
    geometry->diagonal_dominance = (off_diag_sum > 0) ? diag_sum / off_diag_sum : 
        (diag_elements > 0 ? 1.0 : 0.0);
    
    // Calculate locality score (simplified measure of column access locality)
    double locality_sum = 0.0;
    int locality_count = 0;
    for (int i = 0; i < row; i++) {
        int prev_col = -1;
        for (int j = 0; j < col; j++) {
            if (matrix[i][j] != 0.0) {
                if (prev_col >= 0) {
                    locality_sum += 1.0 / (1.0 + abs(j - prev_col));
                    locality_count++;
                }
                prev_col = j;
            }
        }
    }
    geometry->locality_score = (locality_count > 0) ? locality_sum / locality_count : 0.0;
    
    // Suggest block size based on average row density
    if (geometry->avg_nnz_per_row < 8) {
        geometry->suggested_block_size = 128;
    } else if (geometry->avg_nnz_per_row < 32) {
        geometry->suggested_block_size = 256;
    } else {
        geometry->suggested_block_size = 512;
    }
    
    // Recommend strategy based on matrix characteristics
    if (geometry->density > 0.1 || geometry->avg_nnz_per_row > 32) {
        geometry->recommended_strategy = STRATEGY_BLOCK_BASED;
    } else if (geometry->locality_score > 0.7) {
        geometry->recommended_strategy = STRATEGY_VECTORIZED;
    } else {
        geometry->recommended_strategy = STRATEGY_SIMPLE;
    }
    
    free(row_nnz);
    printf("Matrix geometry analysis completed.\n");
}

void mtx_print_geometry(const matrix_geometry_t* geometry) {
    printf("\n=== Matrix Geometry Analysis ===\n");
    printf("Dimensions: %dx%d\n", geometry->rows, geometry->cols);
    printf("Non-zeros: %d (density: %.6f%%)\n", geometry->nnz, geometry->density * 100.0);
    
    printf("\nRow Distribution:\n");
    printf("  Min NNZ per row: %d\n", geometry->min_nnz_per_row);
    printf("  Max NNZ per row: %d\n", geometry->max_nnz_per_row);
    printf("  Avg NNZ per row: %.2f\n", geometry->avg_nnz_per_row);
    printf("  Row variance: %.2f\n", geometry->row_variance);
    printf("  Empty rows: %d\n", geometry->empty_rows);
    printf("  Full rows: %d\n", geometry->full_rows);
    printf("  Row imbalance factor: %.2f\n", geometry->row_imbalance_factor);
    
    printf("\nBandwidth Analysis:\n");
    printf("  Total bandwidth: %d\n", geometry->bandwidth);
    printf("  Lower bandwidth: %d\n", geometry->lower_bandwidth);
    printf("  Upper bandwidth: %d\n", geometry->upper_bandwidth);
    printf("  Max column span: %d\n", geometry->max_column_span);
    
    printf("\nStructural Properties:\n");
    printf("  Diagonal dominance: %.3f\n", geometry->diagonal_dominance);
    printf("  Locality score: %.3f\n", geometry->locality_score);
    printf("  Suggested block size: %d\n", geometry->suggested_block_size);
    
    printf("\nRecommended Strategy: ");
    switch (geometry->recommended_strategy) {
        case STRATEGY_SIMPLE:
            printf("Simple (low density, regular pattern)\n");
            break;
        case STRATEGY_VECTORIZED:
            printf("Vectorized with Caching (good locality)\n");
            break;
        case STRATEGY_BLOCK_BASED:
            printf("Block-based with Cooperative Caching (dense, high NNZ per row)\n");
            break;
    }
    printf("================================\n\n");
}

const char* mtx_get_strategy_name(const matrix_geometry_t* geometry) {
    switch (geometry->recommended_strategy) {
        case STRATEGY_SIMPLE:
            return "Simple Sequential";
        case STRATEGY_VECTORIZED:
            return "Vectorized with Caching";
        case STRATEGY_BLOCK_BASED:
            return "Block-Based with Cooperative Caching";
        default:
            return "Unknown Strategy";
    }
}
