#ifndef SPMV_MTX_H
#define SPMV_MTX_H

#include <stdio.h>
#include <stdlib.h>

#include "config.cuh"
#include "commons.cuh"

/**
 * Adaptive strategy enumeration
 */
typedef enum {
    STRATEGY_SIMPLE,
    STRATEGY_VECTORIZED,
    STRATEGY_BLOCK_BASED
} spmv_strategy_t;

/**
 * Matrix geometry information for adaptive SPMV strategies
 */
typedef struct {
    // Basic statistics
    int rows;
    int cols;
    int nnz;
    double density;
    
    // Row distribution
    int min_nnz_per_row;
    int max_nnz_per_row;
    double avg_nnz_per_row;
    double row_variance;
    
    // Sparsity pattern analysis
    int empty_rows;
    int full_rows;
    double row_imbalance_factor;
    
    // Band structure
    int bandwidth;
    int lower_bandwidth;
    int upper_bandwidth;
    
    // Block structure hints
    int suggested_block_size;
    double diagonal_dominance;
    
    // Memory access patterns
    double locality_score;
    int max_column_span;
    
    // Adaptive strategy recommendations
    spmv_strategy_t recommended_strategy;
    
} matrix_geometry_t;

/**
 * Opens a Matrix Market file specified via command line arguments
 * 
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @return File pointer to the opened Matrix Market file
 */
FILE* mtx_fopen_from_cli(int argc, const char *argv[]);

/**
 * Parses a Matrix Market file and loads its contents into a matrix
 * 
 * @param file File pointer to an open Matrix Market file
 * @param row Pointer to store the number of rows
 * @param col Pointer to store the number of columns
 * @param n Pointer to store the number of non-zero elements
 * @param geometry Pointer to store matrix geometry information (optional, can be NULL)
 * @return Matrix with the parsed values
 */
MATRIX_T mtx_parse(FILE* file, int* row, int* col, int* n, matrix_geometry_t* geometry);

/**
 * Analyzes matrix geometry for adaptive SPMV strategy selection
 * 
 * @param matrix The matrix to analyze
 * @param row Number of rows
 * @param col Number of columns
 * @param n Number of non-zero elements
 * @param geometry Pointer to store the computed geometry information
 */
void mtx_analyze_geometry(MATRIX_T matrix, int row, int col, int n, matrix_geometry_t* geometry);

/**
 * Prints matrix geometry information
 * 
 * @param geometry Matrix geometry structure to print
 */
void mtx_print_geometry(const matrix_geometry_t* geometry);

/**
 * Gets the name of the recommended SPMV strategy
 * 
 * @param geometry Matrix geometry information
 * @return String describing the recommended strategy
 */
const char* mtx_get_strategy_name(const matrix_geometry_t* geometry);

/**
 * Closes a Matrix Market file
 *
 * @param file File pointer to an open Matrix Market file
 */
void mtx_fclose(FILE* file);

#endif /* SPMV_MTX_H */
