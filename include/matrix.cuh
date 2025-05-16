#ifndef SPMV_MATRIX_H
#define SPMV_MATRIX_H

#include <stdio.h>
#include <stdlib.h>

#include "config.cuh"
#include "commons.cuh"

/**
 * Structure representing a compressed sparse row matrix
 */
struct CSR {
    int* row;
    int* col;
    VEC_T val;
};

/**
 * Allocates memory for a matrix of the specified dimensions
 *
 * @param row Number of rows in the matrix
 * @param col Number of columns in the matrix
 * @return Pointer to allocated memory or NULL if allocation fails
 */
MATRIX_T matrix_alloc(int row, int col);

/**
 * Frees memory allocated for a matrix
 *
 * @param matrix Pointer to the matrix to be freed
 */
void matrix_free(MATRIX_T matrix);

/**
 * Prints the contents of a matrix to stdout
 *
 * @param row Number of rows in the matrix
 * @param col Number of columns in the matrix
 * @param matrix Pointer to the matrix matrix
 */
void matrix_print(int row, int col, MATRIX_T matrix);

/**
 * Converts a matrix to CSR format
 *
 * @param row Number of rows in the matrix
 * @param col Number of columns in the matrix
 * @param n Number of non-zero elements in the matrix
 * @param matrix Pointer to the matrix to be converted
 * @return CSR matrix structure
 */
struct CSR* matrix_csr_format(int row, int col, int n, MATRIX_T matrix);

/**
 * Prints the CSR representation of a matrix to stdout
 *
 * @param row Number of rows in the matrix
 * @param col Number of columns in the matrix
 * @param n Number of non-zero elements in the matrix
 * @param csr CSR structure containing the compressed sparse representation
 */
void matrix_print_csr(int row, int col, int n, struct CSR csr);

/**
 * Creates and initializes a random vector
 *
 * @param length Length of the vector
 * @return A newly allocated vector
 */
VEC_T vec_rand(int length);

/**
 * Prints a vector
 */
void vec_print(VEC_T vec, int length);

#endif /* SPMV_MATRIX_H */
