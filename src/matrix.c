#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "matrix.h"
#include "commons.h"


MATRIX_T matrix_alloc(int row, int col) {
    // Allocate memory for row pointers
    MATRIX_T matrix;
    ALLOC(matrix, row * sizeof(VEC_T));
    if (!matrix) {
        FREE(matrix);
        return NULL;
    }

    // Allocate memory for the entire matrix data in one contiguous block
    VEC_T data;
    ALLOC(data, row * col * sizeof(PRIM_T));
    if (!data) {
        FREE(matrix);
        FREE(data);
        return NULL;
    }

    // Set up row pointers to point to appropriate positions in the data block
    for (int i = 0; i < row; i++) {
        matrix[i] = &data[i * col];
    }

    return matrix;
}

void matrix_free(MATRIX_T matrix) {
    if (matrix) {
        FREE(matrix[0]);
        FREE(matrix);
    }
}

void matrix_print(int row, int col, MATRIX_T matrix) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
}

void matrix_print_csr(int row, int col, int n, struct CSR csr) {
    printf("CSR row: ");
    for (int i = 0; i < row + 1; i++) {
        printf("%d ", csr.row[i]);
    }
    printf("\nCSR values: ");
    for (int i = 0; i < n; i++) {
        printf("%f ", csr.val[i]);
    }
    printf("\n");
}

VEC_T vec_ones(int length) {
    VEC_T vec;
    ALLOC(vec, length * sizeof(PRIM_T));

    for (int i = 0; i < length; i++) {
        vec[i] = 1;
    }

    return vec;
}

struct CSR matrix_csr_format(int row, int col, int n, MATRIX_T matrix) {
    int *csr_row, *csr_col;
    VEC_T csr_val;

    ALLOC(csr_row, (row + 1) * sizeof(int));
    ALLOC(csr_col, n * sizeof(int));
    ALLOC(csr_val, n * sizeof(PRIM_T));

    csr_row[0] = 0;
    int non_zero_row = 0;

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (matrix[i][j]) {
                csr_val[non_zero_row] = matrix[i][j];
                csr_col[non_zero_row] = j;
                non_zero_row++;
            }
        }
        csr_row[i + 1] = non_zero_row;
    }
    csr_row[row] = n;
    return (struct CSR) {row, csr_row, csr_col, csr_val };
}