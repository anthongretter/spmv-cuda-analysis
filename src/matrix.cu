#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.cuh"
#include "matrix.cuh"
#include "commons.cuh"


MATRIX_T matrix_alloc(int row, int col) {
    MATRIX_T matrix;
    ALLOC(matrix, row * sizeof(VEC_T));
    if (!matrix) {
        printf("Failed to allocate memory for row pointers\n");
        return NULL;
    }

    VEC_T data;
    ALLOC(data, row * col * sizeof(PRIM_T));
    if (!data) {
        FREE(matrix);
        printf("Failed to allocate memory for data\n");
        return NULL;
    }

    for (int i = 0; i < row; i++) {
        matrix[i] = &data[i * col];
        for (int j = 0; j < col; j++) {
            matrix[i][j] = 0;
        }
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

struct CSR* matrix_csr_format(int row, int col, int n, MATRIX_T matrix) {
    struct CSR* csr;
    ALLOC(csr, sizeof(struct CSR));
    ALLOC(csr->row, (row + 1) * sizeof(int));
    ALLOC(csr->col, n * sizeof(int));
    ALLOC(csr->val, n * sizeof(PRIM_T));

    csr->row[0] = 0;
    int non_zero_row = 0;

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (matrix[i][j]) {
                csr->val[non_zero_row] = matrix[i][j];
                csr->col[non_zero_row] = j;
                non_zero_row++;
            }
        }
        csr->row[i + 1] = non_zero_row;
    }
    csr->row[row] = n;
    return csr;
}

VEC_T vec_rand(int length) {
    VEC_T vec;
    ALLOC(vec, length * sizeof(PRIM_T));
    srand(time(0));

    for (int i = 0; i < length; i++) {
        vec[i] = rand();
    }
    return vec;
}

void vec_print(VEC_T vec, int length) {
    for (int i = 0; i < length; i++) {
        printf("%.2f ", vec[i]);
    }
}
