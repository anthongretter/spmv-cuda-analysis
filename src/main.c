#include <stdio.h>

#include "config.h"
#include "mtx.h"
#include "matrix.h"
#include "commons.h"

int main(int argc, const char *argv[])
{
    int row, col, n;
    VEC_T result;

    FILE* mtx_file = mtx_fopen_from_cli(argc, argv);
    MATRIX_T matrix = mtx_parse(mtx_file, &row, &col, &n);
    mtx_fclose(mtx_file);

    struct CSR csr = matrix_csr_format(row, col, n, matrix);
    VEC_T ones = vec_ones(col);
    ALLOC(result, row * sizeof(PRIM_T));

    long ops = 0;

    printf("Performing sparse matrix-vector multiplication...");
    clock_t start = TIMER_START();
    SPMV(&csr, ones, result, &ops);
    clock_t stop = TIMER_STOP();

    double elapsed = (double) (stop - start) / CLOCKS_PER_SEC;
    printf("\nElapsed time: %f seconds\n", elapsed);
    printf("Total operations: %ld\n", ops);
    printf("Total FLOPs: %.5lf\n", (double) ops / elapsed);

    FREE(csr.row);
    FREE(csr.val);
    matrix_free(matrix);
    return 0;
}