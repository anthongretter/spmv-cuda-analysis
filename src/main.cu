#include <stdio.h>

#include "config.cuh"
#include "mtx.cuh"
#include "matrix.cuh"
#include "commons.cuh"

int main(int argc, const char *argv[])
{
    int row, col, n;
    double elapsed;
    TIMER_T start, stop;

    FILE* mtx_file = mtx_fopen_from_cli(argc, argv);
    MATRIX_T matrix = mtx_parse(mtx_file, &row, &col, &n);
    mtx_fclose(mtx_file);

    SETUP();
    void *ptr_matrix = setup_matrix(row, col, n, matrix);

    VEC_T vec = vec_rand(col);
    VEC_T result;
    ALLOC(result, row * sizeof(PRIM_T));

    printf("\nPerforming sparse matrix-vector multiplication...\n");
    TIMER_START(start);
    SPMV(row, col, n, ptr_matrix, vec, result);
    TIMER_STOP(stop);

    TIMER_DIFF(start, stop, &elapsed);
    printf("Elapsed time: %f seconds\n", elapsed);

    FREE(vec);
    matrix_free(matrix);
    return 0;
}