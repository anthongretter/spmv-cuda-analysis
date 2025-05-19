#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

MATRIX_T mtx_parse(FILE* file, int* row, int* col, int* n) {
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
    return matrix;
}
