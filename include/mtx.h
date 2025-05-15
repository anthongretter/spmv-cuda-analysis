#ifndef SPMV_MTX_H
#define SPMV_MTX_H

#include <stdio.h>
#include <stdlib.h>

#include "config.h"
#include "commons.h"

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
 * @return Matrix with the parsed values
 */
MATRIX_T mtx_parse(FILE* file, int* row, int* col, int* n);

/**
 * Closes a Matrix Market file
 *
 * @param file File pointer to an open Matrix Market file
 */
void mtx_fclose(FILE* file);

#endif /* SPMV_MTX_H */
