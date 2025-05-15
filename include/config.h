#ifndef SPMV_CONFIG_H
#define SPMV_CONFIG_H

/**
 * Configuration header for SPMV program
 * 
 * Includes the appropriate implementation header (GPU or CPU)
 * based on compilation flag GPU_IMP
 */

#ifdef GPU_IMP
#include "gpu.h"
#else
#include "cpu.h"
#endif

//#define ROUTINE(csr_row, csr_col, csr_val, m, n, vec, result, timer) \
//    do {                                                      \
//        SETUP();                                              \
//        int \
//    } while(0)

#endif /* SPMV_CONFIG_H */
