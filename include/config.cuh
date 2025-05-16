#ifndef SPMV_CONFIG_H
#define SPMV_CONFIG_H

/**
 * Configuration header for SPMV program
 * 
 * Includes the appropriate implementation header (GPU or CPU)
 * based on compilation flag (-DGPU)
 */

#if defined GPU
#include "gpu.cuh"
#else
#include "cpu.cuh"
#endif

#endif /* SPMV_CONFIG_H */
