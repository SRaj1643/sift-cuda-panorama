#ifndef CORE_UTILS_CUH
#define CORE_UTILS_CUH

#include <cuda_runtime.h>
#include <iostream>

// Macro to catch and report CUDA runtime errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#endif // CORE_UTILS_CUH
