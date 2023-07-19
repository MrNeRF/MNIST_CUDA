#pragma once

#include <cuda_runtime.h>
#include <iostream>

#undef DEBUG_ERRORS

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
inline void check(T err, const char* const func, const char* const file,
                  const int line) {
#ifdef DEBUG_ERRORS
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
#endif // DEBUG_ERRORS
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
inline void checkLast(const char* const file, const int line) {
#ifdef DEBUG_ERRORS
    cudaDeviceSynchronize();
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
#endif // DEBUG_ERRORS
}