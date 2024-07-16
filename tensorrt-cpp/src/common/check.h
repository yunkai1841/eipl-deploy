#ifndef __CHECK_H__
#define __CHECK_H__

#include <string>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call) __cuda_check(call, __FILE__, __LINE__)

static inline void __cuda_check(cudaError_t code, const char *file, const int line) {
    if (code != cudaSuccess) {
        std::string error = cudaGetErrorString(code);
        std::string message = "CUDA error at " + std::string(file) + ":" + std::to_string(line) + ": " + error;
        std::cerr << message << std::endl;
        abort();
    }
}

#endif // __CHECK_H__
