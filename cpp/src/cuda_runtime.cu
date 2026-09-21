#include "oaa/cuda_runtime.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace oaa {

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__global__ void vector_add_kernel(
    const float* lhs,
    const float* rhs,
    float* output,
    std::size_t size) {

    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index < size) {
        output[index] = lhs[index] + rhs[index];
    }
}

} // namespace

bool CudaRuntime::compiled() {
    return true;
}

bool CudaRuntime::available() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);

    if (status != cudaSuccess) {
        return false;
    }

    return device_count > 0;
}

std::vector<float> CudaRuntime::vector_add(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs) {

    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("vector_add requires equal input sizes");
    }

    if (!available()) {
        throw std::runtime_error(
            "CUDA backend is compiled but no usable CUDA device is available");
    }

    if (lhs.empty()) {
        return {};
    }

    const std::size_t bytes = lhs.size() * sizeof(float);

    float* d_lhs = nullptr;
    float* d_rhs = nullptr;
    float* d_output = nullptr;

    try {
        check_cuda(cudaMalloc(&d_lhs, bytes), "cudaMalloc(lhs)");
        check_cuda(cudaMalloc(&d_rhs, bytes), "cudaMalloc(rhs)");
        check_cuda(cudaMalloc(&d_output, bytes), "cudaMalloc(output)");

        check_cuda(
            cudaMemcpy(d_lhs, lhs.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy(lhs)");
        check_cuda(
            cudaMemcpy(d_rhs, rhs.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy(rhs)");

        constexpr unsigned int threads = 256;
        const unsigned int blocks = static_cast<unsigned int>(
            (lhs.size() + threads - 1) / threads);

        vector_add_kernel<<<blocks, threads>>>(d_lhs, d_rhs, d_output, lhs.size());
        check_cuda(cudaGetLastError(), "vector_add_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "vector_add_kernel synchronize");

        std::vector<float> output(lhs.size());
        check_cuda(
            cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy(output)");

        cudaFree(d_lhs);
        cudaFree(d_rhs);
        cudaFree(d_output);

        return output;
    } catch (...) {
        cudaFree(d_lhs);
        cudaFree(d_rhs);
        cudaFree(d_output);
        throw;
    }
}

} // namespace oaa
