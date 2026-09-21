#include "oaa/cuda_runtime.hpp"

#include <stdexcept>
#include <string>

namespace oaa {

bool CudaRuntime::compiled() {
    return false;
}

bool CudaRuntime::available() {
    return false;
}

std::vector<float> CudaRuntime::vector_add(
    const std::vector<float>&,
    const std::vector<float>&) {

    throw std::runtime_error(
        "CUDA backend is not compiled; configure with OAA_BUILD_CUDA=ON");
}

} // namespace oaa
