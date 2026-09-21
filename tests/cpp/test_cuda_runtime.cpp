#include "oaa/cuda_runtime.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
#if OAA_HAS_CUDA
    assert(oaa::CudaRuntime::compiled());

    if (oaa::CudaRuntime::available()) {
        const auto output = oaa::CudaRuntime::vector_add(
            {1.0F, 2.0F, 3.0F},
            {4.0F, 5.0F, 6.0F});

        assert(output.size() == 3);
        assert(std::fabs(output[0] - 5.0F) < 1e-5F);
        assert(std::fabs(output[1] - 7.0F) < 1e-5F);
        assert(std::fabs(output[2] - 9.0F) < 1e-5F);
    }
#else
    assert(!oaa::CudaRuntime::compiled());
    assert(!oaa::CudaRuntime::available());

    bool threw = false;
    try {
        (void)oaa::CudaRuntime::vector_add({1.0F}, {2.0F});
    } catch (const std::runtime_error&) {
        threw = true;
    }

    assert(threw);
#endif

    std::cout << "OAA CUDA backend tests passed" << std::endl;
    return 0;
}
