#pragma once

#include <vector>

namespace oaa {

class CudaRuntime {
public:
    static bool compiled();
    static bool available();

    static std::vector<float> vector_add(
        const std::vector<float>& lhs,
        const std::vector<float>& rhs);
};

} // namespace oaa
