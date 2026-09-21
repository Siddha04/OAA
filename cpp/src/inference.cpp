#include "oaa/inference.hpp"

#include <stdexcept>

namespace oaa {

Tensor CpuInferenceEngine::linear(
    const Tensor& input,
    const Tensor& weights,
    const Tensor* bias) const {

    Tensor output = Tensor::matmul_2d(input, weights);

    if (bias != nullptr) {
        if (bias->rank() != 1 || bias->shape()[0] != output.shape()[1]) {
            throw std::invalid_argument("bias shape must match linear output width");
        }

        for (std::size_t row = 0; row < output.shape()[0]; ++row) {
            for (std::size_t col = 0; col < output.shape()[1]; ++col) {
                output.at({row, col}) += bias->at({col});
            }
        }
    }

    return output;
}

Tensor CpuInferenceEngine::softmax(const Tensor& logits) const {
    return Tensor::softmax_last_dim(logits);
}

} // namespace oaa
