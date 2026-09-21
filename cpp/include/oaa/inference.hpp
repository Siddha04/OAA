#pragma once

#include "oaa/tensor.hpp"

namespace oaa {

class CpuInferenceEngine {
public:
    CpuInferenceEngine() = default;

    Tensor linear(const Tensor& input, const Tensor& weights, const Tensor* bias = nullptr) const;
    Tensor softmax(const Tensor& logits) const;
};

} // namespace oaa
