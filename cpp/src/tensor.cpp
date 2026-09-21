#include "oaa/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace oaa {

namespace {

std::size_t checked_numel(const std::vector<std::size_t>& shape) {
    std::size_t result = 1;
    for (const auto dim : shape) {
        if (dim == 0 || result > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::invalid_argument("tensor shape contains an invalid size");
        }
        result *= dim;
    }
    return result;
}

void validate_rank_2(const Tensor& tensor, const char* name) {
    if (tensor.rank() != 2) {
        throw std::invalid_argument(std::string(name) + " must be rank 2");
    }
}

} // namespace

Tensor::Tensor(const std::vector<std::size_t>& shape)
    : shape_(shape), data_(checked_numel(shape), 0.0F) {}

Tensor::Tensor(const std::vector<std::size_t>& shape, const std::vector<float>& values)
    : shape_(shape), data_(values) {
    if (data_.size() != checked_numel(shape_)) {
        throw std::invalid_argument("tensor values do not match shape");
    }
}

const std::vector<std::size_t>& Tensor::shape() const noexcept {
    return shape_;
}

std::size_t Tensor::rank() const noexcept {
    return shape_.size();
}

std::size_t Tensor::numel() const noexcept {
    return data_.size();
}

float& Tensor::at(const std::vector<std::size_t>& indices) {
    return data_.at(offset(indices));
}

const float& Tensor::at(const std::vector<std::size_t>& indices) const {
    return data_.at(offset(indices));
}

float* Tensor::data() noexcept {
    return data_.data();
}

const float* Tensor::data() const noexcept {
    return data_.data();
}

void Tensor::fill(float value) noexcept {
    std::fill(data_.begin(), data_.end(), value);
}

std::size_t Tensor::offset(const std::vector<std::size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("tensor index rank mismatch");
    }

    std::size_t linear = 0;
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("tensor index out of range");
        }
        linear = linear * shape_[i] + indices[i];
    }
    return linear;
}

Tensor Tensor::matmul_2d(const Tensor& lhs, const Tensor& rhs) {
    validate_rank_2(lhs, "lhs");
    validate_rank_2(rhs, "rhs");

    const auto lhs_rows = lhs.shape()[0];
    const auto inner = lhs.shape()[1];
    const auto rhs_rows = rhs.shape()[0];
    const auto rhs_cols = rhs.shape()[1];

    if (inner != rhs_rows) {
        throw std::invalid_argument("matmul dimension mismatch");
    }

    Tensor result({lhs_rows, rhs_cols});
    for (std::size_t i = 0; i < lhs_rows; ++i) {
        for (std::size_t k = 0; k < inner; ++k) {
            const float lhs_value = lhs.at({i, k});
            for (std::size_t j = 0; j < rhs_cols; ++j) {
                result.at({i, j}) += lhs_value * rhs.at({k, j});
            }
        }
    }
    return result;
}

Tensor Tensor::softmax_last_dim(const Tensor& input) {
    if (input.rank() == 0) {
        throw std::invalid_argument("softmax requires a non-scalar tensor");
    }

    Tensor result(input.shape());
    const std::size_t width = input.shape().back();
    const std::size_t rows = input.numel() / width;

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t base = row * width;
        float max_value = -std::numeric_limits<float>::infinity();
        for (std::size_t j = 0; j < width; ++j) {
            max_value = std::max(max_value, input.data()[base + j]);
        }

        float sum = 0.0F;
        for (std::size_t j = 0; j < width; ++j) {
            const float value = std::exp(input.data()[base + j] - max_value);
            result.data()[base + j] = value;
            sum += value;
        }

        if (!(sum > 0.0F) || !std::isfinite(sum)) {
            throw std::runtime_error("softmax normalization failed");
        }

        for (std::size_t j = 0; j < width; ++j) {
            result.data()[base + j] /= sum;
        }
    }

    return result;
}

} // namespace oaa
