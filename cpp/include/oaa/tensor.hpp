#pragma once

#include <cstddef>
#include <vector>

namespace oaa {

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(const std::vector<std::size_t>& shape);
    Tensor(const std::vector<std::size_t>& shape, const std::vector<float>& values);

    const std::vector<std::size_t>& shape() const noexcept;
    std::size_t rank() const noexcept;
    std::size_t numel() const noexcept;

    float& at(const std::vector<std::size_t>& indices);
    const float& at(const std::vector<std::size_t>& indices) const;

    float* data() noexcept;
    const float* data() const noexcept;

    void fill(float value) noexcept;

    static Tensor matmul_2d(const Tensor& lhs, const Tensor& rhs);
    static Tensor softmax_last_dim(const Tensor& input);

private:
    std::vector<std::size_t> shape_;
    std::vector<float> data_;

    std::size_t offset(const std::vector<std::size_t>& indices) const;
};

} // namespace oaa
