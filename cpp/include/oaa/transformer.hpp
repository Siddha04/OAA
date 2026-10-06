#pragma once

#include "oaa/model.hpp"
#include "oaa/tensor.hpp"
#include "oaa/types.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace oaa {

class KvCache {
public:
    explicit KvCache(std::size_t num_layers = 0);

    void reset();
    void append(
        std::size_t layer,
        const std::vector<float>& key,
        const std::vector<float>& value);

    std::size_t size() const noexcept;
    std::size_t num_layers() const noexcept;

    const std::vector<float>& keys(std::size_t layer) const;
    const std::vector<float>& values(std::size_t layer) const;

private:
    struct LayerCache {
        std::vector<float> key_data;
        std::vector<float> value_data;
    };

    std::vector<LayerCache> layers_;
    std::size_t tokens_{0};
};

class TransformerModel {
public:
    explicit TransformerModel(const ModelConfig& config);

    const ModelConfig& config() const noexcept;

    void reset_cache();
    std::vector<float> forward_token(std::uint32_t token_id);

    std::vector<std::uint32_t> generate_tokens(
        const std::vector<std::uint32_t>& prompt_tokens,
        const GenerationConfig& config);

private:
    struct BlockWeights {
        Tensor norm_attn;
        Tensor wq;
        Tensor wk;
        Tensor wv;
        Tensor wo;
        Tensor norm_ff;
        Tensor w1;
        Tensor w2;
    };

    ModelConfig config_;
    Tensor token_embeddings_;
    Tensor position_embeddings_;
    std::vector<BlockWeights> blocks_;
    Tensor final_norm_;
    Tensor lm_head_;

    KvCache cache_;

    void validate_token(std::uint32_t token_id) const;

    std::vector<float> normalize_rms(
        const std::vector<float>& input,
        const Tensor& scale) const;

    static std::vector<float> matmul_row(
        const std::vector<float>& input,
        const Tensor& weights);

    static void add_inplace(
        std::vector<float>& dst,
        const std::vector<float>& src);

    static float gelu(float value);

    std::uint32_t sample_token(
        const std::vector<float>& logits,
        const std::vector<std::uint32_t>& history,
        const GenerationConfig& config,
        std::mt19937_64& rng) const;
};

} // namespace oaa
