#include "oaa/transformer.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace oaa {

namespace {

void fill_random(Tensor& tensor, std::mt19937_64& rng, float scale) {
    std::normal_distribution<float> distribution(0.0F, scale);
    for (std::size_t i = 0; i < tensor.numel(); ++i) {
        tensor.data()[i] = distribution(rng);
    }
}

std::vector<float> softmax_vector(const std::vector<float>& logits) {
    if (logits.empty()) {
        throw std::invalid_argument("cannot softmax an empty vector");
    }

    const float max_value = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probabilities(logits.size());

    float sum = 0.0F;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i] - max_value);
        sum += probabilities[i];
    }

    if (!(sum > 0.0F) || !std::isfinite(sum)) {
        throw std::runtime_error("sampling softmax failed");
    }

    for (float& probability : probabilities) {
        probability /= sum;
    }
    return probabilities;
}

} // namespace

KvCache::KvCache(std::size_t num_layers) : layers_(num_layers) {}

void KvCache::reset() {
    for (auto& layer : layers_) {
        layer.key_data.clear();
        layer.value_data.clear();
    }
    tokens_ = 0;
}

void KvCache::append(
    std::size_t layer,
    const std::vector<float>& key,
    const std::vector<float>& value) {

    if (layer >= layers_.size()) {
        throw std::out_of_range("KV cache layer out of range");
    }
    if (key.size() != value.size()) {
        throw std::invalid_argument("KV cache key/value size mismatch");
    }

    layers_[layer].key_data.insert(
        layers_[layer].key_data.end(), key.begin(), key.end());
    layers_[layer].value_data.insert(
        layers_[layer].value_data.end(), value.begin(), value.end());

    if (layer == 0) {
        ++tokens_;
    }
}

std::size_t KvCache::size() const noexcept {
    return tokens_;
}

std::size_t KvCache::num_layers() const noexcept {
    return layers_.size();
}

const std::vector<float>& KvCache::keys(std::size_t layer) const {
    return layers_.at(layer).key_data;
}

const std::vector<float>& KvCache::values(std::size_t layer) const {
    return layers_.at(layer).value_data;
}

TransformerModel::TransformerModel(const ModelConfig& config)
    : config_(config),
      token_embeddings_({config.vocab_size, config.hidden_size}),
      position_embeddings_({config.context_length, config.hidden_size}),
      blocks_(),
      final_norm_({config.hidden_size}),
      lm_head_({config.hidden_size, config.vocab_size}),
      cache_(config.num_layers) {

    if (config_.vocab_size == 0 || config_.hidden_size == 0 ||
        config_.num_layers == 0 || config_.num_heads == 0 ||
        config_.intermediate_size == 0 || config_.context_length == 0) {
        throw std::invalid_argument("invalid transformer configuration");
    }

    if (config_.hidden_size % config_.num_heads != 0) {
        throw std::invalid_argument(
            "hidden_size must be divisible by num_heads");
    }

    std::mt19937_64 rng(config_.seed);
    const float scale =
        1.0F / std::sqrt(static_cast<float>(config_.hidden_size));

    fill_random(token_embeddings_, rng, scale);
    fill_random(position_embeddings_, rng, scale);
    final_norm_.fill(1.0F);

    blocks_.reserve(config_.num_layers);
    for (std::size_t layer = 0; layer < config_.num_layers; ++layer) {
        BlockWeights block{
            Tensor({config_.hidden_size}),
            Tensor({config_.hidden_size, config_.hidden_size}),
            Tensor({config_.hidden_size, config_.hidden_size}),
            Tensor({config_.hidden_size, config_.hidden_size}),
            Tensor({config_.hidden_size, config_.hidden_size}),
            Tensor({config_.hidden_size}),
            Tensor({config_.hidden_size, config_.intermediate_size}),
            Tensor({config_.intermediate_size, config_.hidden_size})};

        block.norm_attn.fill(1.0F);
        block.norm_ff.fill(1.0F);
        fill_random(block.wq, rng, scale);
        fill_random(block.wk, rng, scale);
        fill_random(block.wv, rng, scale);
        fill_random(block.wo, rng, scale);
        fill_random(block.w1, rng, scale);
        fill_random(block.w2, rng, scale);
        blocks_.push_back(std::move(block));
    }

    fill_random(lm_head_, rng, scale);
}

const ModelConfig& TransformerModel::config() const noexcept {
    return config_;
}

void TransformerModel::reset_cache() {
    cache_.reset();
}

void TransformerModel::validate_token(std::uint32_t token_id) const {
    if (token_id >= config_.vocab_size) {
        throw std::out_of_range("token id is outside model vocabulary");
    }
}

std::vector<float> TransformerModel::normalize_rms(
    const std::vector<float>& input,
    const Tensor& scale) const {

    if (input.size() != scale.numel()) {
        throw std::invalid_argument("RMS norm dimension mismatch");
    }

    float sum_squares = 0.0F;
    for (const float value : input) {
        sum_squares += value * value;
    }

    const float mean_square =
        sum_squares / static_cast<float>(input.size());
    const float inverse_rms =
        1.0F / std::sqrt(mean_square + 1e-5F);

    std::vector<float> output(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] * inverse_rms * scale.data()[i];
    }
    return output;
}

std::vector<float> TransformerModel::matmul_row(
    const std::vector<float>& input,
    const Tensor& weights) {

    if (weights.rank() != 2 || input.size() != weights.shape()[0]) {
        throw std::invalid_argument("row matmul dimension mismatch");
    }

    std::vector<float> output(weights.shape()[1], 0.0F);
    for (std::size_t k = 0; k < weights.shape()[0]; ++k) {
        for (std::size_t j = 0; j < weights.shape()[1]; ++j) {
            output[j] += input[k] * weights.at({k, j});
        }
    }
    return output;
}

void TransformerModel::add_inplace(
    std::vector<float>& dst,
    const std::vector<float>& src) {

    if (dst.size() != src.size()) {
        throw std::invalid_argument("residual dimension mismatch");
    }

    for (std::size_t i = 0; i < dst.size(); ++i) {
        dst[i] += src[i];
    }
}

float TransformerModel::gelu(float value) {
    constexpr float kSqrt2OverPi = 0.7978845608F;
    return 0.5F * value *
        (1.0F + std::tanh(
            kSqrt2OverPi *
            (value + 0.044715F * value * value * value)));
}

std::vector<float> TransformerModel::forward_token(std::uint32_t token_id) {
    validate_token(token_id);

    if (cache_.size() >= config_.context_length) {
        throw std::runtime_error("transformer context length exceeded");
    }

    const std::size_t position = cache_.size();
    std::vector<float> hidden(config_.hidden_size);

    for (std::size_t i = 0; i < config_.hidden_size; ++i) {
        hidden[i] =
            token_embeddings_.at({token_id, i}) +
            position_embeddings_.at({position, i});
    }

    const std::size_t head_dim = config_.hidden_size / config_.num_heads;
    const float attention_scale =
        1.0F / std::sqrt(static_cast<float>(head_dim));

    for (std::size_t layer = 0; layer < blocks_.size(); ++layer) {
        const auto& block = blocks_[layer];
        const auto normalized = normalize_rms(hidden, block.norm_attn);
        const auto q = matmul_row(normalized, block.wq);
        const auto k = matmul_row(normalized, block.wk);
        const auto v = matmul_row(normalized, block.wv);

        cache_.append(layer, k, v);

        const auto& cached_keys = cache_.keys(layer);
        const auto& cached_values = cache_.values(layer);
        const std::size_t sequence_length = cache_.size();

        std::vector<float> attention_output(config_.hidden_size, 0.0F);

        for (std::size_t head = 0; head < config_.num_heads; ++head) {
            const std::size_t head_offset = head * head_dim;
            std::vector<float> scores(sequence_length);

            for (std::size_t token = 0; token < sequence_length; ++token) {
                const std::size_t cache_offset =
                    token * config_.hidden_size + head_offset;
                float dot = 0.0F;
                for (std::size_t d = 0; d < head_dim; ++d) {
                    dot += q[head_offset + d] *
                        cached_keys[cache_offset + d];
                }
                scores[token] = dot * attention_scale;
            }

            const auto probabilities = softmax_vector(scores);
            for (std::size_t token = 0; token < sequence_length; ++token) {
                const std::size_t cache_offset =
                    token * config_.hidden_size + head_offset;
                for (std::size_t d = 0; d < head_dim; ++d) {
                    attention_output[head_offset + d] +=
                        probabilities[token] *
                        cached_values[cache_offset + d];
                }
            }
        }

        add_inplace(hidden, matmul_row(attention_output, block.wo));

        const auto normalized_ff = normalize_rms(hidden, block.norm_ff);
        auto mlp_hidden = matmul_row(normalized_ff, block.w1);
        for (float& value : mlp_hidden) {
            value = gelu(value);
        }

        add_inplace(hidden, matmul_row(mlp_hidden, block.w2));
    }

    const auto normalized_final = normalize_rms(hidden, final_norm_);
    return matmul_row(normalized_final, lm_head_);
}

std::uint32_t TransformerModel::sample_token(
    const std::vector<float>& logits,
    const std::vector<std::uint32_t>& history,
    const GenerationConfig& config,
    std::mt19937_64& rng) const {

    if (logits.empty()) {
        throw std::invalid_argument("cannot sample from empty logits");
    }

    std::vector<float> adjusted = logits;

    if (config.repetition_penalty > 1.0F) {
        for (const auto token : history) {
            if (token >= adjusted.size()) {
                continue;
            }
            if (adjusted[token] >= 0.0F) {
                adjusted[token] /= config.repetition_penalty;
            } else {
                adjusted[token] *= config.repetition_penalty;
            }
        }
    }

    if (config.temperature == 0.0F) {
        return static_cast<std::uint32_t>(
            std::distance(
                adjusted.begin(),
                std::max_element(adjusted.begin(), adjusted.end())));
    }

    for (float& value : adjusted) {
        value /= config.temperature;
    }

    std::vector<std::size_t> indices(adjusted.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(
        indices.begin(),
        indices.end(),
        [&adjusted](std::size_t lhs, std::size_t rhs) {
            return adjusted[lhs] > adjusted[rhs];
        });

    if (config.top_k > 0 && config.top_k < indices.size()) {
        indices.resize(config.top_k);
    }

    if (config.top_p < 1.0F) {
        const auto all_probabilities = softmax_vector(adjusted);
        std::vector<std::size_t> filtered;
        filtered.reserve(indices.size());

        float cumulative = 0.0F;
        for (const auto index : indices) {
            filtered.push_back(index);
            cumulative += all_probabilities[index];
            if (cumulative >= config.top_p) {
                break;
            }
        }

        indices = std::move(filtered);
    }

    if (indices.empty()) {
        throw std::runtime_error("sampling candidate set is empty");
    }

    std::vector<float> candidate_logits;
    candidate_logits.reserve(indices.size());
    for (const auto index : indices) {
        candidate_logits.push_back(adjusted[index]);
    }

    const auto probabilities = softmax_vector(candidate_logits);
    std::discrete_distribution<std::size_t> distribution(
        probabilities.begin(), probabilities.end());

    return static_cast<std::uint32_t>(
        indices[distribution(rng)]);
}

std::vector<std::uint32_t> TransformerModel::generate_tokens(
    const std::vector<std::uint32_t>& prompt_tokens,
    const GenerationConfig& config) {

    if (prompt_tokens.empty()) {
        throw std::invalid_argument("prompt must contain at least one token");
    }
    if (prompt_tokens.size() >= config_.context_length) {
        throw std::invalid_argument("prompt is too long for model context");
    }

    reset_cache();

    std::vector<std::uint32_t> history = prompt_tokens;
    std::vector<float> logits;

    for (const auto token : prompt_tokens) {
        logits = forward_token(token);
    }

    std::mt19937_64 rng(config.seed);
    std::vector<std::uint32_t> generated;
    generated.reserve(config.max_tokens);

    for (std::uint32_t step = 0;
         step < config.max_tokens &&
         cache_.size() < config_.context_length;
         ++step) {

        const auto next_token =
            sample_token(logits, history, config, rng);

        generated.push_back(next_token);
        history.push_back(next_token);

        if (step + 1 < config.max_tokens &&
            cache_.size() < config_.context_length) {
            logits = forward_token(next_token);
        }
    }

    return generated;
}

} // namespace oaa
