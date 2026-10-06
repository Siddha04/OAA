#include "oaa/engine.hpp"

#include "oaa/tokenizer.hpp"
#include "oaa/transformer.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace oaa {

namespace {

void validate_generation_config(
    const GenerationConfig& config) {

    if (config.max_tokens == 0) {
        throw std::invalid_argument("max_tokens must be greater than zero");
    }

    if (!std::isfinite(config.temperature) ||
        config.temperature < 0.0F) {
        throw std::invalid_argument(
            "temperature must be finite and non-negative");
    }

    if (!std::isfinite(config.top_p) ||
        config.top_p <= 0.0F ||
        config.top_p > 1.0F) {
        throw std::invalid_argument("top_p must be in (0, 1]");
    }

    if (!std::isfinite(config.repetition_penalty) ||
        config.repetition_penalty < 1.0F) {
        throw std::invalid_argument(
            "repetition_penalty must be at least 1");
    }
}

} // namespace

Engine::Engine() = default;

Engine::~Engine() = default;

bool Engine::load_model(const std::string& path) {
    if (path.empty()) {
        throw std::invalid_argument(
            "model manifest path cannot be empty");
    }

    ManifestModelLoader loader;
    const auto candidate = loader.load(path);

    if (candidate.architecture != "tiny_transformer_v1") {
        throw std::invalid_argument(
            "unsupported model architecture: " +
            candidate.architecture);
    }

    tokenizer_ = create_printable_ascii_tokenizer();

    if (candidate.vocab_size != tokenizer_->vocab_size()) {
        throw std::invalid_argument(
            "tiny_transformer_v1 requires printable ASCII vocabulary size 95");
    }

    model_config_ = candidate;
    model_ = std::make_unique<TransformerModel>(model_config_);
    model_path_ = path;
    model_loaded_ = true;
    return true;
}

void Engine::unload() {
    model_.reset();
    tokenizer_.reset();
    model_config_ = {};
    model_loaded_ = false;
    model_path_.clear();
}

std::vector<std::uint32_t> Engine::generate_tokens(
    const std::string& prompt,
    const GenerationConfig& config) {

    validate_generation_config(config);

    if (!model_loaded_ || !model_ || !tokenizer_) {
        throw std::runtime_error("engine has no loaded model");
    }

    if (prompt.empty()) {
        throw std::invalid_argument("prompt cannot be empty");
    }

    const auto prompt_tokens = tokenizer_->encode(prompt);
    if (prompt_tokens.empty()) {
        throw std::invalid_argument("prompt produced no tokens");
    }

    const auto started = std::chrono::steady_clock::now();
    const auto generated =
        model_->generate_tokens(prompt_tokens, config);
    const auto elapsed =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started);

    ++generation_calls_;
    prompt_tokens_ += prompt_tokens.size();
    generated_tokens_ += generated.size();
    last_generation_seconds_ = elapsed.count();

    return generated;
}

std::string Engine::generate(
    const std::string& prompt,
    const GenerationConfig& config) {

    const auto generated = generate_tokens(prompt, config);
    return tokenizer_->decode(generated);
}

std::vector<std::string> Engine::generate_stream(
    const std::string& prompt,
    const GenerationConfig& config) {

    const auto generated = generate_tokens(prompt, config);

    std::vector<std::string> chunks;
    chunks.reserve(generated.size());

    for (const auto token : generated) {
        chunks.push_back(tokenizer_->decode({token}));
    }

    return chunks;
}

RuntimeStats Engine::get_stats() const {
    return RuntimeStats{
        model_loaded_,
        generation_calls_,
        prompt_tokens_,
        generated_tokens_,
        last_generation_seconds_,
        model_path_};
}

bool Engine::loaded() const {
    return model_loaded_;
}

std::string Engine::status() const {
    return model_loaded_ ? "ready" : "empty";
}

} // namespace oaa
