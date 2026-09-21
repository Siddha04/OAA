#include "oaa/engine.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace oaa {

namespace {

void validate_generation_config(const GenerationConfig& config) {
    if (config.max_tokens == 0) {
        throw std::invalid_argument("max_tokens must be greater than zero");
    }

    if (!std::isfinite(config.temperature) || config.temperature < 0.0F) {
        throw std::invalid_argument("temperature must be finite and non-negative");
    }
}

} // namespace

bool Engine::load_model(const std::string& path) {
    if (path.empty()) {
        throw std::invalid_argument("model path cannot be empty");
    }

    model_path_ = path;
    model_loaded_ = true;
    return true;
}

void Engine::unload() {
    model_loaded_ = false;
    model_path_.clear();
}

std::string Engine::generate(
    const std::string& prompt,
    const GenerationConfig& config) {

    validate_generation_config(config);

    if (!model_loaded_) {
        throw std::runtime_error("engine has no loaded model");
    }

    ++generation_calls_;
    return "OAA_PHASE1_ECHO: " + prompt;
}

std::vector<std::string> Engine::generate_stream(
    const std::string& prompt,
    const GenerationConfig& config) {

    const std::string result = generate(prompt, config);
    constexpr std::size_t chunk_size = 8;

    std::vector<std::string> chunks;
    chunks.reserve((result.size() + chunk_size - 1) / chunk_size);

    for (std::size_t offset = 0; offset < result.size(); offset += chunk_size) {
        chunks.push_back(result.substr(offset, chunk_size));
    }

    return chunks;
}

RuntimeStats Engine::get_stats() const {
    return RuntimeStats{model_loaded_, generation_calls_, model_path_};
}

bool Engine::loaded() const {
    return model_loaded_;
}

std::string Engine::status() const {
    return model_loaded_ ? "ready" : "empty";
}

} // namespace oaa
