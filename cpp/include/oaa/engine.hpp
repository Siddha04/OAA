#pragma once

#include "oaa/model.hpp"
#include "oaa/types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace oaa {

class Tokenizer;
class TransformerModel;

class Engine {
public:
    Engine() = default;

    bool load_model(const std::string& path);
    void unload();

    std::string generate(
        const std::string& prompt,
        const GenerationConfig& config = {});

    std::vector<std::string> generate_stream(
        const std::string& prompt,
        const GenerationConfig& config = {});

    RuntimeStats get_stats() const;
    bool loaded() const;
    std::string status() const;

private:
    bool model_loaded_{false};
    std::string model_path_;
    std::uint64_t generation_calls_{0};
    std::uint64_t prompt_tokens_{0};
    std::uint64_t generated_tokens_{0};
    double last_generation_seconds_{0.0};

    ModelConfig model_config_{};
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<TransformerModel> model_;
    
    std::vector<std::uint32_t> generate_tokens(
        const std::string& prompt,
        const GenerationConfig& config);
};

} // namespace oaa
