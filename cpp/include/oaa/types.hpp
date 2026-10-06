#pragma once

#include <cstdint>
#include <string>

namespace oaa {

struct RuntimeStats {
    bool model_loaded{false};
    std::uint64_t generation_calls{0};
    std::uint64_t prompt_tokens{0};
    std::uint64_t generated_tokens{0};
    double last_generation_seconds{0.0};
    std::string model_path;
};

struct GenerationConfig {
    std::uint32_t max_tokens{128};
    float temperature{0.7F};
    std::uint32_t top_k{0};
    float top_p{1.0F};
    float repetition_penalty{1.0F};
    std::uint64_t seed{0};
};

} // namespace oaa
