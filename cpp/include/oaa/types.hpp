#pragma once

#include <cstdint>
#include <string>

namespace oaa {

struct RuntimeStats {
    bool model_loaded{false};
    std::uint64_t generation_calls{0};
    std::string model_path;
};

struct GenerationConfig {
    std::uint32_t max_tokens{128};
    float temperature{0.7F};
};

} // namespace oaa
