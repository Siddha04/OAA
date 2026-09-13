#pragma once

#include "oaa/types.hpp"
#include <string>

namespace oaa {

class Engine {
public:
    Engine() = default;
    bool load_model(const std::string& path);
    void unload();
    std::string generate(const std::string& prompt, const GenerationConfig& config = {});
    RuntimeStats get_stats() const;
    bool loaded() const;
    std::string status() const;

private:
    bool model_loaded_{false};
    std::string model_path_;
    std::uint64_t generation_calls_{0};
};

} // namespace oaa
