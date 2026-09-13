#include "oaa/engine.hpp"

#include <stdexcept>

namespace oaa {

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

std::string Engine::generate(const std::string& prompt, const GenerationConfig&) {
    if (!model_loaded_) {
        throw std::runtime_error("engine has no loaded model");
    }

    ++generation_calls_;
    return "OAA_PHASE1_ECHO: " + prompt;
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
