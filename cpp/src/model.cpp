#include "oaa/model.hpp"

#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace oaa {

namespace {

std::size_t parse_positive(const std::string& key, const std::string& value) {
    try {
        const auto parsed = std::stoull(value);
        if (parsed == 0 || parsed > std::numeric_limits<std::size_t>::max()) {
            throw std::invalid_argument(key + " must be positive");
        }
        return static_cast<std::size_t>(parsed);
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid " + key + " in model manifest");
    }
}

std::uint64_t parse_seed(const std::string& value) {
    try {
        return static_cast<std::uint64_t>(std::stoull(value));
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid seed in model manifest");
    }
}

} // namespace

ModelConfig ManifestModelLoader::load(const std::string& path) {
    if (path.empty()) {
        throw std::invalid_argument("model manifest path cannot be empty");
    }

    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("unable to open model manifest: " + path);
    }

    ModelConfig candidate{};
    std::string line;

    while (std::getline(input, line)) {
        const auto separator = line.find('=');

        if (separator == std::string::npos) {
            if (!line.empty()) {
                throw std::invalid_argument("invalid model manifest line");
            }
            continue;
        }

        const std::string key = line.substr(0, separator);
        const std::string value = line.substr(separator + 1);

        if (key == "architecture") {
            candidate.architecture = value;
        } else if (key == "vocab_size") {
            candidate.vocab_size = parse_positive(key, value);
        } else if (key == "hidden_size") {
            candidate.hidden_size = parse_positive(key, value);
        } else if (key == "num_layers") {
            candidate.num_layers = parse_positive(key, value);
        } else if (key == "num_heads") {
            candidate.num_heads = parse_positive(key, value);
        } else if (key == "intermediate_size") {
            candidate.intermediate_size = parse_positive(key, value);
        } else if (key == "context_length") {
            candidate.context_length = parse_positive(key, value);
        } else if (key == "seed") {
            candidate.seed = parse_seed(value);
        }
    }

    if (candidate.intermediate_size == 0 && candidate.hidden_size != 0) {
        candidate.intermediate_size = candidate.hidden_size * 4;
    }

    if (candidate.context_length == 0 && candidate.hidden_size != 0) {
        candidate.context_length = 256;
    }

    if (candidate.vocab_size == 0 || candidate.hidden_size == 0 ||
        candidate.num_layers == 0 || candidate.num_heads == 0) {
        throw std::invalid_argument("model manifest is missing required fields");
    }

    if (candidate.hidden_size % candidate.num_heads != 0) {
        throw std::invalid_argument("hidden_size must be divisible by num_heads");
    }

    config_ = candidate;
    loaded_ = true;
    return config_;
}

void ManifestModelLoader::unload() {
    config_ = {};
    loaded_ = false;
}

bool ManifestModelLoader::loaded() const noexcept {
    return loaded_;
}

} // namespace oaa
