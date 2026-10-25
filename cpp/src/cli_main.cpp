#include "oaa/engine.hpp"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#ifndef OAA_VERSION
#define OAA_VERSION "unknown"
#endif

namespace {

struct Options {
    std::string model;
    std::string prompt;
    oaa::GenerationConfig generation;
    bool have_model{false};
    bool have_prompt{false};
};

void print_usage(std::ostream& stream) {
    stream
        << "OAA native CLI " << OAA_VERSION << "\n"
        << "Usage: oaa_native_cli --model <manifest> --prompt <text> [options]\n\n"
        << "Required:\n"
        << "  --model <path>              Local tiny_transformer_v1 manifest\n"
        << "  --prompt <text>             Text prompt (quote multi-word prompts)\n\n"
        << "Options:\n"
        << "  --max-tokens <n>             Generated token limit (default 128)\n"
        << "  --temperature <number>       Sampling temperature (default 0.7)\n"
        << "  --top-k <n>                  Top-k sampling (default 0 = disabled)\n"
        << "  --top-p <number>             Nucleus sampling threshold (default 1.0)\n"
        << "  --repetition-penalty <n>     Repetition penalty (default 1.0)\n"
        << "  --seed <n>                   Sampling seed (default 0)\n"
        << "  --help                       Show this help\n"
        << "  --version                    Show version\n";
}

std::string value_after(int& index, int argc, char* argv[], const std::string& flag) {
    if (index + 1 >= argc) {
        throw std::invalid_argument("missing value for " + flag);
    }
    ++index;
    return argv[index];
}

std::uint64_t parse_unsigned(const std::string& raw, const std::string& name) {
    if (raw.empty() || raw.front() == '-') {
        throw std::invalid_argument(name + " must be a non-negative integer");
    }
    std::size_t consumed = 0;
    const auto parsed = std::stoull(raw, &consumed, 10);
    if (consumed != raw.size()) {
        throw std::invalid_argument("invalid " + name);
    }
    return parsed;
}

float parse_float(const std::string& raw, const std::string& name) {
    std::size_t consumed = 0;
    const auto parsed = std::stof(raw, &consumed);
    if (consumed != raw.size()) {
        throw std::invalid_argument("invalid " + name);
    }
    return parsed;
}

Options parse_options(int argc, char* argv[]) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string flag = argv[index];
        if (flag == "--help" || flag == "-h") {
            print_usage(std::cout);
            std::exit(0);
        } else if (flag == "--version") {
            std::cout << OAA_VERSION << '\n';
            std::exit(0);
        } else if (flag == "--model") {
            options.model = value_after(index, argc, argv, flag);
            options.have_model = true;
        } else if (flag == "--prompt") {
            options.prompt = value_after(index, argc, argv, flag);
            options.have_prompt = true;
        } else if (flag == "--max-tokens") {
            const auto parsed = parse_unsigned(value_after(index, argc, argv, flag), "max-tokens");
            if (parsed == 0 || parsed > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("max-tokens must be between 1 and 4294967295");
            }
            options.generation.max_tokens = static_cast<std::uint32_t>(parsed);
        } else if (flag == "--top-k") {
            const auto parsed = parse_unsigned(value_after(index, argc, argv, flag), "top-k");
            if (parsed > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("top-k is too large");
            }
            options.generation.top_k = static_cast<std::uint32_t>(parsed);
        } else if (flag == "--temperature") {
            options.generation.temperature =
                parse_float(value_after(index, argc, argv, flag), "temperature");
        } else if (flag == "--top-p") {
            options.generation.top_p =
                parse_float(value_after(index, argc, argv, flag), "top-p");
        } else if (flag == "--repetition-penalty") {
            options.generation.repetition_penalty =
                parse_float(value_after(index, argc, argv, flag), "repetition-penalty");
        } else if (flag == "--seed") {
            options.generation.seed =
                parse_unsigned(value_after(index, argc, argv, flag), "seed");
        } else {
            throw std::invalid_argument("unknown option: " + flag);
        }
    }
    if (!options.have_model || options.model.empty()) {
        throw std::invalid_argument("--model is required");
    }
    if (!options.have_prompt || options.prompt.empty()) {
        throw std::invalid_argument("--prompt is required and must not be empty");
    }
    return options;
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        const auto options = parse_options(argc, argv);
        oaa::Engine engine;
        engine.load_model(options.model);
        const auto output = engine.generate(options.prompt, options.generation);
        std::cout << output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "oaa_native_cli: " << error.what() << '\n';
        std::cerr << "Run 'oaa_native_cli --help' for usage.\n";
        return 2;
    }
}
