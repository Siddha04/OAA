#include "oaa/engine.hpp"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

namespace {

std::string write_test_manifest() {
    const std::string path = "oaa_phase5_test.manifest";
    std::ofstream manifest(path);
    manifest
        << "architecture=tiny_transformer_v1\n"
        << "vocab_size=95\n"
        << "hidden_size=16\n"
        << "num_layers=2\n"
        << "num_heads=4\n"
        << "intermediate_size=32\n"
        << "context_length=256\n"
        << "seed=42\n";
    return path;
}

} // namespace

int main() {
    const auto manifest_path = write_test_manifest();

    oaa::Engine engine;

    assert(!engine.loaded());
    assert(engine.status() == "empty");

    assert(engine.load_model(manifest_path));
    assert(engine.loaded());
    assert(engine.status() == "ready");

    oaa::GenerationConfig config;
    config.max_tokens = 8;
    config.temperature = 0.0F;
    config.seed = 123;

    const auto first = engine.generate("hello", config);
    const auto stats_after_first = engine.get_stats();

    const auto second = engine.generate("hello", config);
    assert(first == second);
    assert(first.size() == 8);

    const auto chunks = engine.generate_stream("hello", config);
    std::string reconstructed;
    for (const auto& chunk : chunks) {
        reconstructed += chunk;
    }

    assert(reconstructed == first);

    const auto stats = engine.get_stats();
    assert(stats.model_loaded);
    assert(stats.generation_calls == 3);
    assert(stats.prompt_tokens == 15);
    assert(stats.generated_tokens == 24);
    assert(stats.model_path == manifest_path);
    assert(stats.last_generation_seconds >= 0.0);
    assert(stats_after_first.generated_tokens == 8);

    engine.unload();
    assert(!engine.loaded());

    std::remove(manifest_path.c_str());

    std::cout << "OAA Phase 5 engine tests passed" << std::endl;
    return 0;
}
