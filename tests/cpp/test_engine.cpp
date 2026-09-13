#include "oaa/engine.hpp"

#include <cassert>
#include <iostream>

int main() {
    oaa::Engine engine;

    assert(!engine.loaded());
    assert(engine.status() == "empty");

    const bool loaded = engine.load_model("phase1-placeholder");
    assert(loaded);
    assert(engine.loaded());
    assert(engine.status() == "ready");

    const auto result = engine.generate("hello");
    assert(result == "OAA_PHASE1_ECHO: hello");

    const auto stats = engine.get_stats();
    assert(stats.model_loaded);
    assert(stats.generation_calls == 1);
    assert(stats.model_path == "phase1-placeholder");

    engine.unload();
    assert(!engine.loaded());

    std::cout << "OAA C++ smoke test passed\n";
    return 0;
}
