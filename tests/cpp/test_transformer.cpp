#include "oaa/transformer.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    oaa::KvCache cache(2);
    cache.append(0, {1.0F, 2.0F}, {3.0F, 4.0F});
    cache.append(1, {5.0F, 6.0F}, {7.0F, 8.0F});

    assert(cache.size() == 1);
    assert(cache.num_layers() == 2);
    assert(cache.keys(0).size() == 2);
    assert(cache.values(1).at(1) == 8.0F);

    cache.append(0, {9.0F, 10.0F}, {11.0F, 12.0F});
    cache.append(1, {13.0F, 14.0F}, {15.0F, 16.0F});
    assert(cache.size() == 2);
    assert(cache.keys(0).size() == 4);

    cache.reset();
    assert(cache.size() == 0);
    assert(cache.keys(0).empty());

    oaa::ModelConfig config;
    config.architecture = "tiny_transformer_v1";
    config.vocab_size = 95;
    config.hidden_size = 16;
    config.num_layers = 2;
    config.num_heads = 4;
    config.intermediate_size = 32;
    config.context_length = 64;
    config.seed = 7;

    oaa::TransformerModel model(config);

    const auto logits1 = model.forward_token(1);
    assert(logits1.size() == config.vocab_size);

    model.reset_cache();

    const auto logits2 = model.forward_token(1);
    assert(logits2.size() == logits1.size());

    for (std::size_t i = 0; i < logits1.size(); ++i) {
        assert(std::isfinite(logits1[i]));
        assert(std::isfinite(logits2[i]));
        assert(std::fabs(logits1[i] - logits2[i]) < 1e-6F);
    }

    model.reset_cache();
    const auto generated = model.generate_tokens(
        {1, 2, 3, 4},
        oaa::GenerationConfig{
            6, 0.0F, 0, 1.0F, 1.0F, 99});

    assert(generated.size() == 6);
    for (const auto token : generated) {
        assert(token < config.vocab_size);
    }

    std::cout << "OAA Phase 5 transformer tests passed" << std::endl;
    return 0;
}
