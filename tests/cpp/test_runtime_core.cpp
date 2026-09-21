#include "oaa/inference.hpp"
#include "oaa/memory.hpp"
#include "oaa/model.hpp"
#include "oaa/tensor.hpp"
#include "oaa/tokenizer.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main() {
    {
        oaa::Tensor lhs({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
        oaa::Tensor rhs({3, 2}, {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});
        const auto result = oaa::Tensor::matmul_2d(lhs, rhs);

        assert(result.shape() == std::vector<std::size_t>({2, 2}));
        assert(std::fabs(result.at({0, 0}) - 58.0F) < 1e-5F);
        assert(std::fabs(result.at({0, 1}) - 64.0F) < 1e-5F);
        assert(std::fabs(result.at({1, 0}) - 139.0F) < 1e-5F);
        assert(std::fabs(result.at({1, 1}) - 154.0F) < 1e-5F);

        const auto probabilities = oaa::Tensor::softmax_last_dim(
            oaa::Tensor({1, 3}, {1.0F, 2.0F, 3.0F}));

        float sum = 0.0F;
        for (std::size_t i = 0; i < 3; ++i) {
            sum += probabilities.at({0, i});
        }

        assert(std::fabs(sum - 1.0F) < 1e-5F);
    }

    {
        const oaa::MemoryManager manager;
        auto buffer = manager.allocate(4096);
        assert(buffer.size() == 4096);
        assert(buffer.data() != nullptr);
    }

    {
        auto tokenizer = oaa::create_byte_tokenizer();
        const std::string text = "OAA runtime";
        const auto tokens = tokenizer->encode(text);
        assert(tokens.size() == text.size());
        assert(tokenizer->decode(tokens) == text);
    }

    {
        const std::string manifest_path = "oaa_phase2_test.model";

        {
            std::ofstream manifest(manifest_path);
            manifest << "vocab_size=32000
";
            manifest << "hidden_size=4096
";
            manifest << "num_layers=32
";
            manifest << "num_heads=32
";
        }

        oaa::ManifestModelLoader loader;
        const auto config = loader.load(manifest_path);

        assert(loader.loaded());
        assert(config.vocab_size == 32000);
        assert(config.hidden_size == 4096);
        assert(config.num_layers == 32);
        assert(config.num_heads == 32);

        loader.unload();
        assert(!loader.loaded());

        std::remove(manifest_path.c_str());
    }

    {
        const oaa::CpuInferenceEngine runtime;
        const oaa::Tensor input({1, 2}, {2.0F, 3.0F});
        const oaa::Tensor weights({2, 2}, {1.0F, 0.0F, 0.0F, 1.0F});
        const oaa::Tensor bias({2}, {0.5F, -0.5F});

        const auto output = runtime.linear(input, weights, &bias);
        assert(std::fabs(output.at({0, 0}) - 2.5F) < 1e-5F);
        assert(std::fabs(output.at({0, 1}) - 2.5F) < 1e-5F);
    }

    std::cout << "OAA Phase 2 C++ runtime core tests passed
";
    return 0;
}
