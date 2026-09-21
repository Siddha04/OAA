#pragma once

#include <cstddef>
#include <string>

namespace oaa {

struct ModelConfig {
    std::size_t vocab_size{0};
    std::size_t hidden_size{0};
    std::size_t num_layers{0};
    std::size_t num_heads{0};
};

class ModelLoader {
public:
    virtual ~ModelLoader() = default;

    virtual ModelConfig load(const std::string& path) = 0;
    virtual void unload() = 0;
    virtual bool loaded() const noexcept = 0;
};

class ManifestModelLoader final : public ModelLoader {
public:
    ModelConfig load(const std::string& path) override;
    void unload() override;
    bool loaded() const noexcept override;

private:
    ModelConfig config_{};
    bool loaded_{false};
};

} // namespace oaa
