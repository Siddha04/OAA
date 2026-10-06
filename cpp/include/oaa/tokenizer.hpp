#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace oaa {

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    virtual std::vector<std::uint32_t> encode(const std::string& text) const = 0;
    virtual std::string decode(const std::vector<std::uint32_t>& tokens) const = 0;
    virtual std::size_t vocab_size() const noexcept = 0;
};

class ByteTokenizer final : public Tokenizer {
public:
    std::vector<std::uint32_t> encode(const std::string& text) const override;
    std::string decode(const std::vector<std::uint32_t>& tokens) const override;
    std::size_t vocab_size() const noexcept override;
};

class PrintableAsciiTokenizer final : public Tokenizer {
public:
    std::vector<std::uint32_t> encode(const std::string& text) const override;
    std::string decode(const std::vector<std::uint32_t>& tokens) const override;
    std::size_t vocab_size() const noexcept override;
};

std::unique_ptr<Tokenizer> create_byte_tokenizer();
std::unique_ptr<Tokenizer> create_printable_ascii_tokenizer();

} // namespace oaa
