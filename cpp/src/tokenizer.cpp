#include "oaa/tokenizer.hpp"

#include <stdexcept>

namespace oaa {

std::vector<std::uint32_t> ByteTokenizer::encode(const std::string& text) const {
    std::vector<std::uint32_t> tokens;
    tokens.reserve(text.size());
    for (const unsigned char byte : text) {
        tokens.push_back(static_cast<std::uint32_t>(byte));
    }
    return tokens;
}

std::string ByteTokenizer::decode(const std::vector<std::uint32_t>& tokens) const {
    std::string result;
    result.reserve(tokens.size());
    for (const auto token : tokens) {
        if (token > 255U) {
            throw std::invalid_argument("byte tokenizer token must be in [0, 255]");
        }
        result.push_back(static_cast<char>(token));
    }
    return result;
}

std::size_t ByteTokenizer::vocab_size() const noexcept {
    return 256;
}

std::vector<std::uint32_t> PrintableAsciiTokenizer::encode(
    const std::string& text) const {

    std::vector<std::uint32_t> tokens;
    tokens.reserve(text.size());

    for (const unsigned char byte : text) {
        if (byte < 32U || byte > 126U) {
            tokens.push_back(0U);
        } else {
            tokens.push_back(static_cast<std::uint32_t>(byte - 32U));
        }
    }

    return tokens;
}

std::string PrintableAsciiTokenizer::decode(
    const std::vector<std::uint32_t>& tokens) const {

    std::string result;
    result.reserve(tokens.size());

    for (const auto token : tokens) {
        if (token >= vocab_size()) {
            throw std::invalid_argument(
                "printable ASCII tokenizer token is out of vocabulary");
        }
        result.push_back(static_cast<char>(token + 32U));
    }

    return result;
}

std::size_t PrintableAsciiTokenizer::vocab_size() const noexcept {
    return 95;
}

std::unique_ptr<Tokenizer> create_byte_tokenizer() {
    return std::make_unique<ByteTokenizer>();
}

std::unique_ptr<Tokenizer> create_printable_ascii_tokenizer() {
    return std::make_unique<PrintableAsciiTokenizer>();
}

} // namespace oaa
