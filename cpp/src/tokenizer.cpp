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

std::unique_ptr<Tokenizer> create_byte_tokenizer() {
    return std::make_unique<ByteTokenizer>();
}

} // namespace oaa
