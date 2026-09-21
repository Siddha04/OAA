#include "oaa/memory.hpp"

namespace oaa {

MemoryBuffer::MemoryBuffer(std::size_t bytes) : storage_(bytes) {}

std::size_t MemoryBuffer::size() const noexcept {
    return storage_.size();
}

void MemoryBuffer::resize(std::size_t bytes) {
    storage_.resize(bytes);
}

std::byte* MemoryBuffer::data() noexcept {
    return storage_.data();
}

const std::byte* MemoryBuffer::data() const noexcept {
    return storage_.data();
}

MemoryBuffer MemoryManager::allocate(std::size_t bytes) const {
    return MemoryBuffer(bytes);
}

} // namespace oaa
