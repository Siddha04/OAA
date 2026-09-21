#pragma once

#include <cstddef>
#include <vector>

namespace oaa {

class MemoryBuffer {
public:
    MemoryBuffer() = default;
    explicit MemoryBuffer(std::size_t bytes);

    std::size_t size() const noexcept;
    void resize(std::size_t bytes);

    std::byte* data() noexcept;
    const std::byte* data() const noexcept;

private:
    std::vector<std::byte> storage_;
};

class MemoryManager {
public:
    MemoryBuffer allocate(std::size_t bytes) const;
};

} // namespace oaa
