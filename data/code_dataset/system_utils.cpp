#include <string>
#include <vector>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace sys {

// --- 1. Fast Linear Arena Allocator ---
class ArenaAllocator {
private:
    char* memory;
    size_t total_size;
    size_t offset;

public:
    explicit ArenaAllocator(size_t size) : total_size(size), offset(0) {
        memory = new char[total_size];
    }

    void* allocate(size_t bytes, size_t alignment = alignof(std::max_align_t)) {
        size_t current_ptr = reinterpret_cast<size_t>(memory + offset);
        size_t padding = (alignment - (current_ptr % alignment)) % alignment;

        if (offset + padding + bytes > total_size) {
            throw std::bad_alloc();
        }

        offset += padding;
        void* ptr = memory + offset;
        offset += bytes;
        return ptr;
    }

    void reset() {
        offset = 0;
    }

    ~ArenaAllocator() {
        delete[] memory;
    }
};

// --- 2. String Tokenizer & Splitter ---
std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream token_stream(str);
    while (std::getline(token_stream, token, delimiter)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

// --- 3. Fast Bitwise Manipulation ---
inline uint32_t count_set_bits(uint32_t n) {
    #if defined(__GNUC__) || defined(__clang__)
    return static_cast<uint32_t>(__builtin_popcount(n));
    #else
    uint32_t count = 0;
    while (n > 0) {
        n &= (n - 1);
        count++;
    }
    return count;
    #endif
}

inline uint32_t next_power_of_two(uint32_t n) {
    if (n == 0) return 1;
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

} // namespace sys
