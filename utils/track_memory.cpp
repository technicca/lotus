#include <iostream>
#include <memory>
#include "track_memory.hpp"

trackMemory s_trackMemory;

void* operator new (size_t size) {
    s_trackMemory.totalMemoryAllocated += size;
    return malloc(size);
}

void operator delete (void* memory, size_t size) {
    s_trackMemory.totalMemoryFreed += size;
    free (memory);
}

void operator delete (void* memory) noexcept {
    free(memory);
}

void printMemoryUsage() {
    std::cout << "Total memory usage: " << s_trackMemory.currentMemoryUsage() << " bytes\n";
}