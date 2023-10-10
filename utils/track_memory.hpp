#ifndef TRACK_MEMORY_HPP
#define TRACK_MEMORY_HPP

#include <iostream>
#include <memory>

struct trackMemory {
    uint32_t totalMemoryAllocated = 0;
    uint32_t totalMemoryFreed = 0;
    uint32_t currentMemoryUsage() {
        return totalMemoryAllocated - totalMemoryFreed;
    }
};

extern trackMemory s_trackMemory;

void* operator new (size_t size);
void operator delete (void* memory, size_t size);
void operator delete (void* memory) noexcept;

void printMemoryUsage();

#endif // TRACK_MEMORY_HPP
