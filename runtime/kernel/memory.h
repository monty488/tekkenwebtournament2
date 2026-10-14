#pragma once

#include "guest.h"

namespace guest
{
    // Reserves the 4 GB guest address space and sets g_base.
    bool InitMemory();

    // Allocations in the physical (GPU-visible, 0xA0000000) and virtual (0x40000000) windows.
    // Returned memory is zeroed. Return 0 on failure.
    uint32_t AllocPhysical(uint32_t size, uint32_t alignment = 0x10000);
    uint32_t AllocVirtual(uint32_t size, uint32_t alignment = 0x10000);
    void Free(uint32_t address);
}
