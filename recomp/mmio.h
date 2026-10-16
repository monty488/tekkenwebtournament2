#pragma once

// Force-included into every recompiled translation unit (see CMakeLists.txt), ahead of
// ppc_context.h, whose PPC_MM_* defaults are #ifndef-guarded.
//
// XenonRecomp emits PPC_MM_STORE_U32 for stores followed by eieio, i.e. MMIO writes. Most
// MMIO here is plain guest memory that host threads poll, but the XMA decoder's register
// file (0x7FEA0000) takes commands: each write to a kick/lock/clear register names contexts
// by bit, and a guest often writes the same register several times in a row (one bit per
// write). Those writes must reach the runtime one by one or all but the last are lost.

#include <cstdint>

// runtime/apu/xma.cpp. `value` is the guest register value (what PPC_STORE_U32 takes).
void ttt2_XmaRegisterWrite(uint32_t address, uint32_t value);

#define PPC_MM_STORE_U32(x, y)                                   \
    do                                                           \
    {                                                            \
        const uint32_t mmioAddress_ = (x);                       \
        const uint32_t mmioValue_ = (y);                         \
        if ((mmioAddress_ & 0xFFFF0000u) == 0x7FEA0000u)         \
            ttt2_XmaRegisterWrite(mmioAddress_, mmioValue_);     \
        else                                                     \
            PPC_STORE_U32(mmioAddress_, mmioValue_);             \
    } while (0)
