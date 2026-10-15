#pragma once

// Force-included into every translation unit (see the root CMakeLists.txt) so it is
// seen before XenonRecomp's ppc_context.h, which only defines PPC_CALL_INDIRECT_FUNC
// when it is not already defined.
//
// Indirect calls (function pointers, vtables, bctrl) go through a range- and
// null-checked lookup that reports bad targets instead of jumping to address 0.

#include <cstdint>

struct PPCContext;
void TTT2CallIndirect(PPCContext& ctx, uint8_t* base, uint32_t target);

#define PPC_CALL_INDIRECT_FUNC(x) TTT2CallIndirect(ctx, base, (x))
