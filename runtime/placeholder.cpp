// Built instead of main.cpp until recomp/ppc exists (no game recompiled yet).
// Lets the toolchain, the web shell and the COOP/COEP server be tested end to
// end before any game data is involved.

#include <cstdio>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

int main()
{
    std::printf("[ttt2] runtime built without recompiled game code.\n");
    std::printf("[ttt2] Run scripts/scan-xex.ps1 and scripts/recompile.ps1, then rebuild.\n");
#ifdef __EMSCRIPTEN__
    EM_ASM({
        if (typeof window !== 'undefined' && window.ttt2Status)
            window.ttt2Status('Runtime OK. Game code not recompiled yet.');
    });
#endif
    return 0;
}
