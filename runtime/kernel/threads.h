#pragma once

#include "guest.h"

namespace guest
{
    // Runs a guest function on the *current* host thread with a freshly
    // initialised guest thread environment (PCR, TLS, TEB, stack).
    // arg0/arg1 are passed in r3/r4. Returns the guest function's r3.
    uint32_t RunGuestThread(uint32_t startAddress, uint32_t arg0, uint32_t arg1 = 0);

    uint32_t CurrentThreadId();

    // Diagnostics (kernel/trace.cpp): guest threads register so the stall watchdog can
    // sample them.
    void RegisterGuestThread();
    void UnregisterGuestThread();
    void StartWatchdog();
    void InstallCrashHandler();

    // Host ticks per second as seen by the guest's mftb / KeQueryPerformanceCounter.
    uint64_t TimebaseFrequency();
}
