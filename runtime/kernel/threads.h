#pragma once

#include "guest.h"

#include <initializer_list>

namespace guest
{
    // Runs a guest function on the *current* host thread with a freshly
    // initialised guest thread environment (PCR, TLS, TEB, stack).
    // arg0/arg1 are passed in r3/r4. Returns the guest function's r3.
    uint32_t RunGuestThread(uint32_t startAddress, uint32_t arg0, uint32_t arg1 = 0);

    // A persistent guest environment (PCR, TLS, TEB, stack) for host threads that call
    // into guest code repeatedly: interrupt, DPC and command-processor threads.
    struct GuestEnv
    {
        uint32_t block = 0;
        uint32_t pcr = 0;
        uint32_t stackTop = 0;
    };
    GuestEnv CreateGuestEnv(uint8_t cpu = 0);
    void DestroyGuestEnv(GuestEnv& env);

    // Calls a guest function with up to eight integer arguments (r3..r10); returns r3.
    uint32_t CallGuest(const GuestEnv& env, uint32_t address, std::initializer_list<uint32_t> args);

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
