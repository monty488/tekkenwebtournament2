#pragma once

// Core glue between recompiled PPC code and the host runtime.
//
// Every Xbox 360 kernel/XAM import that default.xex links against becomes a
// call to a PPC function named __imp__<ExportName> (see XenonUtils/xex.cpp).
// The runtime provides those functions with GUEST_IMPORT below; anything not
// implemented gets an auto-generated logging stub (scripts/gen_import_stubs.py).
//
// Calling convention: arguments arrive in r3..r10 (f1.. for floats),
// the return value goes in r3. Guest pointers are 32-bit, big-endian in memory.

#include <cstdint>
#include <cstdio>
#include <atomic>

#include <ppc_config.h>
#include <ppc_context.h>
#include <xbox.h>

namespace guest
{
    // Host pointer to guest address 0. Valid for the whole 4 GB guest space.
    extern uint8_t* g_base;

    inline void* ToHost(uint32_t address)
    {
        return address ? g_base + address : nullptr;
    }

    template<typename T>
    inline T* ToHost(uint32_t address)
    {
        return reinterpret_cast<T*>(ToHost(address));
    }

    inline uint32_t ToGuest(const void* host)
    {
        return host ? uint32_t(reinterpret_cast<const uint8_t*>(host) - g_base) : 0;
    }

    // Reads/writes big-endian scalars at a guest address.
    template<typename T> inline T Load(uint32_t address) { return *ToHost<be<T>>(address); }
    template<typename T> inline void Store(uint32_t address, T value) { *ToHost<be<T>>(address) = value; }

    // Calls a guest function by address (through the recompiled function table).
    inline void Call(PPCContext& ctx, uint32_t address)
    {
        uint8_t* base = g_base; // PPC_CALL_INDIRECT_FUNC expects `ctx` and `base` in scope
        PPC_CALL_INDIRECT_FUNC(address);
    }

    void LogUnimplemented(const char* name);
}

// Logging. Kept printf-style so it works identically on native and Emscripten.
#define TTT2_LOG(fmt, ...) std::fprintf(stderr, "[ttt2] " fmt "\n", ##__VA_ARGS__)

namespace guest
{
    // Records the call for the stall watchdog and, with TTT2_TRACE=1, logs it.
    void TraceImport(const char* name, const PPCContext& ctx);
}

// Defines the host implementation of a kernel/XAM export.
//   GUEST_IMPORT(NtClose) { ... ctx.r3.u64 = STATUS_SUCCESS; }
#define GUEST_IMPORT(name)                                              \
    static void name##_Impl(PPCContext& __restrict ctx, uint8_t* base); \
    PPC_FUNC(__imp__##name)                                             \
    {                                                                   \
        guest::TraceImport(#name, ctx);                                 \
        name##_Impl(ctx, base);                                         \
    }                                                                   \
    static void name##_Impl([[maybe_unused]] PPCContext& __restrict ctx, [[maybe_unused]] uint8_t* base)

// Placeholder used by the generated stub file: logs once, returns 0.
#define GUEST_IMPORT_STUB(name)                         \
    GUEST_IMPORT(name)                                  \
    {                                                   \
        static std::atomic_flag logged = ATOMIC_FLAG_INIT; \
        if (!logged.test_and_set())                     \
            guest::LogUnimplemented(#name);             \
        ctx.r3.u64 = 0;                                 \
    }

// Common NTSTATUS / Win32 codes. XenonUtils/xbox.h pulls in <windows.h> on
// Windows, which already defines some of these as macros (same values), so
// only define what is missing.
#ifndef STATUS_SUCCESS
#define STATUS_SUCCESS               0x00000000u
#endif
#ifndef STATUS_PENDING
#define STATUS_PENDING               0x00000103u
#endif
#ifndef STATUS_NOT_IMPLEMENTED
#define STATUS_NOT_IMPLEMENTED       0xC0000002u
#endif
#ifndef STATUS_INVALID_HANDLE
#define STATUS_INVALID_HANDLE        0xC0000008u
#endif
#ifndef STATUS_INVALID_PARAMETER
#define STATUS_INVALID_PARAMETER     0xC000000Du
#endif
#ifndef STATUS_END_OF_FILE
#define STATUS_END_OF_FILE           0xC0000011u
#endif
#ifndef STATUS_NO_MEMORY
#define STATUS_NO_MEMORY             0xC0000017u
#endif
#ifndef STATUS_OBJECT_NAME_NOT_FOUND
#define STATUS_OBJECT_NAME_NOT_FOUND 0xC0000034u
#endif
#ifndef STATUS_OBJECT_PATH_NOT_FOUND
#define STATUS_OBJECT_PATH_NOT_FOUND 0xC000003Au
#endif

#ifndef ERROR_SUCCESS
#define ERROR_SUCCESS                0u
#endif
#ifndef ERROR_DEVICE_NOT_CONNECTED
#define ERROR_DEVICE_NOT_CONNECTED   0x48Fu
#endif
