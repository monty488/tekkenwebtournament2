// Guest threads, TLS, critical sections and time.

#include "threads.h"
#include "memory.h"
#include "objects.h"
#include "module.h"
#include "sync.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

namespace guest
{
    // Per-thread guest environment. Layout matches what Xbox 360 titles read via
    // r13 (the processor control region); sizes follow Unleashed Recompiled/Xenia.
    constexpr uint32_t PCR_SIZE   = 0xAB0;
    constexpr uint32_t TLS_SIZE   = 0x100;
    constexpr uint32_t TEB_SIZE   = 0x2E0;
    constexpr uint32_t STACK_SIZE = 0x40000;

    static std::atomic<uint32_t> g_nextThreadId{ 1 };
    static thread_local uint32_t t_threadId = 0;

    uint32_t CurrentThreadId()
    {
        if (!t_threadId)
            t_threadId = g_nextThreadId.fetch_add(1);
        return t_threadId;
    }

    GuestEnv CreateGuestEnv(uint8_t cpu)
    {
        // Static TLS comes from the XEX's TLS directory; keep at least TLS_SIZE so titles
        // without one still get a valid block.
        const TlsInfo& tlsInfo = GetTlsInfo();
        const uint32_t tlsSize = (std::max(tlsInfo.dataSize, TLS_SIZE) + 0xF) & ~0xFu;

        GuestEnv env;
        env.block = AllocVirtual(PCR_SIZE + tlsSize + TEB_SIZE + STACK_SIZE);
        if (!env.block)
        {
            TTT2_LOG("CreateGuestEnv: could not allocate thread block");
            return env;
        }

        env.pcr = env.block;
        const uint32_t tls = env.pcr + PCR_SIZE;
        const uint32_t teb = tls + tlsSize;
        env.stackTop = teb + TEB_SIZE + STACK_SIZE;

        if (tlsInfo.rawAddress && tlsInfo.rawSize)
            std::memcpy(ToHost(tls), ToHost(tlsInfo.rawAddress), std::min(tlsInfo.rawSize, tlsSize));

        Store<uint32_t>(env.pcr + 0x000, tls);                // PCR->TlsData
        Store<uint32_t>(env.pcr + 0x100, teb);                // PCR->CurrentThread
        *ToHost<uint8_t>(env.pcr + 0x10C) = cpu;              // PCR->CurrentCpu
        Store<uint32_t>(teb + 0x14C, CurrentThreadId());      // KTHREAD->ThreadId

        RegisterGuestThread();
        return env;
    }

    void DestroyGuestEnv(GuestEnv& env)
    {
        UnregisterGuestThread();
        if (env.block)
            Free(env.block);
        env = {};
    }

    uint32_t CallGuest(const GuestEnv& env, uint32_t address, std::initializer_list<uint32_t> args)
    {
        PPCContext ctx{};
        ctx.r1.u64 = env.stackTop;
        ctx.r13.u64 = env.pcr;
        PPCRegister* argRegs[] = { &ctx.r3, &ctx.r4, &ctx.r5, &ctx.r6, &ctx.r7, &ctx.r8, &ctx.r9, &ctx.r10 };
        size_t i = 0;
        for (uint32_t arg : args)
            if (i < std::size(argRegs))
                argRegs[i++]->u64 = arg;
        ctx.fpscr.loadFromHost();
        Call(ctx, address);
        return ctx.r3.u32;
    }

    // Thrown by ExTerminateThread and caught at the thread's entry, unwinding the
    // recompiled frames in between (they hold no resources).
    struct GuestThreadExit
    {
        uint32_t exitCode;
    };

    uint32_t RunGuestThread(uint32_t startAddress, uint32_t arg0, uint32_t arg1)
    {
        GuestEnv env = CreateGuestEnv();
        if (!env.block)
            return 0;
        uint32_t result;
        try
        {
            result = CallGuest(env, startAddress, { arg0, arg1 });
        }
        catch (const GuestThreadExit& exit)
        {
            result = exit.exitCode;
        }
        DestroyGuestEnv(env);
        return result;
    }

    uint64_t TimebaseFrequency()
    {
#if defined(__wasm__)
        return 49875000; // must match __rdtsc() in patches/XenonRecomp-0001-wasm-support.patch
#else
        // Native x86: mftb is __rdtsc(), so report the measured TSC rate.
        static const uint64_t frequency = []
        {
            using clock = std::chrono::steady_clock;
            const auto t0 = clock::now();
            const uint64_t c0 = __rdtsc();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            const uint64_t c1 = __rdtsc();
            const double seconds = std::chrono::duration<double>(clock::now() - t0).count();
            return uint64_t((c1 - c0) / seconds);
        }();
        return frequency;
#endif
    }

}

using namespace guest;

// NTSTATUS ExCreateThread(u32* handle, u32 stackSize, u32* threadId, u32 xapiStartup,
//                         u32 startAddress, u32 parameter, u32 creationFlags)
GUEST_IMPORT(ExCreateThread)
{
    constexpr uint32_t X_CREATE_SUSPENDED = 0x1;

    const uint32_t pHandle = ctx.r3.u32;
    const uint32_t pThreadId = ctx.r5.u32;
    const uint32_t xapiStartup = ctx.r6.u32;
    const uint32_t startAddress = ctx.r7.u32;
    const uint32_t parameter = ctx.r8.u32;
    const uint32_t flags = ctx.r9.u32;

    auto thread = std::make_shared<GuestThread>();
    thread->id = g_nextThreadId.fetch_add(1);
    thread->suspendCount = (flags & X_CREATE_SUSPENDED) ? 1 : 0;

    std::thread([thread, xapiStartup, startAddress, parameter]
    {
        t_threadId = thread->id;
        SetCurrentGuestThread(thread);
        {
            std::unique_lock lock(WaitMutex());
            WaitCondition().wait(lock, [&] { return thread->suspendCount == 0; });
        }

        // XapiThreadStartup(startAddress, parameter) wraps the real entry point when present.
        const uint32_t exitCode = xapiStartup
            ? RunGuestThread(xapiStartup, startAddress, parameter)
            : RunGuestThread(startAddress, parameter);
        {
            std::lock_guard lock(WaitMutex());
            thread->exitCode = exitCode;
            thread->finished = true;
            DeliverSignals(*thread);
        }
        NotifyWaiters();
    }).detach();

    if (pHandle)
        Store<uint32_t>(pHandle, Handles().Insert(thread));
    if (pThreadId)
        Store<uint32_t>(pThreadId, thread->id);
    ctx.r3.u64 = STATUS_SUCCESS;
}

// NTSTATUS NtResumeThread(u32 handle, u32* previousSuspendCount)
GUEST_IMPORT(NtResumeThread)
{
    auto thread = std::dynamic_pointer_cast<GuestThread>(WaitableFromHandle(ctx.r3.u32));
    if (!thread)
    {
        ctx.r3.u64 = STATUS_INVALID_HANDLE;
        return;
    }

    int previous;
    {
        std::lock_guard lock(WaitMutex());
        previous = thread->suspendCount;
        if (thread->suspendCount > 0)
            --thread->suspendCount;
    }
    NotifyWaiters();

    if (ctx.r4.u32)
        Store<uint32_t>(ctx.r4.u32, uint32_t(previous));
    ctx.r3.u64 = STATUS_SUCCESS;
}

// NTSTATUS NtSuspendThread(u32 handle, u32* previousSuspendCount)
// A thread suspending itself (the common idle pattern: "queue empty -> suspend until a
// producer resumes me") blocks here until NtResumeThread. Suspending another thread only
// records the count: host threads cannot be stopped at an arbitrary guest instruction.
GUEST_IMPORT(NtSuspendThread)
{
    auto thread = std::dynamic_pointer_cast<GuestThread>(WaitableFromHandle(ctx.r3.u32));
    if (!thread)
    {
        ctx.r3.u64 = STATUS_INVALID_HANDLE;
        return;
    }

    int previous;
    {
        std::unique_lock lock(WaitMutex());
        previous = thread->suspendCount++;
        if (ctx.r4.u32)
            Store<uint32_t>(ctx.r4.u32, uint32_t(previous));
        static const bool noSuspend = [] { const char* v = std::getenv("TTT2_NO_SELF_SUSPEND"); return v && *v == '1'; }();
        if (thread->id == CurrentThreadId() && !noSuspend)
            WaitCondition().wait(lock, [&] { return thread->suspendCount == 0; });
        else if (noSuspend)
            --thread->suspendCount;
    }
    ctx.r3.u64 = STATUS_SUCCESS;
}

// void ExTerminateThread(u32 exitCode): never returns.
GUEST_IMPORT(ExTerminateThread)
{
    throw GuestThreadExit{ ctx.r3.u32 };
}

GUEST_IMPORT(KeGetCurrentProcessType)
{
    ctx.r3.u64 = 1; // X_PROCTYPE_USER
}

// ---- TLS -------------------------------------------------------------------

static constexpr uint32_t kTlsSlots = 64;
static std::atomic<uint64_t> g_tlsUsed{ 0 };
static thread_local uint32_t t_tlsValues[kTlsSlots];

GUEST_IMPORT(KeTlsAlloc)
{
    uint64_t used = g_tlsUsed.load();
    for (;;)
    {
        if (used == ~0ull)
        {
            ctx.r3.u64 = 0xFFFFFFFF; // TLS_OUT_OF_INDEXES
            return;
        }
        const uint32_t index = uint32_t(std::countr_one(used));
        if (g_tlsUsed.compare_exchange_weak(used, used | (1ull << index)))
        {
            ctx.r3.u64 = index;
            return;
        }
    }
}

GUEST_IMPORT(KeTlsFree)
{
    if (ctx.r3.u32 < kTlsSlots)
        g_tlsUsed.fetch_and(~(1ull << ctx.r3.u32));
    ctx.r3.u64 = 1;
}

GUEST_IMPORT(KeTlsGetValue)
{
    ctx.r3.u64 = ctx.r3.u32 < kTlsSlots ? t_tlsValues[ctx.r3.u32] : 0;
}

GUEST_IMPORT(KeTlsSetValue)
{
    const bool ok = ctx.r3.u32 < kTlsSlots;
    if (ok)
        t_tlsValues[ctx.r3.u32] = ctx.r4.u32;
    ctx.r3.u64 = ok;
}

// ---- Critical sections -----------------------------------------------------
// XRTL_CRITICAL_SECTION's lock fields are only touched by the kernel, so they
// are kept in host byte order (see XenonUtils/xbox.h).

static void InitCriticalSection(uint32_t address)
{
    auto* cs = ToHost<XRTL_CRITICAL_SECTION>(address);
    cs->LockCount = -1;
    cs->RecursionCount = 0;
    cs->OwningThread = 0;
}

GUEST_IMPORT(RtlInitializeCriticalSection)
{
    InitCriticalSection(ctx.r3.u32);
}

GUEST_IMPORT(RtlInitializeCriticalSectionAndSpinCount)
{
    InitCriticalSection(ctx.r3.u32);
    ctx.r3.u64 = STATUS_SUCCESS;
}

static bool TryEnter(XRTL_CRITICAL_SECTION* cs, uint32_t self)
{
    std::atomic_ref<uint32_t> owner(cs->OwningThread);
    uint32_t expected = 0;
    if (owner.compare_exchange_strong(expected, self, std::memory_order_acquire))
    {
        cs->RecursionCount = 1;
        return true;
    }
    if (expected == self)
    {
        ++cs->RecursionCount;
        return true;
    }
    return false;
}

GUEST_IMPORT(RtlEnterCriticalSection)
{
    auto* cs = ToHost<XRTL_CRITICAL_SECTION>(ctx.r3.u32);
    const uint32_t self = CurrentThreadId();
    const auto start = std::chrono::steady_clock::now();
    bool reported = false;
    for (uint32_t spins = 0; !TryEnter(cs, self); ++spins)
    {
        if (spins > 64)
            std::this_thread::yield();
        if (!reported && (spins & 0xFFF) == 0 && std::chrono::steady_clock::now() - start > std::chrono::seconds(2))
        {
            reported = true;
            TTT2_LOG("T%u waiting >2s for critical section %08X owned by T%u (recursion %d)", self,
                ctx.r3.u32, std::atomic_ref<uint32_t>(cs->OwningThread).load(), cs->RecursionCount);
        }
    }
}

GUEST_IMPORT(RtlTryEnterCriticalSection)
{
    ctx.r3.u64 = TryEnter(ToHost<XRTL_CRITICAL_SECTION>(ctx.r3.u32), CurrentThreadId());
}

GUEST_IMPORT(RtlLeaveCriticalSection)
{
    auto* cs = ToHost<XRTL_CRITICAL_SECTION>(ctx.r3.u32);
    if (--cs->RecursionCount == 0)
        std::atomic_ref<uint32_t>(cs->OwningThread).store(0, std::memory_order_release);
}

// ---- Time ------------------------------------------------------------------

GUEST_IMPORT(KeQueryPerformanceFrequency)
{
    ctx.r3.u64 = TimebaseFrequency();
}

// NTSTATUS KeDelayExecutionThread(u32 mode, u32 alertable, s64* interval)
// Interval is in 100 ns units; negative means relative.
GUEST_IMPORT(KeDelayExecutionThread)
{
    const int64_t interval = Load<int64_t>(ctx.r5.u32);
    if (interval < 0)
        std::this_thread::sleep_for(std::chrono::nanoseconds(-interval * 100));
    else
        std::this_thread::yield();
    ctx.r3.u64 = STATUS_SUCCESS;
}

GUEST_IMPORT(NtYieldExecution)
{
    std::this_thread::yield();
    ctx.r3.u64 = STATUS_SUCCESS;
}
