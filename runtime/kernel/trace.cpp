// Bring-up diagnostics.
//
//   TTT2_TRACE=1        log every kernel/XAM import call (thread, name, r3-r6, return address)
//   TTT2_TRACE_RANGE=<lo>-<hi>  the same, only for calls made from guest code in [lo, hi) (hex)
//   TTT2_TRACE_NEW=1    log each (thread, import, return address) the first time it is seen,
//                       with the time since start: a cheap timeline of new call sites
//   TTT2_WATCHDOG=<s>   if no import is called for <s> seconds (default 5), print each guest
//                       thread's last import and, on Windows, the recompiled functions it is
//                       executing right now (symbolised from ttt2.pdb). 0 disables it.

#include "guest.h"
#include "threads.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#ifdef _WIN32
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#endif

namespace guest
{
    // Per-thread bookkeeping is written lock-free by its own thread on every kernel call
    // and only read (under g_mutex, for the map) by the watchdog/sampler.
    struct ThreadInfo
    {
        std::atomic<const char*> lastImport{ "(none)" };
        std::atomic<uint64_t> lastLr{ 0 };
        std::atomic<uint64_t> calls{ 0 };
#ifdef _WIN32
        HANDLE handle = nullptr;
#endif
    };

    static std::mutex g_mutex;
    static std::unordered_map<uint32_t, std::unique_ptr<ThreadInfo>> g_threads;
    static std::atomic<uint64_t> g_totalCalls{ 0 };
    static thread_local ThreadInfo* t_info = nullptr;
    static const bool g_trace = [] { const char* v = std::getenv("TTT2_TRACE"); return v && *v && *v != '0'; }();

    void RegisterGuestThread()
    {
        std::lock_guard lock(g_mutex);
        auto& slot = g_threads[CurrentThreadId()];
        if (!slot)
            slot = std::make_unique<ThreadInfo>();
#ifdef _WIN32
        if (!slot->handle)
            DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &slot->handle, 0, FALSE, DUPLICATE_SAME_ACCESS);
#endif
        t_info = slot.get();
    }

    void UnregisterGuestThread()
    {
        std::lock_guard lock(g_mutex);
        t_info = nullptr;
        auto it = g_threads.find(CurrentThreadId());
        if (it == g_threads.end())
            return;
#ifdef _WIN32
        if (it->second->handle)
            CloseHandle(it->second->handle);
#endif
        g_threads.erase(it);
    }

    void TraceImport(const char* name, const PPCContext& ctx)
    {
        g_totalCalls.fetch_add(1, std::memory_order_relaxed);
        uint64_t lr = 0;
#ifndef PPC_CONFIG_SKIP_LR
        lr = ctx.lr;
#endif
        if (!t_info)
            RegisterGuestThread();
        t_info->lastImport.store(name, std::memory_order_relaxed);
        t_info->lastLr.store(lr, std::memory_order_relaxed);
        t_info->calls.fetch_add(1, std::memory_order_relaxed);
        const uint32_t tid = CurrentThreadId();
        // TTT2_TRACE_RANGE=<lo>-<hi> (hex): trace only imports called from that code range.
        static const auto range = [] {
            std::pair<uint64_t, uint64_t> r{ 0, 0 };
            if (const char* v = std::getenv("TTT2_TRACE_RANGE"))
                std::sscanf(v, "%llx-%llx", (unsigned long long*)&r.first, (unsigned long long*)&r.second);
            return r;
        }();
        static const bool traceNew = [] { const char* v = std::getenv("TTT2_TRACE_NEW"); return v && *v == '1'; }();
        if (traceNew)
        {
            static const auto start = std::chrono::steady_clock::now();
            thread_local std::unordered_set<uint64_t> seen;
            if (seen.insert((lr << 32) ^ uint64_t(uintptr_t(name))).second)
            {
                const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
                std::fprintf(stderr, "[new] %lld ms T%u %s(%08X, %08X, %08X, %08X) lr=%08llX\n", (long long)ms, tid, name,
                    ctx.r3.u32, ctx.r4.u32, ctx.r5.u32, ctx.r6.u32, (unsigned long long)lr);
            }
        }
        if (g_trace || (range.second && lr >= range.first && lr < range.second))
        {
            std::fprintf(stderr, "[trace] T%u %s(%08X, %08X, %08X, %08X) lr=%08llX\n", tid, name,
                ctx.r3.u32, ctx.r4.u32, ctx.r5.u32, ctx.r6.u32, (unsigned long long)lr);
        }
    }

#ifdef _WIN32
    static void DumpThreadStack(HANDLE thread)
    {
        static std::once_flag symInit;
        std::call_once(symInit, [] {
            SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
            SymInitialize(GetCurrentProcess(), nullptr, TRUE);
        });

        if (SuspendThread(thread) == DWORD(-1))
            return;
        CONTEXT context{};
        context.ContextFlags = CONTEXT_FULL;
        if (GetThreadContext(thread, &context))
        {
            STACKFRAME64 frame{};
            frame.AddrPC.Offset = context.Rip;    frame.AddrPC.Mode = AddrModeFlat;
            frame.AddrFrame.Offset = context.Rbp; frame.AddrFrame.Mode = AddrModeFlat;
            frame.AddrStack.Offset = context.Rsp; frame.AddrStack.Mode = AddrModeFlat;

            for (int depth = 0; depth < 48; ++depth)
            {
                if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, GetCurrentProcess(), thread, &frame, &context,
                    nullptr, SymFunctionTableAccess64, SymGetModuleBase64, nullptr) || !frame.AddrPC.Offset)
                    break;

                alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 256]{};
                auto* symbol = reinterpret_cast<SYMBOL_INFO*>(buffer);
                symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
                symbol->MaxNameLen = 255;
                DWORD64 displacement = 0;
                if (SymFromAddr(GetCurrentProcess(), frame.AddrPC.Offset, &displacement, symbol))
                    std::fprintf(stderr, "        #%d %s+0x%llX\n", depth, symbol->Name, (unsigned long long)displacement);
                else
                    std::fprintf(stderr, "        #%d 0x%llX\n", depth, (unsigned long long)frame.AddrPC.Offset);
            }
        }
        ResumeThread(thread);
    }
#endif

    static void WatchdogThread(int seconds)
    {
        uint64_t lastCalls = g_totalCalls.load();
        auto lastChange = std::chrono::steady_clock::now();
        bool reported = false;
        for (;;)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            const uint64_t calls = g_totalCalls.load();
            const auto now = std::chrono::steady_clock::now();
            if (calls != lastCalls)
            {
                lastCalls = calls;
                lastChange = now;
                reported = false;
                continue;
            }
            if (reported || now - lastChange < std::chrono::seconds(seconds))
                continue;

            reported = true;
            // Three samples a second apart: identical stacks mean a tight loop, changing
            // ones mean slow progress or a loop spanning several functions.
            for (int sample = 1; sample <= 3; ++sample)
            {
                {
                    std::lock_guard lock(g_mutex);
                    std::fprintf(stderr, "[watchdog] sample %d: no import calls for %ds (%llu calls so far). Guest threads:\n",
                        sample, seconds, (unsigned long long)calls);
                    for (auto& [tid, info] : g_threads)
                    {
                        std::fprintf(stderr, "    T%u: %llu calls, last %s (lr=%08llX)\n", tid,
                            (unsigned long long)info->calls.load(), info->lastImport.load(), (unsigned long long)info->lastLr.load());
#ifdef _WIN32
                        if (info->handle)
                            DumpThreadStack(info->handle);
#endif
                    }
                }
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
    }

    void StartWatchdog()
    {
        const char* v = std::getenv("TTT2_WATCHDOG");
        const int seconds = v ? std::atoi(v) : 5;
        if (seconds > 0)
            std::thread(WatchdogThread, seconds).detach();

        // TTT2_SAMPLE_AT=<ms>: dump every guest thread's stack once at that time, regardless
        // of activity (the watchdog never fires while any thread keeps calling the kernel).
        if (const char* at = std::getenv("TTT2_SAMPLE_AT"))
        {
            // TTT2_SAMPLE_AT=<ms>[,<count>[,<interval ms>[,<thread id>]]]
            int ms = 0, count = 1, interval = 500;
            unsigned onlyThread = 0;
            std::sscanf(at, "%d,%d,%d,%u", &ms, &count, &interval, &onlyThread);
            std::thread([ms, count, interval, onlyThread] {
                std::this_thread::sleep_for(std::chrono::milliseconds(ms));
                for (int sample = 0; sample < count; sample++)
                {
                    if (sample)
                        std::this_thread::sleep_for(std::chrono::milliseconds(interval));
                    std::lock_guard lock(g_mutex);
                    std::fprintf(stderr, "[sample] at %d ms:\n", ms + sample * interval);
                    for (auto& [tid, info] : g_threads)
                    {
                        if (onlyThread && tid != onlyThread)
                            continue;
                        std::fprintf(stderr, "    T%u: %llu calls, last %s (lr=%08llX)\n", tid,
                            (unsigned long long)info->calls.load(), info->lastImport.load(), (unsigned long long)info->lastLr.load());
#ifdef _WIN32
                        if (info->handle)
                            DumpThreadStack(info->handle);
#endif
                    }
                }
            }).detach();
        }
    }
}

// ---- Checked indirect calls and crash reports --------------------------------

namespace guest
{
    void PrintHostBacktrace()
    {
#ifdef _WIN32
        static std::once_flag symInit;
        std::call_once(symInit, [] {
            SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
            SymInitialize(GetCurrentProcess(), nullptr, TRUE);
        });
        void* frames[48];
        const USHORT count = CaptureStackBackTrace(1, 48, frames, nullptr);
        for (USHORT i = 0; i < count; i++)
        {
            alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 256]{};
            auto* symbol = reinterpret_cast<SYMBOL_INFO*>(buffer);
            symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
            symbol->MaxNameLen = 255;
            DWORD64 displacement = 0;
            if (SymFromAddr(GetCurrentProcess(), DWORD64(frames[i]), &displacement, symbol))
                std::fprintf(stderr, "        #%u %s+0x%llX\n", i, symbol->Name, (unsigned long long)displacement);
            else
                std::fprintf(stderr, "        #%u %p\n", i, frames[i]);
        }
#endif
    }

#ifdef _WIN32
    static LONG WINAPI CrashFilter(EXCEPTION_POINTERS* info)
    {
        const auto* record = info->ExceptionRecord;
        std::fprintf(stderr, "[crash] exception 0x%08lX at %p on guest thread T%u",
            record->ExceptionCode, record->ExceptionAddress, CurrentThreadId());
        if (record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && record->NumberParameters >= 2)
        {
            const uintptr_t address = uintptr_t(record->ExceptionInformation[1]);
            std::fprintf(stderr, " (%s %p", record->ExceptionInformation[0] ? "write to" : "read from", (void*)address);
            if (g_base && address >= uintptr_t(g_base) && address < uintptr_t(g_base) + PPC_MEMORY_SIZE)
                std::fprintf(stderr, " = guest %08X", uint32_t(address - uintptr_t(g_base)));
            std::fprintf(stderr, ")");
        }
        std::fprintf(stderr, "\n");

        // Walk the faulting thread's stack from the exception context.
        CONTEXT context = *info->ContextRecord;
        STACKFRAME64 frame{};
        frame.AddrPC.Offset = context.Rip;    frame.AddrPC.Mode = AddrModeFlat;
        frame.AddrFrame.Offset = context.Rbp; frame.AddrFrame.Mode = AddrModeFlat;
        frame.AddrStack.Offset = context.Rsp; frame.AddrStack.Mode = AddrModeFlat;
        SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
        SymInitialize(GetCurrentProcess(), nullptr, TRUE);
        for (int depth = 0; depth < 48; ++depth)
        {
            if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, GetCurrentProcess(), GetCurrentThread(), &frame, &context,
                nullptr, SymFunctionTableAccess64, SymGetModuleBase64, nullptr) || !frame.AddrPC.Offset)
                break;
            alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 256]{};
            auto* symbol = reinterpret_cast<SYMBOL_INFO*>(buffer);
            symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
            symbol->MaxNameLen = 255;
            DWORD64 displacement = 0;
            if (SymFromAddr(GetCurrentProcess(), frame.AddrPC.Offset, &displacement, symbol))
                std::fprintf(stderr, "        #%d %s+0x%llX\n", depth, symbol->Name, (unsigned long long)displacement);
            else
                std::fprintf(stderr, "        #%d 0x%llX\n", depth, (unsigned long long)frame.AddrPC.Offset);
        }
        std::fflush(stderr);
        return EXCEPTION_EXECUTE_HANDLER;
    }
#endif

    void InstallCrashHandler()
    {
#ifdef _WIN32
        SetUnhandledExceptionFilter(CrashFilter);
#endif
    }
}

void TTT2CallIndirect(PPCContext& ctx, uint8_t* base, uint32_t target)
{
    if (target >= PPC_CODE_BASE && target < PPC_CODE_BASE + PPC_CODE_SIZE) [[likely]]
    {
        if (PPCFunc* function = PPC_LOOKUP_FUNC(base, target)) [[likely]]
        {
            function(ctx, base);
            return;
        }
    }

    uint64_t lr = 0;
#ifndef PPC_CONFIG_SKIP_LR
    lr = ctx.lr;
#endif
    std::fprintf(stderr, "[crash] indirect call to %08X, which is not a recompiled function (lr=%08llX, r3=%08X, thread T%u)\n",
        target, (unsigned long long)lr, ctx.r3.u32, guest::CurrentThreadId());
    guest::PrintHostBacktrace();
    std::fflush(stderr);
    std::abort();
}
