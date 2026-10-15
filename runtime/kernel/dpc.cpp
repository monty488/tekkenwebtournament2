// Deferred procedure calls. The graphics interrupt handler queues DPCs, whose routines
// signal the events the game's GPU worker threads wait on. They run in order on one
// dispatcher thread with its own guest environment, as on the console's DPC level.
//
// KDPC layout (per Xenia): +0x00 Type (u16), +0x02 SelectedCpu, +0x03 DesiredCpu,
// +0x04 DpcListEntry (8 bytes), +0x0C DeferredRoutine, +0x10 DeferredContext,
// +0x14 SystemArgument1, +0x18 SystemArgument2.

#include "guest.h"
#include "threads.h"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_set>

namespace
{
    struct QueuedDpc
    {
        uint32_t dpc, routine, context, arg1, arg2;
    };

    std::mutex g_mutex;
    std::condition_variable g_cv;
    std::deque<QueuedDpc> g_queue;
    std::unordered_set<uint32_t> g_queued; // DPCs currently in the queue (a DPC queues at most once)
    std::once_flag g_started;

    void DpcThread()
    {
        guest::GuestEnv env = guest::CreateGuestEnv(/*cpu*/ 2);
        for (;;)
        {
            QueuedDpc item;
            {
                std::unique_lock lock(g_mutex);
                g_cv.wait(lock, [] { return !g_queue.empty(); });
                item = g_queue.front();
                g_queue.pop_front();
                g_queued.erase(item.dpc);
            }
            guest::CallGuest(env, item.routine, { item.dpc, item.context, item.arg1, item.arg2 });
        }
    }
}

using namespace guest;

// void KeInitializeDpc(KDPC* dpc, u32 routine, u32 context)
GUEST_IMPORT(KeInitializeDpc)
{
    const uint32_t dpc = ctx.r3.u32;
    Store<uint16_t>(dpc + 0x00, 19);   // DpcObject
    *ToHost<uint8_t>(dpc + 0x02) = 0;
    *ToHost<uint8_t>(dpc + 0x03) = 0;
    Store<uint32_t>(dpc + 0x04, 0);
    Store<uint32_t>(dpc + 0x08, 0);
    Store<uint32_t>(dpc + 0x0C, ctx.r4.u32);
    Store<uint32_t>(dpc + 0x10, ctx.r5.u32);
    Store<uint32_t>(dpc + 0x14, 0);
    Store<uint32_t>(dpc + 0x18, 0);
}

// BOOLEAN KeInsertQueueDpc(KDPC* dpc, u32 arg1, u32 arg2): false if already queued.
GUEST_IMPORT(KeInsertQueueDpc)
{
    std::call_once(g_started, [] { std::thread(DpcThread).detach(); });

    const uint32_t dpc = ctx.r3.u32;
    QueuedDpc item{ dpc, Load<uint32_t>(dpc + 0x0C), Load<uint32_t>(dpc + 0x10), ctx.r4.u32, ctx.r5.u32 };
    bool inserted;
    {
        std::lock_guard lock(g_mutex);
        inserted = g_queued.insert(dpc).second;
        if (inserted)
        {
            Store<uint32_t>(dpc + 0x14, item.arg1);
            Store<uint32_t>(dpc + 0x18, item.arg2);
            g_queue.push_back(item);
        }
    }
    if (inserted)
        g_cv.notify_one();
    ctx.r3.u64 = inserted;
}

// BOOLEAN KeRemoveQueueDpc(KDPC* dpc)
GUEST_IMPORT(KeRemoveQueueDpc)
{
    const uint32_t dpc = ctx.r3.u32;
    std::lock_guard lock(g_mutex);
    bool removed = false;
    if (g_queued.erase(dpc))
    {
        for (auto it = g_queue.begin(); it != g_queue.end(); ++it)
        {
            if (it->dpc == dpc)
            {
                g_queue.erase(it);
                removed = true;
                break;
            }
        }
    }
    ctx.r3.u64 = removed;
}

// Critical regions and APC no-ops: user-mode code on a single-title console needs no
// extra exclusion here.
GUEST_IMPORT(KeEnterCriticalRegion) {}
GUEST_IMPORT(KeLeaveCriticalRegion) {}
GUEST_IMPORT(KiApcNormalRoutineNop) {}
