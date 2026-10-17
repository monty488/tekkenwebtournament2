#include "sync.h"
#include "memory.h"
#include "threads.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>

namespace guest
{
    static std::mutex g_waitMutex;
    static std::condition_variable g_waitCv;

    std::mutex& WaitMutex() { return g_waitMutex; }
    std::condition_variable& WaitCondition() { return g_waitCv; }
    void NotifyWaiters() { g_waitCv.notify_all(); }

    // Dispatcher header types (XDISPATCHER_HEADER.Type).
    enum : uint8_t
    {
        kNotificationEvent = 0, kSynchronizationEvent = 1, kMutant = 2,
        kSemaphore = 5, kThread = 6, kNotificationTimer = 8, kSynchronizationTimer = 9,
    };

    static thread_local std::shared_ptr<GuestThread> t_currentThread;

    std::shared_ptr<GuestThread> CurrentGuestThread()
    {
        if (!t_currentThread)
        {
            t_currentThread = std::make_shared<GuestThread>();
            t_currentThread->id = CurrentThreadId();
        }
        return t_currentThread;
    }

    void SetCurrentGuestThread(std::shared_ptr<GuestThread> thread) { t_currentThread = std::move(thread); }

    std::shared_ptr<Waitable> WaitableFromHandle(uint32_t handle)
    {
        if (handle == 0xFFFFFFFE) // NtCurrentThread()
            return CurrentGuestThread();
        return Handles().Get<Waitable>(handle);
    }

    // Guest dispatcher headers -> host objects.
    static std::unordered_map<uint32_t, std::shared_ptr<Waitable>> g_guestObjects; // guarded by g_waitMutex

    static void RegisterGuestObject(uint32_t address, std::shared_ptr<Waitable> object)
    {
        object->guestObject = address;
        g_guestObjects[address] = std::move(object);
    }

    std::shared_ptr<Waitable> WaitableFromGuest(uint32_t address)
    {
        std::lock_guard lock(g_waitMutex);
        auto it = g_guestObjects.find(address);
        if (it != g_guestObjects.end())
            return it->second;

        const uint8_t type = *ToHost<uint8_t>(address);
        const int32_t signalState = Load<int32_t>(address + 4);
        std::shared_ptr<Waitable> object;
        switch (type)
        {
        case kNotificationEvent:
        case kSynchronizationEvent:
        case kNotificationTimer:
        case kSynchronizationTimer:
        {
            auto e = std::make_shared<Event>();
            e->manualReset = type == kNotificationEvent || type == kNotificationTimer;
            e->signaled = signalState != 0;
            object = e;
            break;
        }
        case kMutant:
        {
            auto m = std::make_shared<Mutant>();
            object = m;
            break;
        }
        case kSemaphore:
        {
            auto s = std::make_shared<Semaphore>();
            s->count = signalState;
            s->maximum = Load<int32_t>(address + 0x10); // KSEMAPHORE.Limit
            object = s;
            break;
        }
        default:
            TTT2_LOG("WaitableFromGuest: unsupported dispatcher type %u at %08X", type, address);
            object = std::make_shared<Event>(); // never signalled
            break;
        }
        RegisterGuestObject(address, object);
        return object;
    }

    // Keeps the guest-visible SignalState in step for objects that have a guest header.
    static void SyncGuestHeader(Waitable& object)
    {
        if (!object.guestObject)
            return;
        int32_t state = 0;
        if (auto* e = dynamic_cast<Event*>(&object)) state = e->signaled;
        else if (auto* s = dynamic_cast<Semaphore*>(&object)) state = s->count;
        else if (auto* m = dynamic_cast<Mutant*>(&object)) state = m->owner == 0;
        else if (auto* t = dynamic_cast<GuestThread*>(&object)) state = t->finished;
        Store<int32_t>(object.guestObject + 4, state);
    }

    static std::optional<std::chrono::steady_clock::time_point> Deadline(uint32_t timeout)
    {
        if (!timeout)
            return std::nullopt; // infinite
        int64_t value = Load<int64_t>(timeout);
        if (value > 0)
        {
            // Absolute FILETIME: convert to a relative delay from now.
            using namespace std::chrono;
            const int64_t now = 116444736000000000ll + duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count() / 100;
            value = -(value - now);
            if (value > 0)
                value = 0;
        }
        return std::chrono::steady_clock::now() + std::chrono::nanoseconds(-value * 100);
    }

    static bool WaitStatsEnabled()
    {
        static const bool enabled = [] { const char* v = std::getenv("TTT2_WAIT_STATS"); return v && *v == '1'; }();
        return enabled;
    }

    // Time from a signal handing an object to a blocked waiter until that waiter runs.
    static void RecordWakeLatency(std::chrono::steady_clock::duration latency)
    {
        static std::atomic<uint64_t> count{ 0 }, totalNs{ 0 }, maxNs{ 0 }, over100us{ 0 }, over1ms{ 0 };
        static std::atomic<int64_t> lastPrint{ 0 };
        if (latency.count() < 0)
            return; // browser workers' clocks can disagree slightly
        const uint64_t ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(latency).count());
        ++count;
        totalNs += ns;
        if (ns > 100000) ++over100us;
        if (ns > 1000000) ++over1ms;
        for (uint64_t m = maxNs.load(); ns > m && !maxNs.compare_exchange_weak(m, ns);) {}
        const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        int64_t last = lastPrint.load();
        if (!last)
            lastPrint.compare_exchange_strong(last, now);
        else if (now - last >= 5000 && lastPrint.compare_exchange_strong(last, now))
        {
            const uint64_t c = count.exchange(0);
            std::fprintf(stderr, "[waits] wake latency: %llu wakes, avg %.1f us, max %.1f ms, %llu over 100 us, %llu over 1 ms\n",
                (unsigned long long)c, c ? totalNs.exchange(0) / 1e3 / double(c) : 0.0, maxNs.exchange(0) / 1e6,
                (unsigned long long)over100us.exchange(0), (unsigned long long)over1ms.exchange(0));
        }
    }

    void DeliverSignals(Waitable& object)
    {
        for (auto it = object.waiters.begin(); it != object.waiters.end();)
        {
            Waiter* w = *it;
            if (w->satisfiedIndex >= 0)
            {
                ++it; // already woken by another object; it unregisters itself
                continue;
            }
            if (w->waitAll)
            {
                // Takes nothing until all its objects are signalled; let it re-check.
                if (object.IsSignaled(w->threadId))
                    w->cv.notify_one();
                ++it;
                continue;
            }
            if (!object.IsSignaled(w->threadId))
                break; // FIFO: later waiters cannot jump the queue
            object.Acquire(w->threadId);
            for (size_t i = 0; i < w->objects->size(); i++)
            {
                if ((*w->objects)[i].get() == &object)
                {
                    w->satisfiedIndex = int(i);
                    break;
                }
            }
            it = object.waiters.erase(it);
            if (WaitStatsEnabled())
                w->signalledAt = std::chrono::steady_clock::now();
            w->cv.notify_one();
        }
        SyncGuestHeader(object);
    }

    // Called with the wait mutex held.
    static void ReportLongWait(uint32_t tid, const std::vector<std::shared_ptr<Waitable>>& objects)
    {
        std::string what;
        char buf[160];
        for (const auto& o : objects)
        {
            if (!o)
                continue;
            if (auto* m = dynamic_cast<Mutant*>(o.get()))
                std::snprintf(buf, sizeof(buf), " mutant %p (owner T%u, recursion %u, %zu waiting)", (void*)m, m->owner, m->recursion, m->waiters.size());
            else if (auto* e = dynamic_cast<Event*>(o.get()))
                std::snprintf(buf, sizeof(buf), " %s event %p (%s, %zu waiting)", e->manualReset ? "manual" : "auto", (void*)e,
                    e->signaled ? "signalled" : "not signalled", e->waiters.size());
            else if (auto* sem = dynamic_cast<Semaphore*>(o.get()))
                std::snprintf(buf, sizeof(buf), " semaphore %p (count %d/%d, %zu waiting)", (void*)sem, sem->count, sem->maximum, sem->waiters.size());
            else if (auto* t = dynamic_cast<GuestThread*>(o.get()))
                std::snprintf(buf, sizeof(buf), " thread T%u (%s)", t->id, t->finished ? "finished" : "running");
            else
                std::snprintf(buf, sizeof(buf), " object %p", (void*)o.get());
            what += buf;
        }
        TTT2_LOG("T%u waiting >2s on%s", tid, what.c_str());
    }

    // Consecutive waits satisfied by an event the waiting thread set itself.
    static thread_local uint32_t t_selfTaken = 0;
    constexpr uint32_t kSelfSpinLimit = 64;

    uint32_t WaitForObjects(const std::vector<std::shared_ptr<Waitable>>& objects, bool waitAll, uint32_t timeout)
    {
        const uint32_t tid = CurrentThreadId();
        const auto deadline = Deadline(timeout);

        std::unique_lock lock(g_waitMutex);

        // Wait-any waiters queue on every object so signals can be handed to them.
        Waiter waiter;
        waiter.threadId = tid;
        waiter.objects = &objects;
        waiter.waitAll = waitAll;
        bool queued = false;
        auto unqueue = [&] {
            if (!queued)
                return;
            for (auto& o : objects)
                if (o)
                    std::erase(o->waiters, &waiter);
            queued = false;
        };

        for (;;)
        {
            if (waiter.satisfiedIndex >= 0)
            {
                unqueue();
                if (WaitStatsEnabled() && waiter.signalledAt.time_since_epoch().count())
                    RecordWakeLatency(std::chrono::steady_clock::now() - waiter.signalledAt);
                return STATUS_SUCCESS + uint32_t(waiter.satisfiedIndex);
            }

            if (waitAll)
            {
                bool all = true;
                for (auto& o : objects)
                    all = all && o && o->IsSignaled(tid);
                if (all)
                {
                    unqueue();
                    for (auto& o : objects)
                    {
                        o->Acquire(tid);
                        SyncGuestHeader(*o);
                    }
                    return STATUS_SUCCESS;
                }
            }
            else
            {
                for (size_t i = 0; i < objects.size(); i++)
                {
                    // Only take state no earlier, still-blocked waiter is owed.
                    auto owedToEarlier = [&](const Waitable& o) {
                        // A mutant's owner re-acquires it recursively whoever is queued:
                        // making it wait behind them would deadlock (they wait for it).
                        if (auto* m = dynamic_cast<const Mutant*>(&o); m && m->owner == tid)
                            return false;
                        for (Waiter* w : o.waiters)
                        {
                            if (w == &waiter)
                                return false;
                            if (w->satisfiedIndex < 0 && !w->waitAll)
                                return true;
                        }
                        return false;
                    };
                    if (objects[i] && objects[i]->IsSignaled(tid) && !owedToEarlier(*objects[i]))
                    {
                        // A thread that keeps taking back an auto-reset event it set itself
                        // (nobody else was waiting) is spinning: the game's "wait until the
                        // counter reaches N" loop sets its event to pass the wakeup on and
                        // then waits on it again. Each round costs the global wait lock, which
                        // every other thread needs (badly so in the browser). After a run of
                        // these, consume the signal but block up to 1 ms first; a set from
                        // another thread still wakes the waiter at once.
                        auto* e = dynamic_cast<Event*>(objects[i].get());
                        if (e && !e->manualReset && e->setter == tid && ++t_selfTaken > kSelfSpinLimit)
                        {
                            e->signaled = false;
                            SyncGuestHeader(*e);
                            if (!queued)
                            {
                                for (auto& o : objects)
                                    if (o)
                                        o->waiters.push_back(&waiter);
                                queued = true;
                            }
                            auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(1);
                            if (deadline && *deadline < until)
                                until = *deadline;
                            while (waiter.satisfiedIndex < 0 && waiter.cv.wait_until(lock, until) != std::cv_status::timeout) {}
                            const int index = waiter.satisfiedIndex >= 0 ? waiter.satisfiedIndex : int(i);
                            unqueue();
                            return STATUS_SUCCESS + uint32_t(index);
                        }
                        if (e && e->setter != tid)
                            t_selfTaken = 0; // (taking a free mutex in between is part of the loop)
                        objects[i]->Acquire(tid);
                        SyncGuestHeader(*objects[i]);
                        unqueue();
                        return STATUS_SUCCESS + uint32_t(i);
                    }
                }
            }

            if (!queued)
            {
                for (auto& o : objects)
                    if (o)
                        o->waiters.push_back(&waiter);
                queued = true;
            }
            t_selfTaken = 0; // really blocking

            if (deadline)
            {
                // Checked after a final pass over the objects above, so a signal that
                // races the deadline still wins.
                if (std::chrono::steady_clock::now() >= *deadline)
                {
                    unqueue();
                    return STATUS_TIMEOUT;
                }
                waiter.cv.wait_until(lock, *deadline);
            }
            else
            {
                // Infinite waits report once if they last over 2 s (what is held, by whom).
                if (waiter.cv.wait_for(lock, std::chrono::seconds(2)) == std::cv_status::timeout && !waiter.reported &&
                    waiter.satisfiedIndex < 0)
                {
                    waiter.reported = true;
                    ReportLongWait(tid, objects);
                }
            }
        }
    }

    WaitStat::WaitStat(uint32_t lr, uint32_t timeout)
    {
        if (!WaitStatsEnabled() || !timeout)
            return;
        const int64_t value = Load<int64_t>(timeout);
        this->lr = lr;
        requestedNs = value <= 0 ? -value * 100 : 0; // absolute deadlines show as 0
        start = std::chrono::steady_clock::now();
    }

    WaitStat::WaitStat(uint32_t lr, int64_t requestedNs)
    {
        if (!WaitStatsEnabled())
            return;
        this->lr = lr;
        this->requestedNs = requestedNs;
        start = std::chrono::steady_clock::now();
    }

    WaitStat::~WaitStat()
    {
        if (requestedNs < 0)
            return;
        struct Entry { uint64_t count = 0, requestedNs = 0, actualNs = 0, maxNs = 0; uint32_t thread = 0; };
        static std::mutex mutex;
        static std::unordered_map<uint32_t, Entry> sites;
        static auto lastPrint = std::chrono::steady_clock::now();
        const auto now = std::chrono::steady_clock::now();
        const uint64_t actual = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count());
        std::lock_guard lock(mutex);
        Entry& e = sites[lr];
        ++e.count;
        e.requestedNs += uint64_t(requestedNs);
        e.actualNs += actual;
        e.maxNs = std::max(e.maxNs, actual);
        e.thread = CurrentThreadId();
        if (now - lastPrint < std::chrono::seconds(5))
            return;
        std::vector<std::pair<uint32_t, Entry>> top(sites.begin(), sites.end());
        std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second.actualNs > b.second.actualNs; });
        const double seconds = std::chrono::duration<double>(now - lastPrint).count();
        std::fprintf(stderr, "[waits] timed waits and sleeps over %.1f s, by time taken:\n", seconds);
        for (size_t i = 0; i < top.size() && i < 10; i++)
        {
            const Entry& t = top[i].second;
            std::fprintf(stderr, "[waits]   lr %08X T%u: %llu calls, asked %.3f ms avg, took %.3f ms avg (max %.1f), %.0f ms total/s\n", top[i].first,
                t.thread, (unsigned long long)t.count, t.requestedNs / 1e6 / double(t.count), t.actualNs / 1e6 / double(t.count), t.maxNs / 1e6,
                t.actualNs / 1e6 / seconds);
        }
        sites.clear();
        lastPrint = now;
    }

    uint32_t SignalObject(const std::shared_ptr<Waitable>& object)
    {
        std::lock_guard lock(g_waitMutex);
        if (auto* e = dynamic_cast<Event*>(object.get()))
        {
            e->signaled = true;
            e->setter = CurrentThreadId();
        }
        else if (auto* m = dynamic_cast<Mutant*>(object.get()))
        {
            if (m->owner != CurrentThreadId())
                return STATUS_MUTANT_NOT_OWNED;
            if (--m->recursion == 0)
                m->owner = 0;
        }
        else if (auto* s = dynamic_cast<Semaphore*>(object.get()))
        {
            if (s->count >= s->maximum)
                return STATUS_SEMAPHORE_LIMIT_EXCEEDED;
            ++s->count;
        }
        DeliverSignals(*object);
        return STATUS_SUCCESS;
    }

    // Guest object (dispatcher header) for a handle, for ObReferenceObjectByHandle.
    static uint32_t GuestObjectFor(const std::shared_ptr<Waitable>& object)
    {
        std::lock_guard lock(g_waitMutex);
        if (object->guestObject)
            return object->guestObject;

        uint8_t type = kNotificationEvent;
        uint32_t size = 0x10;
        if (auto* e = dynamic_cast<Event*>(object.get())) type = e->manualReset ? kNotificationEvent : kSynchronizationEvent;
        else if (dynamic_cast<Mutant*>(object.get())) { type = kMutant; size = 0x20; }
        else if (dynamic_cast<Semaphore*>(object.get())) { type = kSemaphore; size = 0x14; }
        else if (dynamic_cast<GuestThread*>(object.get())) { type = kThread; size = 0x200; } // KTHREAD

        const uint32_t address = AllocVirtual(size);
        *ToHost<uint8_t>(address) = type;
        RegisterGuestObject(address, object);
        SyncGuestHeader(*object);
        return address;
    }
}

using namespace guest;

static uint32_t NewHandle(uint32_t pHandle, std::shared_ptr<Waitable> object)
{
    if (!pHandle)
        return STATUS_INVALID_PARAMETER;
    Store<uint32_t>(pHandle, Handles().Insert(std::move(object)));
    return STATUS_SUCCESS;
}

// ---- Events ----------------------------------------------------------------

// NTSTATUS NtCreateEvent(u32* handle, OBJECT_ATTRIBUTES* attrs, u32 eventType, u32 initialState)
// eventType: 0 = NotificationEvent (manual reset), 1 = SynchronizationEvent (auto reset)
GUEST_IMPORT(NtCreateEvent)
{
    auto e = std::make_shared<Event>();
    e->manualReset = ctx.r5.u32 == 0;
    e->signaled = ctx.r6.u32 != 0;
    ctx.r3.u64 = NewHandle(ctx.r3.u32, e);
}

// NTSTATUS NtSetEvent(u32 handle, u32* previousState)
GUEST_IMPORT(NtSetEvent)
{
    auto e = std::dynamic_pointer_cast<Event>(WaitableFromHandle(ctx.r3.u32));
    if (!e) { ctx.r3.u64 = STATUS_INVALID_HANDLE; return; }
    const uint32_t pPrevious = ctx.r4.u32;
    {
        std::lock_guard lock(WaitMutex());
        if (pPrevious) Store<int32_t>(pPrevious, e->signaled);
    }
    ctx.r3.u64 = SignalObject(e);
}

// NTSTATUS NtClearEvent(u32 handle)
GUEST_IMPORT(NtClearEvent)
{
    auto e = std::dynamic_pointer_cast<Event>(WaitableFromHandle(ctx.r3.u32));
    if (!e) { ctx.r3.u64 = STATUS_INVALID_HANDLE; return; }
    std::lock_guard lock(WaitMutex());
    e->signaled = false;
    if (e->guestObject) Store<int32_t>(e->guestObject + 4, 0);
    ctx.r3.u64 = STATUS_SUCCESS;
}

// NTSTATUS NtPulseEvent(u32 handle, u32* previousState): release current waiters, leave reset.
GUEST_IMPORT(NtPulseEvent)
{
    auto e = std::dynamic_pointer_cast<Event>(WaitableFromHandle(ctx.r3.u32));
    if (!e) { ctx.r3.u64 = STATUS_INVALID_HANDLE; return; }
    {
        // Satisfy the threads waiting right now (one for auto-reset, all for manual),
        // then leave the event reset, atomically.
        std::lock_guard lock(WaitMutex());
        if (ctx.r4.u32) Store<int32_t>(ctx.r4.u32, e->signaled);
        e->signaled = true;
        DeliverSignals(*e);
        e->signaled = false;
        if (e->guestObject) Store<int32_t>(e->guestObject + 4, 0);
    }
    ctx.r3.u64 = STATUS_SUCCESS;
}

// LONG KeSetEvent(KEVENT* event, LONG increment, BOOLEAN wait) -> previous state
GUEST_IMPORT(KeSetEvent)
{
    auto e = std::dynamic_pointer_cast<Event>(WaitableFromGuest(ctx.r3.u32));
    int32_t previous = 0;
    if (e)
    {
        { std::lock_guard lock(WaitMutex()); previous = e->signaled; }
        SignalObject(e);
    }
    ctx.r3.u64 = uint32_t(previous);
}

// LONG KeResetEvent(KEVENT* event) -> previous state
GUEST_IMPORT(KeResetEvent)
{
    auto e = std::dynamic_pointer_cast<Event>(WaitableFromGuest(ctx.r3.u32));
    int32_t previous = 0;
    if (e)
    {
        std::lock_guard lock(WaitMutex());
        previous = e->signaled;
        e->signaled = false;
        Store<int32_t>(ctx.r3.u32 + 4, 0);
    }
    ctx.r3.u64 = uint32_t(previous);
}

// ---- Mutants and semaphores ------------------------------------------------

// NTSTATUS NtCreateMutant(u32* handle, OBJECT_ATTRIBUTES* attrs, u32 initialOwner)
GUEST_IMPORT(NtCreateMutant)
{
    auto m = std::make_shared<Mutant>();
    if (ctx.r5.u32)
    {
        m->owner = CurrentThreadId();
        m->recursion = 1;
    }
    ctx.r3.u64 = NewHandle(ctx.r3.u32, m);
}

// NTSTATUS NtReleaseMutant(u32 handle, u32* previousCount)
GUEST_IMPORT(NtReleaseMutant)
{
    auto m = std::dynamic_pointer_cast<Mutant>(WaitableFromHandle(ctx.r3.u32));
    if (!m) { ctx.r3.u64 = STATUS_INVALID_HANDLE; return; }
    if (ctx.r4.u32)
    {
        std::lock_guard lock(WaitMutex());
        Store<int32_t>(ctx.r4.u32, int32_t(m->recursion));
    }
    ctx.r3.u64 = SignalObject(m);
}

// NTSTATUS NtCreateSemaphore(u32* handle, OBJECT_ATTRIBUTES* attrs, s32 initialCount, s32 maximumCount)
GUEST_IMPORT(NtCreateSemaphore)
{
    auto s = std::make_shared<Semaphore>();
    s->count = ctx.r5.s32;
    s->maximum = ctx.r6.s32;
    ctx.r3.u64 = NewHandle(ctx.r3.u32, s);
}

// NTSTATUS NtReleaseSemaphore(u32 handle, s32 releaseCount, s32* previousCount)
GUEST_IMPORT(NtReleaseSemaphore)
{
    auto s = std::dynamic_pointer_cast<Semaphore>(WaitableFromHandle(ctx.r3.u32));
    if (!s) { ctx.r3.u64 = STATUS_INVALID_HANDLE; return; }
    const int32_t release = ctx.r4.s32;
    {
        std::lock_guard lock(WaitMutex());
        if (ctx.r5.u32) Store<int32_t>(ctx.r5.u32, s->count);
        if (s->count + release > s->maximum)
        {
            ctx.r3.u64 = STATUS_SEMAPHORE_LIMIT_EXCEEDED;
            return;
        }
        s->count += release;
        DeliverSignals(*s);
    }
    ctx.r3.u64 = STATUS_SUCCESS;
}

// ---- Waits -----------------------------------------------------------------

// NTSTATUS NtWaitForSingleObjectEx(u32 handle, u32 waitMode, u32 alertable, s64* timeout)
GUEST_IMPORT(NtWaitForSingleObjectEx)
{
    auto object = WaitableFromHandle(ctx.r3.u32);
    if (!object) { ctx.r3.u64 = STATUS_INVALID_HANDLE; return; }
    WaitStat stat(uint32_t(ctx.lr), ctx.r6.u32);
    ctx.r3.u64 = WaitForObjects({ object }, false, ctx.r6.u32);
}

// NTSTATUS NtSignalAndWaitForSingleObjectEx(u32 signal, u32 wait, u32 waitMode, u32 alertable, s64* timeout)
GUEST_IMPORT(NtSignalAndWaitForSingleObjectEx)
{
    auto signal = WaitableFromHandle(ctx.r3.u32);
    auto wait = WaitableFromHandle(ctx.r4.u32);
    if (!signal || !wait) { ctx.r3.u64 = STATUS_INVALID_HANDLE; return; }
    const uint32_t status = SignalObject(signal);
    if (status != STATUS_SUCCESS) { ctx.r3.u64 = status; return; }
    WaitStat stat(uint32_t(ctx.lr), ctx.r7.u32);
    ctx.r3.u64 = WaitForObjects({ wait }, false, ctx.r7.u32);
}

// NTSTATUS KeWaitForSingleObject(void* object, u32 reason, u32 mode, u32 alertable, s64* timeout)
GUEST_IMPORT(KeWaitForSingleObject)
{
    auto object = WaitableFromGuest(ctx.r3.u32);
    WaitStat stat(uint32_t(ctx.lr), ctx.r7.u32);
    ctx.r3.u64 = WaitForObjects({ object }, false, ctx.r7.u32);
}

// NTSTATUS KeWaitForMultipleObjects(u32 count, void** objects, u32 waitType /* 0 all, 1 any */,
//                                   u32 reason, u32 mode, u32 alertable, s64* timeout, void* waitBlocks)
GUEST_IMPORT(KeWaitForMultipleObjects)
{
    std::vector<std::shared_ptr<Waitable>> objects;
    for (uint32_t i = 0; i < ctx.r3.u32; i++)
        objects.push_back(WaitableFromGuest(Load<uint32_t>(ctx.r4.u32 + i * 4)));
    WaitStat stat(uint32_t(ctx.lr), ctx.r9.u32);
    ctx.r3.u64 = WaitForObjects(objects, ctx.r5.u32 == 0, ctx.r9.u32);
}

// ---- Object references -----------------------------------------------------

// NTSTATUS ObReferenceObjectByHandle(u32 handle, u32 objectType, void** object)
GUEST_IMPORT(ObReferenceObjectByHandle)
{
    auto object = WaitableFromHandle(ctx.r3.u32);
    if (!object)
    {
        ctx.r3.u64 = STATUS_INVALID_HANDLE;
        return;
    }
    if (ctx.r5.u32)
        Store<uint32_t>(ctx.r5.u32, GuestObjectFor(object));
    ctx.r3.u64 = STATUS_SUCCESS;
}

GUEST_IMPORT(ObDereferenceObject)
{
    // Objects live as long as their handles; nothing to release.
}

// NTSTATUS NtDuplicateObject(u32 handle, u32* newHandle, u32 options)
GUEST_IMPORT(NtDuplicateObject)
{
    auto object = WaitableFromHandle(ctx.r3.u32);
    if (!object) { ctx.r3.u64 = STATUS_INVALID_HANDLE; return; }
    if (ctx.r4.u32)
        Store<uint32_t>(ctx.r4.u32, Handles().Insert(object));
    if (ctx.r5.u32 & 1) // DUPLICATE_CLOSE_SOURCE
        Handles().Close(ctx.r3.u32);
    ctx.r3.u64 = STATUS_SUCCESS;
}

// ---- Spinlocks ---------------------------------------------------------------
// A KSPIN_LOCK is one dword: 0 when free, the owner's id while held. Interrupt levels
// are not modelled, so the "raised IRQL" variants behave the same.

static void AcquireSpinLock(uint32_t address)
{
    std::atomic_ref<uint32_t> lock(*ToHost<uint32_t>(address));
    const uint32_t self = CurrentThreadId();
    for (uint32_t spins = 0;; ++spins)
    {
        uint32_t expected = 0;
        if (lock.compare_exchange_weak(expected, self, std::memory_order_acquire))
            return;
        if (spins > 64)
            std::this_thread::yield();
    }
}

static void ReleaseSpinLock(uint32_t address)
{
    std::atomic_ref<uint32_t>(*ToHost<uint32_t>(address)).store(0, std::memory_order_release);
}

// KIRQL KfAcquireSpinLock(KSPIN_LOCK* lock) -> previous IRQL
GUEST_IMPORT(KfAcquireSpinLock)
{
    AcquireSpinLock(ctx.r3.u32);
    ctx.r3.u64 = 0;
}

// void KfReleaseSpinLock(KSPIN_LOCK* lock, KIRQL oldIrql)
GUEST_IMPORT(KfReleaseSpinLock)
{
    ReleaseSpinLock(ctx.r3.u32);
}

GUEST_IMPORT(KeAcquireSpinLockAtRaisedIrql)
{
    AcquireSpinLock(ctx.r3.u32);
}

GUEST_IMPORT(KeReleaseSpinLockFromRaisedIrql)
{
    ReleaseSpinLock(ctx.r3.u32);
}
