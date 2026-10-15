#pragma once

// Kernel dispatcher objects: events, mutants, semaphores and threads, waitable
// through handles (Nt* APIs) or through dispatcher headers embedded in guest
// memory (Ke* APIs). One global lock guards all object state, which keeps
// multi-object waits simple and correct; blocked threads sleep on their own
// condition variable (see Waiter). WaitCondition() is only for suspended threads.

#include "guest.h"
#include "objects.h"

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace guest
{
    std::mutex& WaitMutex();
    std::condition_variable& WaitCondition();
    void NotifyWaiters();

    struct Waitable;

    // A thread blocked in WaitForObjects, queued on each object it waits for. Signals are
    // handed to the oldest eligible wait-any waiter directly, as the NT kernel does: an
    // auto-reset event set while a thread waits on it wakes that thread and stays
    // unsignalled, so the setter cannot take it back by waiting on it right away.
    // Wait-all waiters are only woken to re-check. Each waiter has its own condition
    // variable, so a signal wakes just the threads queued on that object rather than
    // every blocked guest thread (games that spin on Set/WaitForSingleObject would
    // otherwise keep the whole process churning through wakeups).
    struct Waiter
    {
        uint32_t threadId = 0;
        const std::vector<std::shared_ptr<Waitable>>* objects = nullptr;
        bool waitAll = false;
        int satisfiedIndex = -1;
        std::condition_variable cv;
    };

    struct Waitable : KernelObject
    {
        // Both called with WaitMutex() held.
        virtual bool IsSignaled(uint32_t threadId) = 0;
        virtual void Acquire(uint32_t threadId) {}

        std::deque<Waiter*> waiters; // guarded by WaitMutex()

        // Guest-visible object (dispatcher header) for ObReferenceObjectByHandle/Ke* APIs.
        uint32_t guestObject = 0;
    };

    // Hands the object's available state to queued waiters and wakes them (call with
    // WaitMutex() held, after changing the state), then updates the guest header.
    void DeliverSignals(Waitable& object);

    struct Event : Waitable
    {
        bool manualReset = false;
        bool signaled = false;
        bool IsSignaled(uint32_t) override { return signaled; }
        void Acquire(uint32_t) override { if (!manualReset) signaled = false; }
    };

    struct Mutant : Waitable
    {
        uint32_t owner = 0;
        uint32_t recursion = 0;
        bool IsSignaled(uint32_t tid) override { return owner == 0 || owner == tid; }
        void Acquire(uint32_t tid) override { owner = tid; ++recursion; }
    };

    struct Semaphore : Waitable
    {
        int32_t count = 0;
        int32_t maximum = 1;
        bool IsSignaled(uint32_t) override { return count > 0; }
        void Acquire(uint32_t) override { --count; }
    };

    struct GuestThread : Waitable
    {
        uint32_t id = 0;
        int suspendCount = 0;      // guarded by WaitMutex()
        bool finished = false;     // guarded by WaitMutex()
        uint32_t exitCode = 0;
        bool IsSignaled(uint32_t) override { return finished; }
    };

    // The GuestThread object of the calling thread (created on first use for the main thread).
    std::shared_ptr<GuestThread> CurrentGuestThread();
    void SetCurrentGuestThread(std::shared_ptr<GuestThread> thread);

    // Resolves a handle, including the current-thread pseudo-handle (0xFFFFFFFE).
    std::shared_ptr<Waitable> WaitableFromHandle(uint32_t handle);

    // Returns the host object behind a guest dispatcher header, creating it from the
    // header's type and SignalState on first use.
    std::shared_ptr<Waitable> WaitableFromGuest(uint32_t address);

    // Waits for one (waitAll = false) or all objects. `timeout` is the guest's
    // LARGE_INTEGER pointer (0 = infinite; negative = relative 100 ns; positive = absolute).
    // Returns STATUS_SUCCESS + index of the satisfied object, or STATUS_TIMEOUT.
    uint32_t WaitForObjects(const std::vector<std::shared_ptr<Waitable>>& objects, bool waitAll, uint32_t timeout);

    // Signals an object as NtSignalAndWait* does. Returns an NTSTATUS.
    uint32_t SignalObject(const std::shared_ptr<Waitable>& object);
}

#ifndef STATUS_TIMEOUT
#define STATUS_TIMEOUT 0x00000102u
#endif
#ifndef STATUS_MUTANT_NOT_OWNED
#define STATUS_MUTANT_NOT_OWNED 0xC0000046u
#endif
#ifndef STATUS_SEMAPHORE_LIMIT_EXCEEDED
#define STATUS_SEMAPHORE_LIMIT_EXCEEDED 0xC0000047u
#endif
