// Work queue in the Handmade Hero style: one producer (the thread that calls
// add_entry, i.e. the sim's update), N workers, a fixed ring of entries, a
// semaphore to wake sleepers, and complete_all_work where the producer
// helps until every entry is done. Workers are pinned one per physical
// P-core on the i9-12900KF (logical CPUs 0-15 are P-core siblings 2k/2k+1,
// 16-23 are E-cores).
//
// Included by linux_simstack.cpp and the test TU. Needs <pthread.h>,
// <semaphore.h>, <sched.h> and sim_platform.h.

struct PlatformWorkQueueEntry {
    platform_work_queue_callback *callback;
    void *data;
};

#define PLATFORM_WORK_QUEUE_SIZE 256

struct PlatformWorkQueue {
    u32 volatile completion_goal;
    u32 volatile completion_count;
    u32 volatile next_entry_to_write;
    u32 volatile next_entry_to_read;
    sem_t semaphore;
    PlatformWorkQueueEntry entries[PLATFORM_WORK_QUEUE_SIZE];
    u32 thread_count;   // workers + the producer
};

internal PLATFORM_ADD_WORK_ENTRY(linux_add_work_entry) {
    u32 write = queue->next_entry_to_write;
    u32 next = (write + 1) % PLATFORM_WORK_QUEUE_SIZE;
    Assert(next != queue->next_entry_to_read);   // ring full: the producer must complete work sooner
    queue->entries[write].callback = callback;
    queue->entries[write].data = data;
    __atomic_add_fetch(&queue->completion_goal, 1, __ATOMIC_SEQ_CST);
    __atomic_store_n(&queue->next_entry_to_write, next, __ATOMIC_RELEASE);   // publish after the entry is written
    sem_post(&queue->semaphore);
}

// Returns 1 if there was nothing to do (caller may sleep).
internal b32 linux_do_next_work_entry(PlatformWorkQueue *queue) {
    u32 read = __atomic_load_n(&queue->next_entry_to_read, __ATOMIC_ACQUIRE);
    u32 next = (read + 1) % PLATFORM_WORK_QUEUE_SIZE;
    if (read == __atomic_load_n(&queue->next_entry_to_write, __ATOMIC_ACQUIRE)) return 1;
    if (__atomic_compare_exchange_n(&queue->next_entry_to_read, &read, next, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        PlatformWorkQueueEntry entry = queue->entries[read];
        entry.callback(queue, entry.data);
        __atomic_add_fetch(&queue->completion_count, 1, __ATOMIC_SEQ_CST);
    }
    return 0;
}

internal PLATFORM_COMPLETE_ALL_WORK(linux_complete_all_work) {
    while (__atomic_load_n(&queue->completion_goal, __ATOMIC_ACQUIRE) !=
           __atomic_load_n(&queue->completion_count, __ATOMIC_ACQUIRE)) {
        linux_do_next_work_entry(queue);
    }
    queue->completion_goal = 0;
    queue->completion_count = 0;
}

struct LinuxWorkerStart {
    PlatformWorkQueue *queue;
    s32 cpu;   // -1: not pinned
};

internal void *linux_worker_thread(void *param) {
    LinuxWorkerStart *start = (LinuxWorkerStart *)param;
    PlatformWorkQueue *queue = start->queue;
    if (start->cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(start->cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
    for (;;) {
        if (!linux_do_next_work_entry(queue)) continue;
        // Nothing to do: spin ~50 us before sleeping, so back-to-back batches
        // (one per coupling window / master step) don't pay the OS wake latency.
        u64 t0 = __rdtsc();
        b32 found = 0;
        while (__rdtsc() - t0 < 150000) {   // ~50 us at ~3 GHz TSC
            if (__atomic_load_n(&queue->next_entry_to_read, __ATOMIC_ACQUIRE) !=
                __atomic_load_n(&queue->next_entry_to_write, __ATOMIC_ACQUIRE)) { found = 1; break; }
            _mm_pause();
        }
        if (!found) sem_wait(&queue->semaphore);
    }
    return 0;
}

// Starts `workers` threads for the queue; worker i is pinned to logical CPU
// first_cpu + i * cpu_stride (skipped when that is not a P-core thread, i.e.
// >= 16, or the machine is smaller). The high-priority queue uses CPUs
// 2, 4, ..., 14 (one per physical P-core, core 0 left to the main thread);
// the background queue's single thread sits on CPU 1, core 0's sibling.
internal void linux_make_work_queue(PlatformWorkQueue *queue, u32 workers, s32 first_cpu, s32 cpu_stride) {
    memset(queue, 0, sizeof(*queue));
    sem_init(&queue->semaphore, 0, 0);
    queue->thread_count = workers + 1;
    long cpus = sysconf(_SC_NPROCESSORS_ONLN);
    static LinuxWorkerStart starts[96];
    static u32 used;
    for (u32 i = 0; i < workers && used < ArrayCount(starts); ++i) {
        LinuxWorkerStart *start = starts + used++;
        start->queue = queue;
        s32 cpu = first_cpu + (s32)i * cpu_stride;
        start->cpu = (cpus >= 16 && cpu < 16) ? cpu : -1;
        pthread_t thread;
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, Megabytes(4));
        pthread_create(&thread, &attr, linux_worker_thread, start);
        pthread_detach(thread);
        pthread_attr_destroy(&attr);
    }
}
