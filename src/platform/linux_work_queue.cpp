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
        if (linux_do_next_work_entry(queue)) sem_wait(&queue->semaphore);
    }
    return 0;
}

// thread_count includes the producer, so thread_count - 1 workers are started,
// pinned to CPUs 2, 4, ..., 14 (one per physical P-core, leaving core 0 to the
// main thread). Pinning is skipped if the machine has fewer CPUs.
internal void linux_make_work_queue(PlatformWorkQueue *queue, u32 thread_count) {
    memset(queue, 0, sizeof(*queue));
    sem_init(&queue->semaphore, 0, 0);
    if (thread_count == 0) thread_count = 1;
    queue->thread_count = thread_count;
    long cpus = sysconf(_SC_NPROCESSORS_ONLN);
    static LinuxWorkerStart starts[64];
    for (u32 i = 0; i + 1 < thread_count && i < ArrayCount(starts); ++i) {
        starts[i].queue = queue;
        s32 cpu = (s32)(2 * (i + 1));
        starts[i].cpu = (cpus >= 16 && cpu < 16) ? cpu : -1;
        pthread_t thread;
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, Megabytes(1));
        pthread_create(&thread, &attr, linux_worker_thread, &starts[i]);
        pthread_detach(thread);
        pthread_attr_destroy(&attr);
    }
}
