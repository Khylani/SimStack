#ifndef BASE_DEBUG_H
#define BASE_DEBUG_H

// Cycle-counter profiler. TIMED_BLOCK(name) records rdtsc deltas and hit
// counts into a DebugTable that lives in platform-owned memory (so the sim
// module has no state of its own and the platform can print both sides).
//
// Slot assignment is static: __COUNTER__ within a TU plus a per-TU base, so
// no registration or hashing happens at runtime. Each TU defines
// DEBUG_RECORD_BASE before including this header (platform 0, sim 128).
//
// Everything compiles to nothing when SIM_INTERNAL=0.

#include "base_types.h"

#if SIM_COMPILER_MSVC
#include <intrin.h>
SIM_FORCE_INLINE u64 read_cpu_timer() { return __rdtsc(); }
#else
#include <x86intrin.h>
SIM_FORCE_INLINE u64 read_cpu_timer() { return __rdtsc(); }
#endif

#define DEBUG_MAX_RECORDS 256
#define DEBUG_RECORDS_PER_TU 128

struct DebugRecord {
    const char *name;     // 0 = slot unused
    const char *file;
    u32 line;
    u32 hit_count;        // this frame
    u64 cycles;           // this frame
    u64 total_hits;       // since start
    u64 total_cycles;     // since start
};

struct DebugTable {
    DebugRecord records[DEBUG_MAX_RECORDS];
    u64 frame_index;
};

#if SIM_INTERNAL

#ifndef DEBUG_RECORD_BASE
#define DEBUG_RECORD_BASE 0
#endif

// Set by the platform at startup and by the sim module in every sim_update
// (re-patched after a hot reload, since the DLL's globals are reset).
global DebugTable *g_debug_table;

struct TimedBlock {
    DebugRecord *record;
    u64 start;

    TimedBlock(u32 slot, const char *name, const char *file, u32 line) {
        Assert(slot < DEBUG_MAX_RECORDS);
        record = g_debug_table ? g_debug_table->records + slot : 0;
        if (record) {
            record->name = name;
            record->file = file;
            record->line = line;
        }
        start = read_cpu_timer();
    }
    ~TimedBlock() {
        if (record) {
            u64 dt = read_cpu_timer() - start;
            record->cycles += dt;
            record->hit_count += 1;
            record->total_cycles += dt;
            record->total_hits += 1;
        }
    }
};

#define TIMED_BLOCK_(name, counter) TimedBlock CONCAT(timed_block_, counter)(DEBUG_RECORD_BASE + (counter), name, __FILE__, __LINE__)
#define TIMED_BLOCK(name) TIMED_BLOCK_(name, __COUNTER__)
#define TIMED_FUNCTION() TIMED_BLOCK(__FUNCTION__)

SIM_FORCE_INLINE void debug_table_end_frame(DebugTable *table) {
    for (u32 i = 0; i < DEBUG_MAX_RECORDS; ++i) {
        table->records[i].cycles = 0;
        table->records[i].hit_count = 0;
    }
    ++table->frame_index;
}

#else

#define TIMED_BLOCK(name)
#define TIMED_FUNCTION()
SIM_FORCE_INLINE void debug_table_end_frame(DebugTable *table) { (void)table; }

#endif // SIM_INTERNAL

#endif // BASE_DEBUG_H
