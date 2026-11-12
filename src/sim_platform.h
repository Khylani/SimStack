#ifndef SIM_PLATFORM_H
#define SIM_PLATFORM_H

// The one header shared by the platform executable and the sim module.
// It defines the memory block handed to the sim, the input it receives, the
// render commands it produces, the services the platform offers, and the
// exported entry points.
//
// Rules: the sim never calls the OS. All sim state lives inside SimMemory's
// blocks, so the platform can unload and reload the sim module at any time
// and the state is preserved. The sim module must hold no mutable globals
// other than pointers re-patched in sim_on_reload / every update.

#include "base/base_types.h"
#include "base/base_arena.h"
#include "base/base_math.h"
#include "base/base_simd.h"
#include "base/base_debug.h"
#include "gpu/gpu_api.h"

// ----------------------------------------------------------------------------
// Platform services exposed to the sim (function pointer table).

struct PlatformFileContents {
    void *data;    // pushed onto the arena passed to read_entire_file
    u64 size;
};

#define PLATFORM_READ_ENTIRE_FILE(name) PlatformFileContents name(Arena *arena, const char *path)
typedef PLATFORM_READ_ENTIRE_FILE(platform_read_entire_file_fn);

#define PLATFORM_WRITE_ENTIRE_FILE(name) b32 name(const char *path, const void *data, u64 size)
typedef PLATFORM_WRITE_ENTIRE_FILE(platform_write_entire_file_fn);

#define PLATFORM_NOW_NS(name) u64 name(void)
typedef PLATFORM_NOW_NS(platform_now_ns_fn);

#define PLATFORM_LOG(name) void name(const char *fmt, ...)
typedef PLATFORM_LOG(platform_log_fn);

// Work queue (Handmade Hero style): the sim pushes (callback, data) entries
// and then calls complete_all_work, which helps until every entry has run.
// Entries must be complete before sim_update returns, so no callback pointer
// into the sim module survives a hot reload.
struct PlatformWorkQueue;
#define PLATFORM_WORK_QUEUE_CALLBACK(name) void name(PlatformWorkQueue *queue, void *data)
typedef PLATFORM_WORK_QUEUE_CALLBACK(platform_work_queue_callback);

#define PLATFORM_ADD_WORK_ENTRY(name) void name(PlatformWorkQueue *queue, platform_work_queue_callback *callback, void *data)
typedef PLATFORM_ADD_WORK_ENTRY(platform_add_work_entry_fn);

#define PLATFORM_COMPLETE_ALL_WORK(name) void name(PlatformWorkQueue *queue)
typedef PLATFORM_COMPLETE_ALL_WORK(platform_complete_all_work_fn);

struct PlatformAPI {
    platform_read_entire_file_fn *read_entire_file;
    platform_write_entire_file_fn *write_entire_file;
    platform_now_ns_fn *now_ns;
    platform_log_fn *log;
    platform_add_work_entry_fn *add_work_entry;
    platform_complete_all_work_fn *complete_all_work;
};

// ----------------------------------------------------------------------------
// Memory handed to the sim. Fixed sizes chosen by the platform at startup;
// the sim carves its arenas out of these on its first update.

enum SimScene {
    SimScene_Particles,   // M1 demo: SoA particle swirl
    SimScene_Dbd,         // M2: DBD reactor zones (plug flow), plasma chemistry
    SimScene_Sweep,       // M2: independent zones over an operating-point grid
};

enum SimBackend {
    SimBackend_Cpu,       // AVX2 lanes across the work queue
    SimBackend_Gpu,       // CUDA, one thread per zone (bit-identical to the CPU)
};

struct SimConfig {
    u32 scene;            // SimScene
    u32 particle_count;   // particles scene
    u32 seed;
    u32 zone_count;       // dbd / sweep scenes
    u32 backend;          // SimBackend
    b32 averaged;         // dbd / sweep: continuous source instead of discharge pulses
    const char *csv_path; // dbd / sweep: write results here (0: none)
};

struct SimMemory {
    b32 is_initialized;

    u64 permanent_size;   // lives for the whole process
    void *permanent_base;
    u64 frame_size;       // cleared at the start of every sim_update
    void *frame_base;
    u64 scratch_size;     // begin_temp/end_temp scopes only
    void *scratch_base;

    SimConfig config;
    PlatformAPI platform;
    PlatformWorkQueue *work_queue;   // threads = worker_count (including the caller)
    u32 worker_count;
    GpuAPI gpu;                      // gpu.available == 0 without --gpu or without the module
    DebugTable *debug;    // platform-owned profiler table (may be 0 when SIM_INTERNAL=0)
};

// ----------------------------------------------------------------------------
// Input. Buttons record transitions so a press and release inside a single
// frame is not lost.

struct SimButtonState {
    b32 ended_down;
    u32 half_transition_count;
};

SIM_FORCE_INLINE b32 was_pressed(SimButtonState b) {
    return (b.half_transition_count > 1) || (b.half_transition_count == 1 && b.ended_down);
}
SIM_FORCE_INLINE b32 is_down(SimButtonState b) { return b.ended_down; }

enum SimKey {
    SimKey_Escape,
    SimKey_Space,
    SimKey_R,
    SimKey_P,
    SimKey_F1,
    SimKey_Left,
    SimKey_Right,
    SimKey_Up,
    SimKey_Down,
    SimKey_Count,
};

struct SimInput {
    f32 dt_frame;             // wall-clock seconds since the last update (clamped by platform)
    u32 window_width;
    u32 window_height;

    f32 mouse_x, mouse_y;     // pixels, origin top-left
    SimButtonState mouse_buttons[3];
    SimButtonState keys[SimKey_Count];

    b32 headless;             // no renderer: platform drives with a fixed dt
    b32 compute_state_hash;   // ask the sim to fill SimStats::state_hash
};

// ----------------------------------------------------------------------------
// Output: a few batches of SoA points, each with its own colour and size.
// The platform uploads each batch's three arrays straight into vertex buffers.

struct RenderPointBatch {
    u32 count;                // valid entries (padded count may be larger)
    const f32 *px, *py, *pz;  // SoA positions in sim memory (valid until next update)
    v4 color;
    f32 point_size;
};

#define RENDER_MAX_BATCHES 32

struct RenderCommands {
    v4 clear_color;
    m4 view_proj;
    u32 batch_count;
    RenderPointBatch batches[RENDER_MAX_BATCHES];
};

struct SimStats {
    u64 step_count;           // fixed steps since start (survives hot reload)
    u32 steps_this_frame;
    f64 sim_time;
    f32 dt_fixed;
    u32 particle_count;
    u64 state_hash;           // only when input->compute_state_hash
    u64 permanent_used, permanent_size;
    u64 frame_used, frame_size;
    u64 scratch_high_water, scratch_size;
    b32 paused;
    u32 reload_count;         // incremented by sim_on_reload
    u32 thread_count;         // threads the sim used for its batches
    char summary[512];        // scene-specific one-liner for headless / periodic prints
};

// ----------------------------------------------------------------------------
// Exported entry points of the sim module (C ABI).

#if SIM_OS_WINDOWS
#define SIM_API extern "C" __declspec(dllexport)
#else
#define SIM_API extern "C" __attribute__((visibility("default")))
#endif

#define SIM_UPDATE(name) void name(SimMemory *memory, SimInput *input, RenderCommands *render, SimStats *stats)
typedef SIM_UPDATE(sim_update_fn);

#define SIM_ON_RELOAD(name) void name(SimMemory *memory)
typedef SIM_ON_RELOAD(sim_on_reload_fn);

#define SIM_UPDATE_EXPORT_NAME "sim_update"
#define SIM_ON_RELOAD_EXPORT_NAME "sim_on_reload"

#endif // SIM_PLATFORM_H
