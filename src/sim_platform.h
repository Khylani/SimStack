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
    u32 precision;        // dbd / sweep: 0 = f32 lanes (fast, default), 1 = f64 lanes (reference)
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
    u64 ui_size;          // UI: fonts + the size-class heap Dear ImGui/ImPlot allocate from
    void *ui_base;        // (reset wholesale on hot reload; nothing is malloc'd at runtime)

    SimConfig config;
    PlatformAPI platform;
    PlatformWorkQueue *work_queue;   // high priority, P-cores: threads = worker_count (including the caller)
    u32 worker_count;
    PlatformWorkQueue *background_queue;   // one thread: long jobs that overlap the platform's render/swap.
                                           // The platform completes it before any hot reload.
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
    // editing / UI keys
    SimKey_Tab,
    SimKey_Enter,
    SimKey_Backspace,
    SimKey_Delete,
    SimKey_Home,
    SimKey_End,
    SimKey_PageUp,
    SimKey_PageDown,
    SimKey_Ctrl,
    SimKey_Shift,
    SimKey_Alt,
    SimKey_A,
    SimKey_C,
    SimKey_V,
    SimKey_X,
    SimKey_Y,
    SimKey_Z,
    SimKey_F2,
    SimKey_Count,
};

struct SimInput {
    f32 dt_frame;             // wall-clock seconds since the last update (clamped by platform)
    f32 last_work_ms;         // previous frame's CPU work (update + render submit), excluding the swap/vsync wait
    u32 window_width;
    u32 window_height;

    f32 mouse_x, mouse_y;     // pixels, origin top-left
    f32 mouse_wheel;          // notches this frame, + = away from the user
    b32 mouse_inside;
    SimButtonState mouse_buttons[3];
    char text[64];            // UTF-8 typed this frame (NUL-terminated)
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
    const u32 *rgba;          // optional per-point colour (0xAABBGGRR); 0: use `color`
    v4 color;
    f32 point_size;
};

#define RENDER_MAX_BATCHES 32

// UI: indexed triangles in window pixels (origin top-left), one texture per
// command, clipped by a scissor rect. The vertex layout matches ImDrawVert.
struct RenderUiVertex {
    f32 x, y, u, v;
    u32 rgba;
};

struct RenderUiCmd {
    u32 idx_offset, elem_count;   // into ui_idx
    u32 vtx_offset;               // added to every index of this command
    u32 texture;                  // RenderTexOp id
    f32 clip_x0, clip_y0, clip_x1, clip_y1;
};

enum RenderTexOpKind {
    RenderTexOp_Upload,           // create or replace the whole texture (RGBA8)
    RenderTexOp_Destroy,
};

struct RenderTexOp {
    u32 kind, id, width, height;
    const u8 *rgba;               // valid until the end of the frame
};

#define RENDER_MAX_TEX_OPS 16

struct RenderCommands {
    v4 clear_color;
    m4 view_proj;
    f32 scene_x, scene_y, scene_w, scene_h;   // 3D view rect in window pixels (w == 0: whole window)
    u32 batch_count;
    RenderPointBatch batches[RENDER_MAX_BATCHES];

    u32 tex_op_count;
    RenderTexOp tex_ops[RENDER_MAX_TEX_OPS];
    u32 ui_vtx_count, ui_idx_count, ui_cmd_count;
    const RenderUiVertex *ui_vtx;   // in the sim's frame arena
    const u32 *ui_idx;
    const RenderUiCmd *ui_cmds;
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

// Called on the old module just before the platform unloads it for a hot
// reload: the last chance to run code that lives in it (e.g. tear down
// third-party contexts holding function pointers into it). Optional export.
#define SIM_BEFORE_UNLOAD(name) void name(SimMemory *memory)
typedef SIM_BEFORE_UNLOAD(sim_before_unload_fn);

#define SIM_UPDATE_EXPORT_NAME "sim_update"
#define SIM_ON_RELOAD_EXPORT_NAME "sim_on_reload"
#define SIM_BEFORE_UNLOAD_EXPORT_NAME "sim_before_unload"

#endif // SIM_PLATFORM_H
