#ifndef GPU_API_H
#define GPU_API_H

// C ABI of the optional GPU module (simstack_gpu.{so,dll}, built from
// kernels.cu with `build.sh --cuda`). The platform/sim never include CUDA
// headers; they only see these plain function signatures.
//
// The platform dlopens the module when asked (--gpu), calls gpu_init, and
// hands the sim a GpuAPI table inside SimMemory. The module owns one device
// bump arena; the sim allocates from it once and keeps device pointers in its
// own (host) state, so a sim hot reload keeps every device buffer.
//
// The module is not hot-reloaded: the device arena and the CUDA context live
// as long as the process.

#include "../base/base_types.h"

#if SIM_OS_WINDOWS
#define GPU_API extern "C" __declspec(dllexport)
#else
#define GPU_API extern "C" __attribute__((visibility("default")))
#endif

struct KinMech;
struct KinMethod;
struct KinControl;

// Device pointers for one batched kinetics launch: the DbdZones SoA layout
// (src/sim/dbd.cpp), one lane per zone.
struct GpuKinLaunch {
    f64 *y, *h, *steps, *rejects, *pulses, *failed;            // solver writes
    f64 *k, *src, *pulse, *period, *pulse_first, *pulse_count; // solver reads
    u32 padded;     // array stride (lanes); every lane is integrated
    u32 precision;  // 0: f64 lanes (reference), 1: f32 lanes, delta formulation (see dbd_lanes.h)
    f64 duration;   // master step length; the solver runs in local time [0, duration]
};

struct GpuInfo {
    char name[128];
    u32 sm_major, sm_minor, multiprocessors;
    u64 total_bytes, free_bytes;   // device memory at init
    u64 arena_bytes;
};

// Returns 0 on success, otherwise a cudaError_t value cast to s32.
#define GPU_INIT(name) s32 name(u32 device_index, u64 arena_bytes, GpuInfo *info)
typedef GPU_INIT(gpu_init_fn);

#define GPU_ERROR_STRING(name) const char *name(s32 error)
typedef GPU_ERROR_STRING(gpu_error_string_fn);

// Bump allocation from the module's device arena (256-byte aligned). 0 when full.
#define GPU_ALLOC(name) void *name(u64 size)
typedef GPU_ALLOC(gpu_alloc_fn);

#define GPU_ARENA_USED(name) u64 name(void)
typedef GPU_ARENA_USED(gpu_arena_used_fn);

#define GPU_COPY_TO_DEVICE(name) s32 name(void *dst_device, const void *src_host, u64 size)
typedef GPU_COPY_TO_DEVICE(gpu_copy_to_device_fn);

#define GPU_COPY_TO_HOST(name) s32 name(void *dst_host, const void *src_device, u64 size)
typedef GPU_COPY_TO_HOST(gpu_copy_to_host_fn);

#define GPU_SYNC(name) s32 name(void)
typedef GPU_SYNC(gpu_sync_fn);

// Advances one SoA position/velocity set by dt on the device (M1 demo kernel).
#define GPU_STEP_POINTS(name) s32 name(f32 *x, f32 *y, f32 *z, f32 *vx, f32 *vy, f32 *vz, u32 count, f32 dt)
typedef GPU_STEP_POINTS(gpu_step_points_fn);

// Copies mechanism tables, method and control into constant memory. Call
// when any of them changes (init, method switch, tolerance edit).
#define GPU_KIN_SET_MECH(name) s32 name(const KinMech *mech, const KinMethod *method, const KinControl *control)
typedef GPU_KIN_SET_MECH(gpu_kin_set_mech_fn);

// Launches the batched Rosenbrock advance over [t0, t1] for every lane and
// waits for it. Bit-identical to the CPU path (built with -fmad=false).
#define GPU_KIN_ADVANCE(name) s32 name(const GpuKinLaunch *launch, f32 *kernel_ms)
typedef GPU_KIN_ADVANCE(gpu_kin_advance_fn);

#define GPU_EXPORT_NAMES(X) \
    X(gpu_init) X(gpu_error_string) X(gpu_alloc) X(gpu_arena_used) X(gpu_copy_to_device) X(gpu_copy_to_host) \
    X(gpu_sync) X(gpu_step_points) X(gpu_kin_set_mech) X(gpu_kin_advance)

// The table the platform fills from dlsym and hands to the sim.
struct GpuAPI {
    b32 available;
    GpuInfo info;
#define GPU_API_MEMBER(fn) fn##_fn *fn;
    GPU_EXPORT_NAMES(GPU_API_MEMBER)
#undef GPU_API_MEMBER
};

#endif // GPU_API_H
