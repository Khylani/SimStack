#ifndef GPU_API_H
#define GPU_API_H

// C ABI of the optional GPU module (simstack_gpu.{so,dll}, built from
// kernels.cu with `build.sh --cuda`). The platform/sim never include CUDA
// headers; they only see these plain function signatures.
//
// STATUS (milestone 1): the module builds standalone when the CUDA toolkit is
// present but is not loaded by the platform yet. GPU arenas, GL interop and
// the point-animation kernel from the architecture plan land with the first
// real GPU workload.

#include "../base/base_types.h"

#if SIM_OS_WINDOWS
#define GPU_API extern "C" __declspec(dllexport)
#else
#define GPU_API extern "C" __attribute__((visibility("default")))
#endif

// Returns 0 on success, otherwise a cudaError_t value cast to s32.
#define GPU_INIT(name) s32 name(u32 device_index)
typedef GPU_INIT(gpu_init_fn);

// Advances one SoA position/velocity set by dt on the device. Pointers are
// device pointers obtained from the module's own arena (gpu_alloc).
#define GPU_STEP_POINTS(name) s32 name(f32 *x, f32 *y, f32 *z, f32 *vx, f32 *vy, f32 *vz, u32 count, f32 dt)
typedef GPU_STEP_POINTS(gpu_step_points_fn);

#define GPU_ALLOC(name) void *name(u64 size)
typedef GPU_ALLOC(gpu_alloc_fn);

#define GPU_COPY_TO_DEVICE(name) s32 name(void *dst_device, const void *src_host, u64 size)
typedef GPU_COPY_TO_DEVICE(gpu_copy_to_device_fn);

#define GPU_COPY_TO_HOST(name) s32 name(void *dst_host, const void *src_device, u64 size)
typedef GPU_COPY_TO_HOST(gpu_copy_to_host_fn);

#define GPU_SYNC(name) s32 name(void)
typedef GPU_SYNC(gpu_sync_fn);

#endif // GPU_API_H
