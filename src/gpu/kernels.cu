// Simstack GPU module: the single nvcc translation unit.
//   nvcc -arch=sm_86 -O3 -lineinfo --shared -Xcompiler -fPIC -Isrc src/gpu/kernels.cu -o build/simstack_gpu.so
//
// STATUS (milestone 1): compiles with CUDA 12.x / 13.x (13.4 confirmed on an
// RTX 3060 12 GB, sm_86); not yet loaded by the platform. The bump arena
// mirrors base_arena.h so host and device layouts stay in sync.
//
// Note: base_types.h does not define the `global` macro under __CUDACC__
// (it would rewrite __global__ into __attribute__((static))); use `static`.

#include <cuda_runtime.h>
#include "gpu_api.h"

#define GPU_ARENA_SIZE Gigabytes(1)   // placeholder; the plan targets most of the 12 GB on the RTX 3060

struct GpuArena {
    u8 *base;
    u64 size;
    u64 used;
};

static GpuArena g_gpu_arena;

__global__ void step_points_kernel(f32 *x, f32 *y, f32 *z, f32 *vx, f32 *vy, f32 *vz, u32 count, f32 dt) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const f32 spring = 4.0f, swirl = 0.6f, damp = 1.0f - 0.02f * dt;
    f32 px = x[i], py = y[i], pz = z[i];
    f32 ax = -spring * px + swirl * pz;
    f32 ay = -spring * py;
    f32 az = -spring * pz - swirl * px;
    f32 nvx = fmaf(ax, dt, vx[i]) * damp;
    f32 nvy = fmaf(ay, dt, vy[i]) * damp;
    f32 nvz = fmaf(az, dt, vz[i]) * damp;
    vx[i] = nvx; vy[i] = nvy; vz[i] = nvz;
    x[i] = fmaf(nvx, dt, px);
    y[i] = fmaf(nvy, dt, py);
    z[i] = fmaf(nvz, dt, pz);
}

GPU_API GPU_INIT(gpu_init) {
    cudaError_t err = cudaSetDevice((int)device_index);
    if (err != cudaSuccess) return (s32)err;
    void *base = 0;
    err = cudaMalloc(&base, GPU_ARENA_SIZE);
    if (err != cudaSuccess) return (s32)err;
    g_gpu_arena.base = (u8 *)base;
    g_gpu_arena.size = GPU_ARENA_SIZE;
    g_gpu_arena.used = 0;
    return 0;
}

GPU_API GPU_ALLOC(gpu_alloc) {
    u64 aligned = AlignPow2(size, 256);
    if (g_gpu_arena.used + aligned > g_gpu_arena.size) return 0;
    void *result = g_gpu_arena.base + g_gpu_arena.used;
    g_gpu_arena.used += aligned;
    return result;
}

GPU_API GPU_COPY_TO_DEVICE(gpu_copy_to_device) {
    return (s32)cudaMemcpy(dst_device, src_host, size, cudaMemcpyHostToDevice);
}

GPU_API GPU_COPY_TO_HOST(gpu_copy_to_host) {
    return (s32)cudaMemcpy(dst_host, src_device, size, cudaMemcpyDeviceToHost);
}

GPU_API GPU_STEP_POINTS(gpu_step_points) {
    u32 block = 256;
    u32 grid = (count + block - 1) / block;
    step_points_kernel<<<grid, block>>>(x, y, z, vx, vy, vz, count, dt);
    return (s32)cudaGetLastError();
}

GPU_API GPU_SYNC(gpu_sync) {
    return (s32)cudaDeviceSynchronize();
}
