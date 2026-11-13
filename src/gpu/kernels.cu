// Simstack GPU module: the single nvcc translation unit.
//   nvcc -arch=sm_86 -O3 -fmad=false -lineinfo --shared -Xcompiler -fPIC -Isrc src/gpu/kernels.cu -o build/simstack_gpu.so
//
// -fmad=false is required: the kinetics kernel must perform exactly the
// operations the CPU path performs (fused only where the source says
// lane_fmadd / lane_fnmadd) so CPU and GPU results are bit-identical.
//
// Note: base_types.h does not define the `global` macro under __CUDACC__
// (it would rewrite __global__ into __attribute__((static))); use `static`.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu_api.h"
#include "../sim/dbd_lanes.h"
#include "air_mech_gen.h"

struct GpuArena {
    u8 *base;
    u64 size;
    u64 used;
};

static GpuArena g_gpu_arena;
static cudaEvent_t g_kin_start, g_kin_stop, g_async_begin, g_async_end;
static cudaStream_t g_stream;
static u32 g_kin_block = 64;
static b32 g_kin_generated;   // the uploaded mechanism matches AirGenOps

// Per-thread capacities for the GPU instantiation. Every lane's Jacobian/LU,
// stage vectors and rate constants live in local memory, about
// 8 * (nnz + 9 * species + reactions) bytes per lane. The humid-air
// mechanism is 17 species / 40 reactions / 213 non-zeros (~4 KB per lane).
typedef KinTableOps<24, 48, 256> KinOpsGpu;

// Mechanism tables: read in the same order by every thread (uniform), so
// constant-memory broadcasts. Staging them in shared memory instead was
// measured slower (30 KB per block costs occupancy), see docs/M2_KINETICS.md.
__constant__ KinMech c_mech;
__constant__ KinMethod c_method;
__constant__ KinControl c_control;

// ----------------------------------------------------------------------------
// Kernels

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

// One thread per lane (zone): the same dbd_advance_group the CPU runs, with V = f64.
template <class V, class Ops, bool Delta>
__global__ void __launch_bounds__(64) kin_advance_kernel(GpuKinLaunch L) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (L.lane_count ? L.lane_count : L.padded)) return;
    DbdLaneArrays a;
    a.y = L.y; a.h = L.h; a.steps = L.steps; a.rejects = L.rejects; a.pulses = L.pulses; a.failed = L.failed;
    a.k = L.k; a.src = L.src; a.pulse = L.pulse; a.period = L.period;
    a.pulse_first = L.pulse_first; a.pulse_count = L.pulse_count;
    a.padded = L.padded;
    dbd_advance_group<V, Ops, Delta>(&c_mech, &c_method, &c_control, a, i, L.duration);
}

// ----------------------------------------------------------------------------
// Exports

GPU_API GPU_INIT(gpu_init) {
    memset(info, 0, sizeof(*info));
    cudaError_t err = cudaSetDevice((int)device_index);
    if (err != cudaSuccess) return (s32)err;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, (int)device_index);
    if (err != cudaSuccess) return (s32)err;
    snprintf(info->name, sizeof(info->name), "%s", prop.name);
    info->sm_major = (u32)prop.major;
    info->sm_minor = (u32)prop.minor;
    info->multiprocessors = (u32)prop.multiProcessorCount;
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    info->free_bytes = free_b;
    info->total_bytes = total_b;

    void *base = 0;
    err = cudaMalloc(&base, arena_bytes);
    if (err != cudaSuccess) return (s32)err;
    g_gpu_arena.base = (u8 *)base;
    g_gpu_arena.size = arena_bytes;
    g_gpu_arena.used = 0;
    info->arena_bytes = arena_bytes;

    cudaEventCreate(&g_kin_start);
    cudaEventCreate(&g_kin_stop);
    cudaEventCreate(&g_async_begin);
    cudaEventCreate(&g_async_end);
    cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking);
    const char *block = getenv("SIMSTACK_KIN_BLOCK");   // experiments only
    if (block && atoi(block) > 0 && atoi(block) <= 64) g_kin_block = (u32)atoi(block);
    return 0;
}

GPU_API GPU_ERROR_STRING(gpu_error_string) {
    return cudaGetErrorString((cudaError_t)error);
}

GPU_API GPU_ALLOC(gpu_alloc) {
    u64 aligned = AlignPow2(size, 256);
    if (g_gpu_arena.used + aligned > g_gpu_arena.size) return 0;
    void *result = g_gpu_arena.base + g_gpu_arena.used;
    g_gpu_arena.used += aligned;
    return result;
}

GPU_API GPU_ARENA_USED(gpu_arena_used) {
    return g_gpu_arena.used;
}

GPU_API GPU_ARENA_RESET(gpu_arena_reset) {
    cudaDeviceSynchronize();
    g_gpu_arena.used = 0;
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

GPU_API GPU_KIN_SET_MECH(gpu_kin_set_mech) {
    if (!kin_capacity_ok<KinOpsGpu>(mech)) return (s32)cudaErrorInvalidValue;   // raise KinOpsGpu and rebuild
    g_kin_generated = hash_fnv1a_64(mech, sizeof(KinMech)) == AirGenOps::signature;
    cudaError_t err = cudaMemcpyToSymbol(c_mech, mech, sizeof(KinMech));
    if (err == cudaSuccess) err = cudaMemcpyToSymbol(c_method, method, sizeof(KinMethod));
    if (err == cudaSuccess) err = cudaMemcpyToSymbol(c_control, control, sizeof(KinControl));
    return (s32)err;
}

internal void gpu_kin_launch(const GpuKinLaunch *launch, cudaStream_t stream) {
    u32 block = g_kin_block;
    u32 lanes = launch->lane_count ? launch->lane_count : launch->padded;
    u32 grid = (lanes + block - 1) / block;
    if (launch->precision == 1) {
        if (g_kin_generated) kin_advance_kernel<f32, AirGenOps, true><<<grid, block, 0, stream>>>(*launch);
        else kin_advance_kernel<f32, KinOpsGpu, true><<<grid, block, 0, stream>>>(*launch);
    } else {
        if (g_kin_generated) kin_advance_kernel<f64, AirGenOps, false><<<grid, block, 0, stream>>>(*launch);
        else kin_advance_kernel<f64, KinOpsGpu, false><<<grid, block, 0, stream>>>(*launch);
    }
}

GPU_API GPU_HOST_REGISTER(gpu_host_register) {
    cudaError_t err = cudaHostRegister(host, size, cudaHostRegisterDefault);
    if (err == cudaErrorHostMemoryAlreadyRegistered) { cudaGetLastError(); return 0; }
    return (s32)err;
}

GPU_API GPU_HOST_UNREGISTER(gpu_host_unregister) {
    cudaStreamSynchronize(g_stream);
    return (s32)cudaHostUnregister(host);
}

GPU_API GPU_KIN_ADVANCE_ASYNC(gpu_kin_advance_async) {
    u64 pitch = (u64)launch->padded * sizeof(f64);
    u64 width = (u64)(launch->lane_count ? launch->lane_count : launch->padded) * sizeof(f64);
    cudaEventRecord(g_async_begin, g_stream);
    cudaError_t err = cudaMemcpy2DAsync(device_block, pitch, host_block, pitch, width, rows, cudaMemcpyHostToDevice, g_stream);
    if (err != cudaSuccess) return (s32)err;
    cudaEventRecord(g_kin_start, g_stream);
    gpu_kin_launch(launch, g_stream);
    cudaEventRecord(g_kin_stop, g_stream);
    err = cudaMemcpy2DAsync(host_block, pitch, device_block, pitch, width, mutable_rows, cudaMemcpyDeviceToHost, g_stream);
    if (err != cudaSuccess) return (s32)err;
    cudaEventRecord(g_async_end, g_stream);
    return (s32)cudaGetLastError();
}

GPU_API GPU_KIN_WAIT(gpu_kin_wait) {
    cudaError_t err = cudaEventSynchronize(g_async_end);
    if (err != cudaSuccess) return (s32)err;
    if (kernel_ms) cudaEventElapsedTime(kernel_ms, g_kin_start, g_kin_stop);
    if (total_ms) cudaEventElapsedTime(total_ms, g_async_begin, g_async_end);
    return (s32)cudaGetLastError();
}

GPU_API GPU_KIN_ADVANCE(gpu_kin_advance) {
    u32 block = g_kin_block;
    u32 grid = ((launch->lane_count ? launch->lane_count : launch->padded) + block - 1) / block;
    cudaEventRecord(g_kin_start);
    if (launch->precision == 1) {
        if (g_kin_generated) kin_advance_kernel<f32, AirGenOps, true><<<grid, block>>>(*launch);
        else kin_advance_kernel<f32, KinOpsGpu, true><<<grid, block>>>(*launch);
    } else {
        if (g_kin_generated) kin_advance_kernel<f64, AirGenOps, false><<<grid, block>>>(*launch);
        else kin_advance_kernel<f64, KinOpsGpu, false><<<grid, block>>>(*launch);
    }
    cudaEventRecord(g_kin_stop);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return (s32)err;
    err = cudaEventSynchronize(g_kin_stop);
    if (err != cudaSuccess) return (s32)err;
    if (kernel_ms) cudaEventElapsedTime(kernel_ms, g_kin_start, g_kin_stop);
    return (s32)cudaGetLastError();
}
