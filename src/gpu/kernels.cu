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
// Warp-per-zone policy: 32 threads cooperate on one zone. Every thread runs
// the integrator's control flow on identical copies of the scalars (uniform,
// no divergence); only fun / -J / LU / solve are split across the warp, with
// per-warp scratch in shared memory. Each element is produced by the same
// operation sequence as the CPU table walk (kin_solver.h), so results stay
// bit-identical:
//   fun      rates in parallel, then each species accumulates its own terms in
//            reaction order
//   -J       derivatives d(r,p) in parallel, then each non-zero accumulates its
//            terms in (r, p) order
//   LU       eliminations in order, each elimination's updates across the warp
//   solve    column-oriented: forward ascending k, backward descending k (the
//            CPU walks each row's upper entries in descending column order to match)

#define WARP_MAX_N   24
#define WARP_MAX_R   48
#define WARP_MAX_NNZ 256

struct KinWarpTables {
    u16 fun_begin[WARP_MAX_N + 1];
    u8  fun_r[WARP_MAX_R * KIN_MAX_NET];
    s8  fun_coef[WARP_MAX_R * KIN_MAX_NET];
    u16 jac_begin[WARP_MAX_NNZ + 1];
    u8  jac_rp[WARP_MAX_R * KIN_MAX_REACTANTS * KIN_MAX_NET];      // r * 3 + p
    s8  jac_coef[WARP_MAX_R * KIN_MAX_REACTANTS * KIN_MAX_NET];
    u16 lcol_begin[WARP_MAX_N + 1], ucol_begin[WARP_MAX_N + 1];
    u8  lcol_row[WARP_MAX_NNZ], ucol_row[WARP_MAX_NNZ];
    u16 lcol_idx[WARP_MAX_NNZ], ucol_idx[WARP_MAX_NNZ];
};
__constant__ KinWarpTables c_warp;
static b32 g_warp_ok;           // the uploaded mechanism fits the warp tables
static u32 g_kin_mode;          // 0: thread per zone, 1: warp per zone

// Host: lists in exactly the CPU's accumulation order.
static b32 kin_build_warp_tables(const KinMech *m, KinWarpTables *t) {
    memset(t, 0, sizeof(*t));
    u32 n = m->species_count, R = m->reaction_count, nnz = m->nnz;
    if (n > WARP_MAX_N || R > WARP_MAX_R || nnz > WARP_MAX_NNZ) return 0;
    u32 c = 0;
    for (u32 sp = 0; sp < n; ++sp) {
        t->fun_begin[sp] = (u16)c;
        for (u32 r = 0; r < R; ++r)
            for (u32 e = 0; e < m->reactions[r].net_count; ++e)
                if (m->reactions[r].net_species[e] == sp) { t->fun_r[c] = (u8)r; t->fun_coef[c] = m->reactions[r].net_coef[e]; ++c; }
    }
    t->fun_begin[n] = (u16)c;
    c = 0;
    for (u32 idx = 0; idx < nnz; ++idx) {
        t->jac_begin[idx] = (u16)c;
        for (u32 r = 0; r < R; ++r)
            for (u32 p = 0; p < m->reactions[r].reactant_count; ++p)
                for (u32 e = 0; e < m->reactions[r].net_count; ++e)
                    if (m->reactions[r].jac_index[p][e] == idx) {
                        t->jac_rp[c] = (u8)(r * 3 + p); t->jac_coef[c] = (s8)-m->reactions[r].net_coef[e]; ++c;
                    }
    }
    t->jac_begin[nnz] = (u16)c;
    u32 lc = 0, uc = 0;
    for (u32 k = 0; k < n; ++k) {
        t->lcol_begin[k] = (u16)lc;
        t->ucol_begin[k] = (u16)uc;
        for (u32 i = 0; i < n; ++i) {
            for (u32 e = m->lower_begin[i]; e < m->lower_begin[i + 1]; ++e)
                if (m->lower[e].col == k) { t->lcol_row[lc] = (u8)i; t->lcol_idx[lc] = m->lower[e].index; ++lc; }
            for (u32 e = m->upper_begin[i]; e < m->upper_begin[i + 1]; ++e)
                if (m->upper[e].col == k) { t->ucol_row[uc] = (u8)i; t->ucol_idx[uc] = m->upper[e].index; ++uc; }
        }
    }
    t->lcol_begin[n] = (u16)lc;
    t->ucol_begin[n] = (u16)uc;
    return 1;
}

template <class V>
struct KinWarpScratch {
    V rate[WARP_MAX_R];
    V deriv[WARP_MAX_R * 3];
    V vec[WARP_MAX_N];
    V lu[WARP_MAX_NNZ];
};

template <class V>
struct KinWarpOps {
    enum { species = WARP_MAX_N, reactions = WARP_MAX_R, nnz = 1 };   // the LU lives in shared memory
    static __device__ __forceinline__ KinWarpScratch<V> *scratch() {
        extern __shared__ __align__(16) unsigned char kin_smem[];
        return (KinWarpScratch<V> *)kin_smem + (threadIdx.x >> 5);
    }
    static __device__ __forceinline__ u32 lane() { return threadIdx.x & 31u; }
    static SIM_HD u32 n(const KinMech *m) { return m->species_count; }
    static __device__ __forceinline__ bool writer() { return lane() == 0; }

    static __device__ void fun(const KinMech *m, const V *k, const V *src, const V *y, V *f) {
        KinWarpScratch<V> *S = scratch();
        u32 R = m->reaction_count, nsp = m->species_count;
        for (u32 r = lane(); r < R; r += 32) {
            const KinReaction *rx = m->reactions + r;
            V rate = k[r];
            for (u32 p = 0; p < rx->reactant_count; ++p) rate = rate * y[rx->reactant[p]];
            S->rate[r] = rate;
        }
        __syncwarp();
        for (u32 sp = lane(); sp < nsp; sp += 32) {
            V v = src[sp];
            for (u32 e = c_warp.fun_begin[sp]; e < c_warp.fun_begin[sp + 1]; ++e)
                v = lane_fmadd((V)c_warp.fun_coef[e], S->rate[c_warp.fun_r[e]], v);
            S->vec[sp] = v;
        }
        __syncwarp();
        for (u32 sp = 0; sp < nsp; ++sp) f[sp] = S->vec[sp];
        __syncwarp();
    }

    static __device__ void neg_jac(const KinMech *m, const V *k, const V *y, V *) {
        KinWarpScratch<V> *S = scratch();
        u32 R = m->reaction_count;
        for (u32 item = lane(); item < R * 3; item += 32) {
            u32 r = item / 3, p = item % 3;
            const KinReaction *rx = m->reactions + r;
            if (p >= rx->reactant_count) continue;
            V d = k[r];
            for (u32 q = 0; q < rx->reactant_count; ++q)
                if (q != p) d = d * y[rx->reactant[q]];
            S->deriv[item] = d;
        }
        __syncwarp();
        for (u32 idx = lane(); idx < m->nnz; idx += 32) {
            V v = (V)0;
            for (u32 e = c_warp.jac_begin[idx]; e < c_warp.jac_begin[idx + 1]; ++e)
                v = lane_fmadd((V)c_warp.jac_coef[e], S->deriv[c_warp.jac_rp[e]], v);
            S->lu[idx] = v;
        }
        __syncwarp();
    }

    static __device__ void factor(const KinMech *m, V ghinv, V *) {
        KinWarpScratch<V> *S = scratch();
        for (u32 i = 0; i < m->species_count; ++i) {
            for (u32 d = m->div_begin[i]; d < m->div_begin[i + 1]; ++d) {
                KinLuDiv div = m->lu_div[d];
                if (lane() == 0) S->lu[div.ik] = S->lu[div.ik] * S->lu[div.kk];
                __syncwarp();
                V l = S->lu[div.ik];
                for (u32 sb = div.sub_begin + lane(); sb < div.sub_end; sb += 32) {
                    KinLuSub sub = m->lu_sub[sb];
                    S->lu[sub.ij] = lane_fnmadd(l, S->lu[sub.kj], S->lu[sub.ij]);
                }
                __syncwarp();
            }
            if (lane() == 0) { u32 ii = m->diag[i]; S->lu[ii] = (V)1 / (S->lu[ii] + ghinv); }
            __syncwarp();
        }
    }

    static __device__ void solve(const KinMech *m, const V *, V *x) {
        KinWarpScratch<V> *S = scratch();
        u32 nsp = m->species_count;
        for (u32 sp = lane(); sp < nsp; sp += 32) S->vec[sp] = x[sp];
        __syncwarp();
        for (u32 kc = 0; kc < nsp; ++kc) {
            V xk = S->vec[kc];
            for (u32 e = c_warp.lcol_begin[kc] + lane(); e < c_warp.lcol_begin[kc + 1]; e += 32) {
                u32 row = c_warp.lcol_row[e];
                S->vec[row] = lane_fnmadd(S->lu[c_warp.lcol_idx[e]], xk, S->vec[row]);
            }
            __syncwarp();
        }
        for (u32 kc = nsp; kc-- > 0;) {
            if (lane() == 0) S->vec[kc] = S->vec[kc] * S->lu[m->diag[kc]];
            __syncwarp();
            V xk = S->vec[kc];
            for (u32 e = c_warp.ucol_begin[kc] + lane(); e < c_warp.ucol_begin[kc + 1]; e += 32) {
                u32 row = c_warp.ucol_row[e];
                S->vec[row] = lane_fnmadd(S->lu[c_warp.ucol_idx[e]], xk, S->vec[row]);
            }
            __syncwarp();
        }
        for (u32 sp = 0; sp < nsp; ++sp) x[sp] = S->vec[sp];
        __syncwarp();
    }
};

#define KIN_WARPS_PER_BLOCK 4

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

// One warp per zone (KinWarpOps): same dbd_advance_group, cooperative kernels.
template <class V, bool Delta>
__global__ void __launch_bounds__(32 * KIN_WARPS_PER_BLOCK) kin_advance_warp_kernel(GpuKinLaunch L) {
    u32 zone = blockIdx.x * KIN_WARPS_PER_BLOCK + (threadIdx.x >> 5);
    if (zone >= (L.lane_count ? L.lane_count : L.padded)) return;   // whole warps exit together
    DbdLaneArrays a;
    a.y = L.y; a.h = L.h; a.steps = L.steps; a.rejects = L.rejects; a.pulses = L.pulses; a.failed = L.failed;
    a.k = L.k; a.src = L.src; a.pulse = L.pulse; a.period = L.period;
    a.pulse_first = L.pulse_first; a.pulse_count = L.pulse_count;
    a.padded = L.padded;
    dbd_advance_group<V, KinWarpOps<V>, Delta>(&c_mech, &c_method, &c_control, a, zone, L.duration);
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

// Every copy and kernel goes on the module's one stream (non-blocking w.r.t. the
// legacy default stream), and synchronous copies wait for that stream. A plain
// cudaMemcpy from pageable memory returns once the data is *staged*, before the
// DMA lands, and does not order against a non-blocking stream: a kernel on
// g_stream could then read partly-old inputs (it did: run-to-run differences).
GPU_API GPU_COPY_TO_DEVICE(gpu_copy_to_device) {
    cudaError_t err = cudaMemcpyAsync(dst_device, src_host, size, cudaMemcpyHostToDevice, g_stream);
    if (err != cudaSuccess) return (s32)err;
    return (s32)cudaStreamSynchronize(g_stream);
}

GPU_API GPU_COPY_TO_HOST(gpu_copy_to_host) {
    cudaError_t err = cudaMemcpyAsync(dst_host, src_device, size, cudaMemcpyDeviceToHost, g_stream);
    if (err != cudaSuccess) return (s32)err;
    return (s32)cudaStreamSynchronize(g_stream);
}

GPU_API GPU_STEP_POINTS(gpu_step_points) {
    u32 block = 256;
    u32 grid = (count + block - 1) / block;
    step_points_kernel<<<grid, block, 0, g_stream>>>(x, y, z, vx, vy, vz, count, dt);
    return (s32)cudaGetLastError();
}

GPU_API GPU_SYNC(gpu_sync) {
    return (s32)cudaStreamSynchronize(g_stream);
}

GPU_API GPU_KIN_SET_MECH(gpu_kin_set_mech) {
    if (!kin_capacity_ok<KinOpsGpu>(mech)) return (s32)cudaErrorInvalidValue;   // raise KinOpsGpu and rebuild
    g_kin_generated = hash_fnv1a_64(mech, sizeof(KinMech)) == AirGenOps::signature;
    KinWarpTables wt;
    g_warp_ok = kin_build_warp_tables(mech, &wt);
    if (g_warp_ok) {
        cudaError_t werr = cudaMemcpyToSymbol(c_warp, &wt, sizeof(wt));
        if (werr != cudaSuccess) return (s32)werr;
    }
    cudaError_t err = cudaMemcpyToSymbol(c_mech, mech, sizeof(KinMech));
    if (err == cudaSuccess) err = cudaMemcpyToSymbol(c_method, method, sizeof(KinMethod));
    if (err == cudaSuccess) err = cudaMemcpyToSymbol(c_control, control, sizeof(KinControl));
    return (s32)err;
}

internal void gpu_kin_launch(const GpuKinLaunch *launch, cudaStream_t stream) {
    u32 block = g_kin_block;
    u32 lanes = launch->lane_count ? launch->lane_count : launch->padded;
    u32 grid = (lanes + block - 1) / block;
    if (g_kin_mode == 1 && g_warp_ok) {
        u32 wgrid = (lanes + KIN_WARPS_PER_BLOCK - 1) / KIN_WARPS_PER_BLOCK;
        u32 wthreads = 32 * KIN_WARPS_PER_BLOCK;
        if (launch->precision == 1)
            kin_advance_warp_kernel<f32, true><<<wgrid, wthreads, KIN_WARPS_PER_BLOCK * sizeof(KinWarpScratch<f32>), stream>>>(*launch);
        else
            kin_advance_warp_kernel<f64, false><<<wgrid, wthreads, KIN_WARPS_PER_BLOCK * sizeof(KinWarpScratch<f64>), stream>>>(*launch);
        return;
    }
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
    cudaEventRecord(g_kin_start, g_stream);
    gpu_kin_launch(launch, g_stream);
    cudaEventRecord(g_kin_stop, g_stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return (s32)err;
    err = cudaEventSynchronize(g_kin_stop);
    if (err != cudaSuccess) return (s32)err;
    if (kernel_ms) cudaEventElapsedTime(kernel_ms, g_kin_start, g_kin_stop);
    return (s32)cudaGetLastError();
}

GPU_API GPU_KIN_SET_MODE(gpu_kin_set_mode) {
    g_kin_mode = mode;
    return (mode == 1 && !g_warp_ok) ? (s32)cudaErrorInvalidValue : 0;
}
