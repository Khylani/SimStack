// DBD reactor zones: the time-dependent ("dynamic") plasma-chemistry domain.
//
// Each zone is a well-mixed gas volume with its own operating point
// (temperature, pressure, humidity, discharge power and frequency). Zones
// are lanes of the batched Rosenbrock solver: four per AVX2 register on the
// CPU, one per thread on the GPU. All state is SoA in the permanent arena,
// padded to SIMD_LANE_COUNT, so the same arrays feed both paths.
//
// Two discharge models per zone:
//   pulsed    one instantaneous radical injection per AC half-cycle
//             (micro-discharges are ~10-100 ns, far below the solver's
//             resolution), integration restarts at each pulse
//   averaged  the same energy as a continuous source term (cheaper,
//             loses the pulse-to-pulse radical peaks)
//
// Zones can be chained as a plug-flow reactor (gas from zone i flows into
// zone i+1 over the residence time) or run independently as a sweep.
// Included by simstack.cpp and the test TU.

#include "kinetics/kin_build.cpp"
#include "dbd_lanes.h"
#include "kinetics/air_mech.cpp"
#include "air_mech_gen.h"   // build/gen: straight-line kernels for air_mech (src/tools/kin_codegen.cpp)

// 1 if the generated kernels were made from exactly this mechanism.
internal b32 dbd_generated_matches(KinModel *model) {
    return hash_fnv1a_64(&model->mech, sizeof(KinMech)) == AirGenOps::signature;
}

enum DbdDischargeMode {
    DbdDischarge_Pulsed,
    DbdDischarge_Averaged,
};

// Every per-zone array lives in one contiguous f64 block. The GPU mirror is
// one device allocation of the same size, and a device pointer is the device
// base plus the host array's offset: host owns the layout, device gets
// offsets. The block starts with what the solver writes, so a download only
// needs the first `mutable_count` doubles.
struct DbdZones {
    u32 count;          // zones in use
    u32 padded;         // multiple of SIMD_LANE_COUNT; padding lanes are inert copies of zone 0
    u32 n_species, n_reactions;

    f64 *block;
    u64 block_count;    // doubles in block
    u64 mutable_count;  // doubles from block start that the solver writes

    // solver state (written by the solver), [species][padded] / [padded]
    f64 *y, *h;
    f64 *steps, *rejects, *pulses, *failed;
    // solver inputs, [reaction][padded] / [species][padded] / [padded]
    f64 *k, *src, *pulse, *period;
    f64 *pulse_first, *pulse_count;   // this master step, local time (host computes from pulse_index)
    // host only
    f64 *pulse_index;                 // pulses applied since reset (absolute pulse clock)
    f64 *T_K, *p_Pa, *x_H2O, *power_W_cm3, *freq_Hz;
};

internal void dbd_zones_alloc(DbdZones *z, Arena *arena, KinModel *model, u32 count) {
    z->count = count;
    z->padded = SoaPadCount(count);
    z->n_species = model->mech.species_count;
    z->n_reactions = model->mech.reaction_count;
    u64 p = z->padded, ns = z->n_species, nr = z->n_reactions;
    struct { f64 **ptr; u64 count; } layout[] = {
        { &z->y, ns * p }, { &z->h, p },
        { &z->steps, p }, { &z->rejects, p }, { &z->pulses, p }, { &z->failed, p },
        { &z->k, nr * p }, { &z->src, ns * p }, { &z->pulse, ns * p }, { &z->period, p },
        { &z->pulse_first, p }, { &z->pulse_count, p }, { &z->pulse_index, p },
        { &z->T_K, p }, { &z->p_Pa, p }, { &z->x_H2O, p }, { &z->power_W_cm3, p }, { &z->freq_Hz, p },
    };
    u64 total = 0;
    for (u32 i = 0; i < ArrayCount(layout); ++i) total += layout[i].count;   // each a multiple of 8 doubles
    z->block = push_array_aligned(arena, f64, total, 64);
    z->block_count = total;
    u64 offset = 0;
    for (u32 i = 0; i < ArrayCount(layout); ++i) {
        *layout[i].ptr = z->block + offset;
        offset += layout[i].count;
        if (layout[i].ptr == &z->failed) z->mutable_count = offset;
    }
}

// Fills every zone with feed gas at its operating point and clears counters.
internal void dbd_zones_reset(DbdZones *z, AirSpecies *sp) {
    for (u32 i = 0; i < z->padded; ++i) {
        f64 feed[KIN_MAX_SPECIES];
        air_feed(sp, z->n_species, z->T_K[i], z->p_Pa[i], z->x_H2O[i], feed);
        for (u32 s = 0; s < z->n_species; ++s) z->y[s * z->padded + i] = feed[s];
        z->h[i] = 0.0;
        z->pulse_index[i] = 0.0;
        z->steps[i] = z->rejects[i] = z->pulses[i] = z->failed[i] = 0.0;
    }
}

// Copies zone 0's operating point into the padding lanes so they integrate
// something physical (and cost the same as a real lane) instead of zeros.
internal void dbd_zones_fill_padding(DbdZones *z) {
    f64 *arrays[] = { z->T_K, z->p_Pa, z->x_H2O, z->power_W_cm3, z->freq_Hz };
    for (u32 a = 0; a < ArrayCount(arrays); ++a)
        for (u32 i = z->count; i < z->padded; ++i) arrays[a][i] = arrays[a][0];
}

// Rate constants from each zone's temperature and pressure (third-body
// density M = p / kT). Host libm per zone and reaction: call when T or p change.
internal void dbd_zones_prepare_rates(DbdZones *z, KinModel *model) {
    TIMED_FUNCTION();
    u32 p = z->padded;
    for (u32 i = 0; i < p; ++i) {
        f64 k[KIN_MAX_REACTIONS];
        f64 M = z->p_Pa[i] / (BOLTZMANN_J_K * z->T_K[i]) * 1e-6;
        kin_eval_rate_constants(model, z->T_K[i], M, k);
        for (u32 r = 0; r < z->n_reactions; ++r) z->k[r * p + i] = k[r];
    }
}

// Discharge source from the operating point and the current composition
// (G-values scale with parent mole fractions). Cheap; call every master step.
// Also schedules this step's pulses: the pulse clock is absolute and f64
// (pulse j of a zone fires at j * period, j >= 1), the solver gets the first
// pulse in local time and how many fall in (t0, t1].
internal void dbd_zones_prepare_source(DbdZones *z, AirSpecies *sp, AirSourceModel *source, DbdDischargeMode mode,
                                       f64 t0, f64 t1) {
    TIMED_FUNCTION();
    u32 p = z->padded, n = z->n_species;
    for (u32 i = 0; i < p; ++i) {
        f64 y[KIN_MAX_SPECIES], delta[KIN_MAX_SPECIES];
        for (u32 s = 0; s < n; ++s) y[s] = z->y[s * p + i];
        f64 freq = z->freq_Hz[i];
        b32 on = z->power_W_cm3[i] > 0.0 && freq > 0.0;
        f64 pulses_per_s = 2.0 * freq;   // one micro-discharge burst per half-cycle
        if (mode == DbdDischarge_Pulsed && on) {
            air_source_delta(sp, source, y, n, z->power_W_cm3[i] / pulses_per_s, delta);
            for (u32 s = 0; s < n; ++s) { z->pulse[s * p + i] = delta[s]; z->src[s * p + i] = 0.0; }
            f64 period = 1.0 / pulses_per_s;
            z->period[i] = period;
            f64 next = z->pulse_index[i] + 1.0, count = 0.0;
            while ((next + count) * period <= t1) count += 1.0;
            z->pulse_first[i] = next * period - t0;
            z->pulse_count[i] = count;
        } else {
            air_source_delta(sp, source, y, n, on ? z->power_W_cm3[i] : 0.0, delta);   // per second
            for (u32 s = 0; s < n; ++s) { z->src[s * p + i] = delta[s]; z->pulse[s * p + i] = 0.0; }
            z->period[i] = 0.0;
            z->pulse_first[i] = 0.0;
            z->pulse_count[i] = 0.0;
        }
    }
}

// After an advance: the scheduled pulses have been applied (failed lanes keep
// their state frozen; their clock still moves so they do not replay pulses).
internal void dbd_zones_commit_pulses(DbdZones *z) {
    for (u32 i = 0; i < z->padded; ++i) z->pulse_index[i] += z->pulse_count[i];
}

internal DbdLaneArrays dbd_lane_arrays(DbdZones *z) {
    DbdLaneArrays a;
    a.y = z->y; a.h = z->h; a.steps = z->steps; a.rejects = z->rejects; a.pulses = z->pulses; a.failed = z->failed;
    a.k = z->k; a.src = z->src; a.pulse = z->pulse; a.period = z->period;
    a.pulse_first = z->pulse_first; a.pulse_count = z->pulse_count;
    a.padded = z->padded;
    return a;
}

// Advances every zone over one master step of length `duration` on the CPU
// (sources and pulses prepared for it). V = f64x4 (AVX2) or f64 (reference);
// Ops = generated straight-line kernels or the table walk.
template <class V, class Ops = KinOpsDefault, bool Delta = false>
internal void dbd_advance_cpu(DbdZones *z, const KinMech *m, const KinMethod *method, const KinControl *ctl,
                              f64 duration, u32 first_zone = 0, u32 end_zone = U32_MAX) {
    const u32 w = LaneTraits<V>::width;
    u32 end = Minimum(end_zone, z->padded);
    Assert(first_zone % w == 0);
    DbdLaneArrays a = dbd_lane_arrays(z);
    for (u32 i0 = first_zone; i0 < end; i0 += w) dbd_advance_group<V, Ops, Delta>(m, method, ctl, a, i0, duration);
}

// Whole master step on one thread: source/pulse schedule, advance, commit.
// (Tests and tools; the sim uses dbd_advance_threads / dbd_advance_gpu.)
template <class V, class Ops = KinOpsDefault, bool Delta = false>
internal void dbd_step_cpu(DbdZones *z, KinModel *model, AirSpecies *sp, AirSourceModel *source, DbdDischargeMode mode,
                           const KinMethod *method, const KinControl *ctl, f64 t0, f64 t1) {
    dbd_zones_prepare_source(z, sp, source, mode, t0, t1);
    dbd_advance_cpu<V, Ops, Delta>(z, &model->mech, method, ctl, t1 - t0);
    dbd_zones_commit_pulses(z);
}

// Lane precision of the solver. Storage is always f64.
//   f64  f64x4 lanes (4 zones per AVX2 register), plain formulation: reference
//   f32  f32x8 lanes (8 zones per register), delta formulation over the f64 state
enum DbdPrecision {
    DbdPrecision_F64,
    DbdPrecision_F32,
};

// CPU advance spread over the platform work queue. Lane groups are
// independent, so the result does not depend on the thread count.
struct DbdWork {
    DbdZones *z;
    const KinMech *m;
    const KinMethod *method;
    const KinControl *ctl;
    f64 duration;
    u32 first, end;
    b32 generated;
    u32 precision;   // DbdPrecision
};

internal void dbd_work_run(DbdWork *w) {
    if (w->precision == DbdPrecision_F32) {
        if (w->generated) dbd_advance_cpu<f32x8, AirGenOps, true>(w->z, w->m, w->method, w->ctl, w->duration, w->first, w->end);
        else dbd_advance_cpu<f32x8, KinOpsDefault, true>(w->z, w->m, w->method, w->ctl, w->duration, w->first, w->end);
    } else {
        if (w->generated) dbd_advance_cpu<f64x4, AirGenOps>(w->z, w->m, w->method, w->ctl, w->duration, w->first, w->end);
        else dbd_advance_cpu<f64x4>(w->z, w->m, w->method, w->ctl, w->duration, w->first, w->end);
    }
}

internal PLATFORM_WORK_QUEUE_CALLBACK(dbd_work_advance) {
    dbd_work_run((DbdWork *)data);
}

// Lanes [first_lane, end_lane) (multiples of 8; default: all).
internal void dbd_advance_threads(DbdZones *z, const KinMech *m, const KinMethod *method, const KinControl *ctl,
                                  f64 duration, b32 generated, u32 precision, PlatformAPI *api,
                                  PlatformWorkQueue *queue, u32 threads, Arena *frame,
                                  u32 first_lane = 0, u32 end_lane = U32_MAX) {
    TIMED_FUNCTION();
    const u32 w = precision == DbdPrecision_F32 ? (u32)LaneTraits<f32x8>::width : (u32)LaneTraits<f64x4>::width;
    end_lane = Minimum(end_lane, z->padded);
    if (first_lane >= end_lane) return;
    u32 groups = (end_lane - first_lane) / w;
    if (!queue || threads <= 1 || groups <= 1) {
        DbdWork all = { z, m, method, ctl, duration, first_lane, end_lane, generated, precision };
        dbd_work_run(&all);
        return;
    }
    // ~4 chunks per thread so uneven lane costs (stiffer zones) balance out.
    u32 chunks = Minimum(groups, 4 * threads);
    u32 per = (groups + chunks - 1) / chunks;
    TempMemory temp = begin_temp(frame);
    DbdWork *work = push_array(frame, DbdWork, chunks);
    u32 used = 0;
    for (u32 g = 0; g < groups; g += per) {
        DbdWork *item = work + used++;
        item->z = z; item->m = m; item->method = method; item->ctl = ctl;
        item->duration = duration;
        item->first = first_lane + g * w;
        item->end = first_lane + Minimum(groups, g + per) * w;
        item->generated = generated;
        item->precision = precision;
        api->add_work_entry(queue, dbd_work_advance, item);
    }
    api->complete_all_work(queue);
    end_temp(temp);
}

// GPU mirror of a DbdZones block (device memory from the GPU module's arena).
struct DbdGpu {
    f64 *d_block;
    u64 capacity;       // doubles
    f32 last_kernel_ms;
    u64 launches;
};

// Uploads the whole zone block, advances every lane on the device, downloads
// the solver-written prefix. Returns 0 on success, else the CUDA error code.
internal s32 dbd_advance_gpu(DbdZones *z, GpuAPI *gpu, DbdGpu *g, f64 duration, u32 precision = DbdPrecision_F64) {
    TIMED_FUNCTION();
    if (!g->d_block || g->capacity < z->block_count) {
        g->d_block = (f64 *)gpu->gpu_alloc(sizeof(f64) * z->block_count);   // device arena; reset with the scene
        g->capacity = g->d_block ? z->block_count : 0;
        if (!g->d_block) return 2;   // cudaErrorMemoryAllocation
    }
    s32 err = gpu->gpu_copy_to_device(g->d_block, z->block, sizeof(f64) * z->block_count);
    if (err) return err;
#define DEV(field) (g->d_block + (z->field - z->block))
    GpuKinLaunch L = {};
    L.y = DEV(y); L.h = DEV(h);
    L.steps = DEV(steps); L.rejects = DEV(rejects); L.pulses = DEV(pulses); L.failed = DEV(failed);
    L.k = DEV(k); L.src = DEV(src); L.pulse = DEV(pulse); L.period = DEV(period);
    L.pulse_first = DEV(pulse_first); L.pulse_count = DEV(pulse_count);
#undef DEV
    L.padded = z->padded;
    L.precision = precision;
    L.duration = duration;
    err = gpu->gpu_kin_advance(&L, &g->last_kernel_ms);
    if (err) return err;
    ++g->launches;
    return gpu->gpu_copy_to_host(z->block, g->d_block, sizeof(f64) * z->mutable_count);
}

// Hybrid CPU + GPU: lanes [0, n) go to the GPU asynchronously while the CPU
// work queue integrates [n, padded). Both paths perform identical IEEE
// operations, so the result is bit-identical to either device alone for ANY
// split; the split can therefore follow measurements (timing-dependent)
// without affecting determinism.
//
// Split choice: the step's wall time is measured for every split tried (in
// groups of 8 lanes, 0 = CPU only ... all = GPU only) and kept as an EMA. The
// split with the best time is used; every 8th step a neighbour of it is tried
// at several scales (+-1, +-4, +-16 groups). This finds the optimum including
// the GPU's fixed latency (launch, copies, per-thread dependency chain), which a
// throughput ratio misses, and follows drift as the chemistry gets stiffer.
#define DBD_HYBRID_BUCKETS 1025
struct DbdHybrid {
    f32 step_ms[DBD_HYBRID_BUCKETS];     // EMA of step wall time per split (index = split / 8), 0 = untried
    u32 tried_at[DBD_HYBRID_BUCKETS];    // step count when each split was last measured
    u32 split;                           // lanes given to the GPU next step (multiple of 8)
    u32 steps;
    u32 explore_interval;                // steps between probes: 8, doubling to 1024 while probes lose
    u32 next_probe;
    u32 best_bucket;
    b32 probing;
    u32 phase;                           // 0: CPU-only baseline, 1-2: GPU probes, 3: model split + local search
    f64 cpu_ms_per_lane, gpu_probe_ms[2];
    u32 gpu_probe_lanes[2];
    b32 registered;                      // host block page-locked
    void *registered_block;
    f32 last_gpu_ms, last_cpu_ms;
    s32 force_split;                     // tests: >= 0 fixes the split
};

internal u32 dbd_hybrid_best(DbdHybrid *hy, u32 buckets) {
    u32 best = hy->split / 8;
    f32 best_ms = 1e30f;
    for (u32 b = 0; b < buckets; ++b)
        if (hy->step_ms[b] > 0.0f && hy->step_ms[b] < best_ms) { best_ms = hy->step_ms[b]; best = b; }
    return best;
}

internal s32 dbd_advance_hybrid(DbdZones *z, GpuAPI *gpu, DbdGpu *g, DbdHybrid *hy, f64 duration, u32 precision,
                                const KinMech *m, const KinMethod *method, const KinControl *ctl, b32 generated,
                                PlatformAPI *api, PlatformWorkQueue *queue, u32 threads, Arena *frame) {
    TIMED_FUNCTION();
    if (!g->d_block || g->capacity < z->block_count) {
        g->d_block = (f64 *)gpu->gpu_alloc(sizeof(f64) * z->block_count);
        g->capacity = g->d_block ? z->block_count : 0;
        if (!g->d_block) return 2;
    }
    if (!hy->registered || hy->registered_block != z->block) {
        s32 err = gpu->gpu_host_register(z->block, sizeof(f64) * z->block_count);
        if (err) return err;
        hy->registered = 1;
        hy->registered_block = z->block;
        for (u32 b = 0; b < DBD_HYBRID_BUCKETS; ++b) { hy->step_ms[b] = 0.0f; hy->tried_at[b] = 0; }
        hy->split = 0;                   // phase 0: CPU-only baseline
        hy->phase = 0;
        hy->explore_interval = 8;
        hy->next_probe = 0;
        hy->best_bucket = hy->split / 8;
        hy->probing = 0;
    }
    u32 padded = z->padded;
    u32 buckets = Minimum(padded / 8 + 1, (u32)DBD_HYBRID_BUCKETS);
    u32 split = hy->force_split >= 0 ? (u32)hy->force_split : hy->split;
    split = Minimum(split & ~7u, padded);

    u64 t0 = api->now_ns();
    s32 err = 0;
    if (split > 0) {
#define DEV(field) (g->d_block + (z->field - z->block))
        GpuKinLaunch L = {};
        L.y = DEV(y); L.h = DEV(h);
        L.steps = DEV(steps); L.rejects = DEV(rejects); L.pulses = DEV(pulses); L.failed = DEV(failed);
        L.k = DEV(k); L.src = DEV(src); L.pulse = DEV(pulse); L.period = DEV(period);
        L.pulse_first = DEV(pulse_first); L.pulse_count = DEV(pulse_count);
#undef DEV
        L.padded = padded;
        L.lane_count = split;
        L.precision = precision;
        L.duration = duration;
        err = gpu->gpu_kin_advance_async(&L, z->block, g->d_block, z->block_count / padded, z->mutable_count / padded);
        if (err) return err;
    }
    dbd_advance_threads(z, m, method, ctl, duration, generated, precision, api, queue, threads, frame, split, padded);
    u64 t1 = api->now_ns();
    f32 kernel_ms = 0.0f, gpu_ms = 0.0f;
    if (split > 0) {
        err = gpu->gpu_kin_wait(&kernel_ms, &gpu_ms);
        if (err) return err;
        g->last_kernel_ms = kernel_ms;
        ++g->launches;
    }
    u64 t2 = api->now_ns();
    hy->last_gpu_ms = gpu_ms;
    hy->last_cpu_ms = (f32)((f64)(t1 - t0) * 1e-6);
    if (hy->force_split >= 0) return 0;

    // record this split's step time, then choose the next split
    f32 step_ms = (f32)((f64)(t2 - t0) * 1e-6);
    u32 b = split / 8;
    hy->step_ms[b] = hy->step_ms[b] > 0.0f ? 0.7f * hy->step_ms[b] + 0.3f * step_ms : step_ms;
    hy->tried_at[b] = ++hy->steps;

    // Start: model T_gpu(n) = o + n c_g, T_cpu(n) = (N - n) c_c from one CPU-only
    // step and two GPU probes, then the split that equalizes them (0 if the GPU's
    // fixed latency alone exceeds the whole CPU step).
    if (hy->phase < 3) {
        if (hy->phase == 0) {
            hy->cpu_ms_per_lane = step_ms / (f64)padded;
            hy->gpu_probe_lanes[0] = Minimum(8u, padded);
            hy->split = hy->gpu_probe_lanes[0];
            hy->phase = 1;
            return 0;
        }
        u32 i = hy->phase - 1;
        hy->gpu_probe_ms[i] = gpu_ms;
        if (hy->phase == 1 && padded >= 32) {
            hy->gpu_probe_lanes[1] = AlignPow2(padded / 2, 8);
            hy->split = hy->gpu_probe_lanes[1];
            hy->phase = 2;
            return 0;
        }
        f64 o = hy->gpu_probe_ms[0], cg = 0.0;
        if (hy->phase == 2) {
            cg = Maximum((hy->gpu_probe_ms[1] - hy->gpu_probe_ms[0]) / (f64)(hy->gpu_probe_lanes[1] - hy->gpu_probe_lanes[0]), 1e-9);
            o = Maximum(hy->gpu_probe_ms[0] - cg * hy->gpu_probe_lanes[0], 0.0);
        } else {
            cg = hy->gpu_probe_ms[0] / (f64)hy->gpu_probe_lanes[0];
            o = 0.0;
        }
        f64 cc = hy->cpu_ms_per_lane;
        f64 n = ((f64)padded * cc - o) / (cg + cc);
        n = Clamp(n, 0.0, (f64)padded);
        f64 t_split = Maximum(o + n * cg, ((f64)padded - n) * cc);
        u32 lanes = t_split < 0.95 * (f64)padded * cc ? ((u32)(n + 4.0) & ~7u) : 0u;
        hy->split = Minimum(lanes, padded);
        hy->best_bucket = hy->split / 8;
        hy->next_probe = hy->steps + hy->explore_interval;
        hy->phase = 3;
        return 0;
    }
    u32 best = dbd_hybrid_best(hy, buckets);
    if (hy->probing) {
        // a probe either found a better split (keep exploring eagerly) or not (back off)
        hy->explore_interval = best != hy->best_bucket ? 8 : Minimum(hy->explore_interval * 2, 1024u);
        hy->probing = 0;
    }
    hy->best_bucket = best;
    u32 next = best;
    if (hy->steps >= hy->next_probe) {
        static const s32 offsets[] = { 1, -1, 4, -4, 16, -16, 64, -64, 256, -256 };
        f32 best_ms = hy->step_ms[best];
        for (u32 t = 0; t < ArrayCount(offsets); ++t) {
            s32 cand = (s32)best + offsets[(hy->steps / 8 + t) % ArrayCount(offsets)];
            if (cand < 0 || cand >= (s32)buckets) continue;
            // skip splits already known to be clearly worse, unless their measurement is stale
            b32 known_bad = hy->step_ms[cand] > 1.25f * best_ms && hy->steps - hy->tried_at[cand] < 4096;
            if (known_bad) continue;
            next = (u32)cand;
            hy->probing = 1;
            break;
        }
        hy->next_probe = hy->steps + hy->explore_interval;
    }
    hy->split = next * 8;
    return 0;
}

// Plug flow along the zone chain: zone i relaxes toward zone i-1 (zone 0
// toward the feed) with time constant residence/count. Upwind, explicit,
// processed back to front so each zone sees its neighbour's old state.
internal void dbd_flow_exchange(DbdZones *z, AirSpecies *sp, f64 residence_s, f64 dt) {
    TIMED_FUNCTION();
    if (residence_s <= 0.0 || z->count == 0) return;
    f64 a = dt * (f64)z->count / residence_s;
    Assert(a <= 1.0);
    u32 p = z->padded, n = z->n_species;
    for (u32 i = z->count; i-- > 0;) {
        f64 up[KIN_MAX_SPECIES];
        if (i == 0) air_feed(sp, n, z->T_K[0], z->p_Pa[0], z->x_H2O[0], up);
        else for (u32 s = 0; s < n; ++s) up[s] = z->y[s * p + i - 1];
        for (u32 s = 0; s < n; ++s) {
            f64 *y = z->y + s * p + i;
            *y += a * (up[s] - *y);
        }
    }
}

internal KinControl dbd_default_control() {
    KinControl c = {};
    c.rtol = 1e-3;          // products stay within ~1e-5 of a 1e-9 reference (see --bench tolerance study)
    c.atol = 1e8;           // cm^-3 (~4 ppt): outlet products unchanged to 5 digits vs 1e4, 1.8x fewer steps (see docs)
    c.h_min = 1e-14;
    c.h_max = 0.0;
    c.h_start = 1e-9;       // after a pulse the radical decay starts on ns-us scales
    c.max_steps = 20000;
    c.clamp_negative = 1;
    return c;
}

// Mole fraction in ppm of species s in zone i.
internal f64 dbd_ppm(DbdZones *z, u32 s, u32 i) {
    f64 total = 0.0;
    for (u32 q = 0; q < z->n_species; ++q) total += z->y[q * z->padded + i];
    return total > 0.0 ? z->y[s * z->padded + i] / total * 1e6 : 0.0;
}
