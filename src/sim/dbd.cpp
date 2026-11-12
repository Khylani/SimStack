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
    f64 *y, *h, *pulse_index;
    f64 *steps, *rejects, *pulses, *failed;
    // solver inputs, [reaction][padded] / [species][padded] / [padded]
    f64 *k, *src, *pulse, *period;
    // operating point per zone
    f64 *T_K, *p_Pa, *x_H2O, *power_W_cm3, *freq_Hz;
};

internal void dbd_zones_alloc(DbdZones *z, Arena *arena, KinModel *model, u32 count) {
    z->count = count;
    z->padded = SoaPadCount(count);
    z->n_species = model->mech.species_count;
    z->n_reactions = model->mech.reaction_count;
    u64 p = z->padded, ns = z->n_species, nr = z->n_reactions;
    struct { f64 **ptr; u64 count; } layout[] = {
        { &z->y, ns * p }, { &z->h, p }, { &z->pulse_index, p },
        { &z->steps, p }, { &z->rejects, p }, { &z->pulses, p }, { &z->failed, p },
        { &z->k, nr * p }, { &z->src, ns * p }, { &z->pulse, ns * p }, { &z->period, p },
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
internal void dbd_zones_prepare_source(DbdZones *z, AirSpecies *sp, AirSourceModel *source, DbdDischargeMode mode) {
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
            z->period[i] = 1.0 / pulses_per_s;
        } else {
            air_source_delta(sp, source, y, n, on ? z->power_W_cm3[i] : 0.0, delta);   // per second
            for (u32 s = 0; s < n; ++s) { z->src[s * p + i] = delta[s]; z->pulse[s * p + i] = 0.0; }
            z->period[i] = 0.0;
        }
    }
}

internal void dbd_zones_prepare(DbdZones *z, KinModel *model, AirSpecies *sp, AirSourceModel *source,
                                DbdDischargeMode mode) {
    dbd_zones_prepare_rates(z, model);
    dbd_zones_prepare_source(z, sp, source, mode);
}

// Advances every zone over [t0, t1] on the CPU. V = f64x4 (AVX2) or f64 (reference).
template <class V>
internal void dbd_advance_cpu(DbdZones *z, const KinMech *m, const KinMethod *method, const KinControl *ctl,
                              f64 t0, f64 t1, u32 first_zone = 0, u32 end_zone = U32_MAX) {
    const u32 w = LaneTraits<V>::width;
    u32 end = Minimum(end_zone, z->padded);
    Assert(first_zone % w == 0);
    for (u32 i0 = first_zone; i0 < end; i0 += w) {
        dbd_advance_group<V>(m, method, ctl, z->y, z->k, z->src, z->pulse, z->period, z->h, z->pulse_index,
                             z->steps, z->rejects, z->pulses, z->failed, z->y, z->padded, i0, t0, t1);
    }
}

// CPU advance spread over the platform work queue. Lane groups are
// independent, so the result does not depend on the thread count.
struct DbdWork {
    DbdZones *z;
    const KinMech *m;
    const KinMethod *method;
    const KinControl *ctl;
    f64 t0, t1;
    u32 first, end;
};

internal PLATFORM_WORK_QUEUE_CALLBACK(dbd_work_advance) {
    DbdWork *w = (DbdWork *)data;
    dbd_advance_cpu<f64x4>(w->z, w->m, w->method, w->ctl, w->t0, w->t1, w->first, w->end);
}

internal void dbd_advance_threads(DbdZones *z, const KinMech *m, const KinMethod *method, const KinControl *ctl,
                                  f64 t0, f64 t1, PlatformAPI *api, PlatformWorkQueue *queue, u32 threads, Arena *frame) {
    TIMED_FUNCTION();
    const u32 w = LaneTraits<f64x4>::width;
    u32 groups = z->padded / w;
    if (!queue || threads <= 1 || groups <= 1) {
        dbd_advance_cpu<f64x4>(z, m, method, ctl, t0, t1);
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
        item->t0 = t0; item->t1 = t1;
        item->first = g * w;
        item->end = Minimum(groups, g + per) * w;
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
internal s32 dbd_advance_gpu(DbdZones *z, GpuAPI *gpu, DbdGpu *g, f64 t0, f64 t1) {
    TIMED_FUNCTION();
    if (!g->d_block || g->capacity < z->block_count) {
        g->d_block = (f64 *)gpu->gpu_alloc(sizeof(f64) * z->block_count);   // grows the bump arena; fine for a few resizes
        g->capacity = g->d_block ? z->block_count : 0;
        if (!g->d_block) return 2;   // cudaErrorMemoryAllocation
    }
    s32 err = gpu->gpu_copy_to_device(g->d_block, z->block, sizeof(f64) * z->block_count);
    if (err) return err;
#define DEV(field) (g->d_block + (z->field - z->block))
    GpuKinLaunch L = {};
    L.y = DEV(y); L.h = DEV(h); L.pulse_index = DEV(pulse_index);
    L.steps = DEV(steps); L.rejects = DEV(rejects); L.pulses = DEV(pulses); L.failed = DEV(failed);
    L.k = DEV(k); L.src = DEV(src); L.pulse = DEV(pulse); L.period = DEV(period);
#undef DEV
    L.padded = z->padded;
    L.t0 = t0;
    L.t1 = t1;
    err = gpu->gpu_kin_advance(&L, &g->last_kernel_ms);
    if (err) return err;
    ++g->launches;
    return gpu->gpu_copy_to_host(z->block, g->d_block, sizeof(f64) * z->mutable_count);
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
    c.atol = 1e4;           // cm^-3: ~4e-16 mole fraction, far below anything that matters
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
