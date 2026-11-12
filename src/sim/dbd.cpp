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

struct DbdZones {
    u32 count;          // zones in use
    u32 padded;         // multiple of SIMD_LANE_COUNT; padding lanes are inert copies of zone 0
    u32 n_species, n_reactions;

    // solver state, [species][padded] / [reaction][padded] / [padded]
    f64 *y, *k, *src, *pulse;
    f64 *period, *h, *pulse_index;
    f64 *steps, *rejects, *pulses, *failed;

    // operating point per zone
    f64 *T_K, *p_Pa, *x_H2O, *power_W_cm3, *freq_Hz;
};

internal void dbd_zones_alloc(DbdZones *z, Arena *arena, KinModel *model, u32 count) {
    z->count = count;
    z->padded = SoaPadCount(count);
    z->n_species = model->mech.species_count;
    z->n_reactions = model->mech.reaction_count;
    u32 p = z->padded;
    z->y = push_array_aligned(arena, f64, (u64)z->n_species * p, SIMD_ALIGN);
    z->k = push_array_aligned(arena, f64, (u64)z->n_reactions * p, SIMD_ALIGN);
    z->src = push_array_aligned(arena, f64, (u64)z->n_species * p, SIMD_ALIGN);
    z->pulse = push_array_aligned(arena, f64, (u64)z->n_species * p, SIMD_ALIGN);
    f64 **lane_arrays[] = { &z->period, &z->h, &z->pulse_index, &z->steps, &z->rejects, &z->pulses, &z->failed,
                            &z->T_K, &z->p_Pa, &z->x_H2O, &z->power_W_cm3, &z->freq_Hz };
    for (u32 i = 0; i < ArrayCount(lane_arrays); ++i) *lane_arrays[i] = push_array_aligned(arena, f64, p, SIMD_ALIGN);
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

// Rate constants and discharge source from the operating point and the
// current composition. Cheap (host libm per zone); call once per master step.
internal void dbd_zones_prepare(DbdZones *z, KinModel *model, AirSpecies *sp, AirSourceModel *source,
                                DbdDischargeMode mode) {
    TIMED_FUNCTION();
    u32 p = z->padded, n = z->n_species;
    for (u32 i = 0; i < p; ++i) {
        f64 y[KIN_MAX_SPECIES], k[KIN_MAX_REACTIONS], delta[KIN_MAX_SPECIES];
        f64 total = 0.0;
        for (u32 s = 0; s < n; ++s) { y[s] = z->y[s * p + i]; total += y[s]; }
        kin_eval_rate_constants(model, z->T_K[i], total, k);
        for (u32 r = 0; r < z->n_reactions; ++r) z->k[r * p + i] = k[r];

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
    c.rtol = 1e-4;
    c.atol = 1e2;           // cm^-3: ~4e-18 mole fraction, far below anything that matters
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
