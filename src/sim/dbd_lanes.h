#ifndef DBD_LANES_H
#define DBD_LANES_H

// The per-lane-group DBD advance, shared verbatim by the CPU (V = f64x4 or
// f64, called from dbd.cpp) and the GPU (V = f64, one thread per zone,
// called from kernels.cu). Loads one group of zones from the SoA arrays,
// runs kin_advance, stores it back. y_soa and y_out may alias.

#include "kinetics/kin_solver.h"

// Pointers into the DbdZones SoA block that one advance needs. Plain data,
// passed by value to the CUDA kernel.
struct DbdLaneArrays {
    // solver writes
    f64 *y, *h, *steps, *rejects, *pulses, *failed;
    // solver reads
    const f64 *k, *src, *pulse, *period, *pulse_first, *pulse_count;
    u32 padded;
};

// One group of LaneTraits<V>::width zones starting at lane i0, over local
// time [0, duration]. Storage is f64 whatever the lane precision: V = f32 /
// f32x8 converts on load and store. Counters are per call and added to the
// f64 totals, so f32 lanes never hold large integers.
//
// Delta (f32 lanes): integrate the increment over the step-start state and add
// it to the f64 storage, so storage never round-trips through f32.
template <class V, class Ops = KinOpsDefault, bool Delta = false>
SIM_HD void dbd_advance_group(const KinMech *m, const KinMethod *method, const KinControl *ctl,
                              DbdLaneArrays a, u32 i0, f64 duration) {
    V y[Ops::species], k[Ops::reactions], src[Ops::species], pulse[Ops::species];
    V base[Delta ? Ops::species : 1];
    const u32 n = Ops::n(m);
    const u32 p = a.padded;
    for (u32 s = 0; s < n; ++s) {
        if (Delta) {
            base[s] = lane_load_f64(a.y + s * p + i0, (V *)0);
            y[s] = lane_splat((V *)0, 0.0);
        } else {
            y[s] = lane_load_f64(a.y + s * p + i0, (V *)0);
        }
        src[s] = lane_load_f64(a.src + s * p + i0, (V *)0);
        pulse[s] = lane_load_f64(a.pulse + s * p + i0, (V *)0);
    }
    for (u32 r = 0; r < (u32)Ops::reactions && r < m->reaction_count; ++r) k[r] = lane_load_f64(a.k + r * p + i0, (V *)0);
    V h = lane_load_f64(a.h + i0, (V *)0);
    V per = lane_load_f64(a.period + i0, (V *)0);
    V first = lane_load_f64(a.pulse_first + i0, (V *)0);
    V count = lane_load_f64(a.pulse_count + i0, (V *)0);
    V zero = lane_splat((V *)0, 0.0);
    KinLaneStats<V> st;
    st.steps = zero;
    st.rejects = zero;
    st.pulses = zero;
    st.failed = lane_load_f64(a.failed + i0, (V *)0);

    kin_advance<V, Ops, Delta>(m, method, ctl, k, src, pulse, per, first, count, y, lane_splat((V *)0, duration), h, st,
                               Delta ? base : (const V *)0);

    for (u32 s = 0; s < n; ++s) {
        if (Delta) {
            f64 *dst = a.y + s * p + i0;
            lane_accumulate_f64(dst, y[s]);
            // base was the f32 rounding of the f64 state, so base + y >= 0 can still
            // leave the f64 sum a few ulp below zero.
            if (ctl->clamp_negative)
                for (u32 j = 0; j < (u32)LaneTraits<V>::width; ++j) dst[j] = dst[j] > 0.0 ? dst[j] : 0.0;
        } else {
            lane_store_f64(a.y + s * p + i0, y[s]);
        }
    }
    lane_store_f64(a.h + i0, h);
    lane_store_f64(a.failed + i0, st.failed);
    lane_accumulate_f64(a.steps + i0, st.steps);
    lane_accumulate_f64(a.rejects + i0, st.rejects);
    lane_accumulate_f64(a.pulses + i0, st.pulses);
}

#endif // DBD_LANES_H
