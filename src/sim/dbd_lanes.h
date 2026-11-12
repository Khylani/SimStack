#ifndef DBD_LANES_H
#define DBD_LANES_H

// The per-lane-group DBD advance, shared verbatim by the CPU (V = f64x4 or
// f64, called from dbd.cpp) and the GPU (V = f64, one thread per zone,
// called from kernels.cu). Loads one group of zones from the SoA arrays,
// runs kin_advance, stores it back. y_soa and y_out may alias.

#include "kinetics/kin_solver.h"

// One group of LaneTraits<V>::width zones starting at lane i0.
template <class V, class Ops = KinOpsDefault>
SIM_HD void dbd_advance_group(const KinMech *m, const KinMethod *method, const KinControl *ctl,
                              const f64 *y_soa, const f64 *k_soa, const f64 *src_soa, const f64 *pulse_soa,
                              const f64 *period, f64 *h_soa, f64 *pulse_index, f64 *steps, f64 *rejects,
                              f64 *pulses, f64 *failed, f64 *y_out, u32 padded, u32 i0, f64 t0, f64 t1) {
    V y[Ops::species], k[Ops::reactions], src[Ops::species], pulse[Ops::species];
    const u32 n = Ops::n(m);
    for (u32 s = 0; s < n; ++s) {
        y[s] = lane_load(y_soa + s * padded + i0, (V *)0);
        src[s] = lane_load(src_soa + s * padded + i0, (V *)0);
        pulse[s] = lane_load(pulse_soa + s * padded + i0, (V *)0);
    }
    for (u32 r = 0; r < (u32)Ops::reactions && r < m->reaction_count; ++r) k[r] = lane_load(k_soa + r * padded + i0, (V *)0);
    V h = lane_load(h_soa + i0, (V *)0);
    V pidx = lane_load(pulse_index + i0, (V *)0);
    V per = lane_load(period + i0, (V *)0);
    KinLaneStats<V> st;
    st.steps = lane_load(steps + i0, (V *)0);
    st.rejects = lane_load(rejects + i0, (V *)0);
    st.pulses = lane_load(pulses + i0, (V *)0);
    st.failed = lane_load(failed + i0, (V *)0);

    kin_advance<V, Ops>(m, method, ctl, k, src, pulse, per, y, lane_splat((V *)0, t0), lane_splat((V *)0, t1), h, pidx, st);

    for (u32 s = 0; s < n; ++s) lane_store(y_out + s * padded + i0, y[s]);
    lane_store(h_soa + i0, h);
    lane_store(pulse_index + i0, pidx);
    lane_store(steps + i0, st.steps);
    lane_store(rejects + i0, st.rejects);
    lane_store(pulses + i0, st.pulses);
    lane_store(failed + i0, st.failed);
}

#endif // DBD_LANES_H
