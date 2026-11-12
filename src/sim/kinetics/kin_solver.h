#ifndef KIN_SOLVER_H
#define KIN_SOLVER_H

// Batched stiff ODE solver for mass-action chemistry: adaptive Rosenbrock
// methods with a sparse no-pivot LU, written once over a lane type V and
// instantiated as
//
//   V = f64     CPU reference / one CUDA thread per lane (kernels.cu)
//   V = f64x4   AVX2, four lanes per register (sim module)
//
// Each lane is an independent reactor (zone, sweep point) with its own time,
// step size and error control; lanes that finish early or fail are masked
// out while the others continue. All index tables (KinMech) are uniform
// across lanes, so there are no gathers and no divergent branches.
//
// Only lane_* operations are used (see base_lane.h), so every instantiation
// performs the same IEEE operations in the same order and produces the same
// bits on AVX2, scalar and CUDA.
//
// Included by the sim TU, the test TU and the nvcc TU.

#include "../../base/base_lane.h"
#include "kin_mech.h"

#define KIN_FAC_MIN   0.2     // step-size change limits per step (KPP defaults)
#define KIN_FAC_MAX   6.0
#define KIN_FAC_SAFE  0.9
#define KIN_FAC_REJ   0.1     // after two rejections in a row

// Per-lane counters, kept as V so they live in the same registers as the state.
template <class V>
struct KinLaneStats {
    V steps;        // accepted
    V rejects;
    V pulses;       // discharge pulses applied
    V failed;       // 1.0 once the lane exceeded max_steps (state is the last accepted one)
};

// dy/dt = src + sum_r net_r * k_r * prod(y[reactants of r])
template <class V>
SIM_HD void kin_fun(const KinMech *m, const V *k, const V *src, const V *y, V *f) {
    const u32 n = m->species_count;
    for (u32 s = 0; s < n; ++s) f[s] = src[s];
    for (u32 r = 0; r < m->reaction_count; ++r) {
        const KinReaction *rx = m->reactions + r;
        V rate = k[r];
        for (u32 p = 0; p < rx->reactant_count; ++p) rate = rate * y[rx->reactant[p]];
        for (u32 e = 0; e < rx->net_count; ++e) {
            u32 s = rx->net_species[e];
            f[s] = lane_fmadd(lane_splat((V *)0, (f64)rx->net_coef[e]), rate, f[s]);
        }
    }
}

// Compact sparse Jacobian (KinMech::nnz entries, fill-in positions stay 0).
template <class V>
SIM_HD void kin_jac(const KinMech *m, const V *k, const V *y, V *jac) {
    V zero = lane_splat((V *)0, 0.0);
    for (u32 i = 0; i < m->nnz; ++i) jac[i] = zero;
    for (u32 r = 0; r < m->reaction_count; ++r) {
        const KinReaction *rx = m->reactions + r;
        for (u32 p = 0; p < rx->reactant_count; ++p) {
            V d = k[r];
            for (u32 q = 0; q < rx->reactant_count; ++q) {
                if (q != p) d = d * y[rx->reactant[q]];
            }
            for (u32 e = 0; e < rx->net_count; ++e) {
                u32 idx = rx->jac_index[p][e];
                jac[idx] = lane_fmadd(lane_splat((V *)0, (f64)rx->net_coef[e]), d, jac[idx]);
            }
        }
    }
}

// a = I*ghinv - jac, then in-place no-pivot LU (L unit-diagonal, both in a).
template <class V>
SIM_HD void kin_decomp(const KinMech *m, const V *jac, V ghinv, V *a) {
    for (u32 i = 0; i < m->nnz; ++i) a[i] = -jac[i];
    for (u32 s = 0; s < m->species_count; ++s) a[m->diag[s]] = a[m->diag[s]] + ghinv;
    for (u32 d = 0; d < m->lu_div_count; ++d) {
        KinLuDiv div = m->lu_div[d];
        V l = a[div.ik] / a[div.kk];
        a[div.ik] = l;
        for (u32 s = div.sub_begin; s < div.sub_end; ++s) {
            KinLuSub sub = m->lu_sub[s];
            a[sub.ij] = lane_fnmadd(l, a[sub.kj], a[sub.ij]);
        }
    }
}

// Solves (LU) x = b in place.
template <class V>
SIM_HD void kin_solve(const KinMech *m, const V *a, V *x) {
    const u32 n = m->species_count;
    for (u32 i = 0; i < n; ++i) {
        for (u32 e = m->lower_begin[i]; e < m->lower_begin[i + 1]; ++e) {
            KinRowEntry le = m->lower[e];
            x[i] = lane_fnmadd(a[le.index], x[le.col], x[i]);
        }
    }
    for (u32 ii = n; ii-- > 0;) {
        for (u32 e = m->upper_begin[ii]; e < m->upper_begin[ii + 1]; ++e) {
            KinRowEntry ue = m->upper[e];
            x[ii] = lane_fnmadd(a[ue.index], x[ue.col], x[ii]);
        }
        x[ii] = x[ii] / a[m->diag[ii]];
    }
}

// (1/err)^(1/order) * safety, clamped to [FAC_MIN, FAC_MAX]. No libm.
template <class V>
SIM_HD V kin_step_factor(V err, u32 order) {
    V lo = lane_splat((V *)0, 1e-4), hi = lane_splat((V *)0, 1e4);
    V e = lane_min(lane_max(err, lo), hi);
    V root;
    if (order == 2)      root = lane_sqrt(e);
    else if (order == 4) root = lane_sqrt(lane_sqrt(e));
    else                 root = lane_cbrt_clamped(e);
    V fac = lane_splat((V *)0, KIN_FAC_SAFE) / root;
    return lane_min(lane_max(fac, lane_splat((V *)0, KIN_FAC_MIN)), lane_splat((V *)0, KIN_FAC_MAX));
}

// Integrates every lane from t to t_end (per lane). On return t == t_end for
// every lane that did not fail; h holds the step size to continue with.
template <class V>
SIM_HD void kin_integrate(const KinMech *m, const KinMethod *method, const KinControl *ctl, const V *k,
                          const V *src, V *y, V &t, V t_end, V &h, KinLaneStats<V> &stats) {
    typedef typename LaneTraits<V>::Mask M;
    const u32 n = m->species_count;
    const V zero = lane_splat((V *)0, 0.0), one = lane_splat((V *)0, 1.0);
    const V h_min = lane_splat((V *)0, ctl->h_min);
    const V rtol = lane_splat((V *)0, ctl->rtol), atol = lane_splat((V *)0, ctl->atol);
    const V inv_n = lane_splat((V *)0, 1.0 / (f64)n);

    V jac[KIN_MAX_NNZ];
    V lu[KIN_MAX_NNZ];
    V kstage[KIN_MAX_STAGES][KIN_MAX_SPECIES];
    V fcn[KIN_MAX_SPECIES], ytmp[KIN_MAX_SPECIES];

    M failed = lane_gt(stats.failed, zero);
    M active = lane_and(lane_lt(t, t_end), lane_not(failed));
    M reject_last = lane_mask_splat((M *)0, false);
    M reject_more = lane_mask_splat((M *)0, false);
    h = lane_select(lane_gt(h, zero), h, lane_splat((V *)0, ctl->h_start));

    u32 iterations = 0;
    while (lane_any(active)) {
        if (++iterations > ctl->max_steps) {
            stats.failed = lane_select(active, one, stats.failed);
            break;
        }
        V remaining = t_end - t;
        V h_cap = remaining;
        if (ctl->h_max > 0.0) h_cap = lane_min(h_cap, lane_splat((V *)0, ctl->h_max));
        V h_wanted = lane_max(h, h_min);                       // step the controller asked for
        M last = lane_ge(h_wanted, remaining);
        V hs = lane_select(last, remaining, lane_min(h_wanted, h_cap));
        hs = lane_select(active, hs, one);                     // keep idle lanes finite

        kin_fun(m, k, src, y, fcn);
        kin_jac(m, k, y, jac);
        V ghinv = one / (hs * lane_splat((V *)0, method->gamma0));
        kin_decomp(m, jac, ghinv, lu);
        V hinv = one / hs;

        for (u32 st = 0; st < method->stages; ++st) {
            u32 base = st * (st - 1) / 2;   // packed (st, j) offset; unused when st == 0
            if (st > 0 && method->new_f[st]) {
                for (u32 s = 0; s < n; ++s) ytmp[s] = y[s];
                for (u32 j = 0; j < st; ++j) {
                    f64 a = method->a[base + j];
                    if (a == 0.0) continue;
                    V av = lane_splat((V *)0, a);
                    for (u32 s = 0; s < n; ++s) ytmp[s] = lane_fmadd(av, kstage[j][s], ytmp[s]);
                }
                kin_fun(m, k, src, ytmp, fcn);
            }
            for (u32 s = 0; s < n; ++s) kstage[st][s] = fcn[s];
            for (u32 j = 0; j < st; ++j) {
                f64 c = method->c[base + j];
                if (c == 0.0) continue;
                V hc = lane_splat((V *)0, c) * hinv;
                for (u32 s = 0; s < n; ++s) kstage[st][s] = lane_fmadd(hc, kstage[j][s], kstage[st][s]);
            }
            kin_solve(m, lu, kstage[st]);
        }

        // y_new into ytmp, error norm into err
        V err = zero;
        for (u32 s = 0; s < n; ++s) {
            V ynew = y[s], yerr = zero;
            for (u32 st = 0; st < method->stages; ++st) {
                if (method->m[st] != 0.0) ynew = lane_fmadd(lane_splat((V *)0, method->m[st]), kstage[st][s], ynew);
                if (method->e[st] != 0.0) yerr = lane_fmadd(lane_splat((V *)0, method->e[st]), kstage[st][s], yerr);
            }
            V scale = lane_fmadd(rtol, lane_max(lane_abs(y[s]), lane_abs(ynew)), atol);
            V q = yerr / scale;
            err = lane_fmadd(q, q, err);
            ytmp[s] = ynew;
        }
        err = lane_sqrt(err * inv_n);
        M finite = lane_is_finite(err);
        err = lane_select(finite, err, lane_splat((V *)0, 1e10));

        V fac = kin_step_factor(err, method->order);
        M ok = lane_or(lane_le(err, one), lane_le(hs, h_min));
        if (ctl->fixed_step) {
            ok = lane_mask_splat((M *)0, true);
            fac = one;
        }
        M accept = lane_and(active, ok);
        M reject = lane_and(active, lane_not(ok));

        for (u32 s = 0; s < n; ++s) {
            V ynew = ytmp[s];
            if (ctl->clamp_negative) ynew = lane_max(ynew, zero);
            y[s] = lane_select(accept, ynew, y[s]);
        }
        t = lane_select(accept, lane_select(last, t_end, t + hs), t);

        // Next step size. After an accepted step that was shortened to land
        // on t_end, do not let the short step shrink h if its error was fine.
        V h_acc = hs * fac;
        h_acc = lane_select(reject_last, lane_min(h_acc, hs), h_acc);
        h_acc = lane_select(lane_and(last, lane_ge(fac, one)), lane_max(h_acc, h_wanted), h_acc);
        V h_rej = lane_select(reject_more, hs * lane_splat((V *)0, KIN_FAC_REJ), lane_min(hs * fac, hs));
        if (ctl->fixed_step) { h_acc = h_wanted; h_rej = h_wanted; }
        h = lane_select(accept, h_acc, lane_select(reject, h_rej, h));

        reject_more = lane_select_mask(active, lane_and(reject, reject_last), reject_more);
        reject_last = lane_select_mask(active, reject, reject_last);
        stats.steps = stats.steps + lane_select(accept, one, zero);
        stats.rejects = stats.rejects + lane_select(reject, one, zero);
        active = lane_and(lane_lt(t, t_end), lane_not(lane_gt(stats.failed, zero)));
    }
}

// Advances [t0, t1] with optional discharge pulses: an instantaneous source
// `pulse` added at t = (pulse_index + 1) * period (period <= 0: no pulses).
// Integration stops exactly on each pulse, applies it, restarts with h_start.
template <class V>
SIM_HD void kin_advance(const KinMech *m, const KinMethod *method, const KinControl *ctl, const V *k,
                        const V *src, const V *pulse, V period, V *y, V t0, V t1, V &h, V &pulse_index,
                        KinLaneStats<V> &stats) {
    typedef typename LaneTraits<V>::Mask M;
    const u32 n = m->species_count;
    const V zero = lane_splat((V *)0, 0.0), one = lane_splat((V *)0, 1.0);
    const V never = lane_splat((V *)0, 1e300);
    M pulsed = lane_gt(period, zero);
    V t = t0;
    for (;;) {
        V next_pulse = lane_select(pulsed, (pulse_index + one) * period, never);
        V seg_end = lane_min(next_pulse, t1);
        kin_integrate(m, method, ctl, k, src, y, t, seg_end, h, stats);

        M alive = lane_not(lane_gt(stats.failed, zero));
        M hit = lane_and(lane_and(pulsed, alive), lane_and(lane_le(next_pulse, t), lane_le(next_pulse, t1)));
        if (lane_any(hit)) {
            for (u32 s = 0; s < n; ++s) y[s] = y[s] + lane_select(hit, pulse[s], zero);
            pulse_index = pulse_index + lane_select(hit, one, zero);
            stats.pulses = stats.pulses + lane_select(hit, one, zero);
            h = lane_select(hit, lane_splat((V *)0, ctl->h_start), h);
        }
        M done = lane_or(lane_ge(t, t1), lane_not(alive));
        if (lane_all(done)) break;
    }
}

// ----------------------------------------------------------------------------
// Method coefficients (KPP Rosenbrock family, Sandu et al., Atmos. Env. 1997).

inline KinMethod kin_method(KinMethodId id) {
    KinMethod r = {};
    switch (id) {
        case KinMethod_Ros2: {
            f64 g = 1.0 + 1.0 / 1.4142135623730950488;
            r.name = "ros2";
            r.stages = 2; r.order = 2;
            r.a[0] = 1.0 / g;
            r.c[0] = -2.0 / g;
            r.m[0] = 3.0 / (2.0 * g); r.m[1] = 1.0 / (2.0 * g);
            r.e[0] = 1.0 / (2.0 * g); r.e[1] = 1.0 / (2.0 * g);
            r.gamma0 = g;
            r.new_f[0] = r.new_f[1] = 1;
        } break;
        case KinMethod_Ros3: {
            r.name = "ros3";
            r.stages = 3; r.order = 3;
            r.a[0] = 1.0; r.a[1] = 1.0; r.a[2] = 0.0;
            r.c[0] = -0.10156171083877702091975600115545e+01;
            r.c[1] =  0.40759956452537699824805835358067e+01;
            r.c[2] =  0.92076794298330791242156818474003e+01;
            r.m[0] =  0.1e+01;
            r.m[1] =  0.61697947043828245592553615689730e+01;
            r.m[2] = -0.42772256543218573326238373806514e+00;
            r.e[0] =  0.5e+00;
            r.e[1] = -0.29079558716805469821718236208017e+01;
            r.e[2] =  0.22354069897811569627360909276199e+00;
            r.gamma0 = 0.43586652150845899941601945119356;
            r.new_f[0] = 1; r.new_f[1] = 1; r.new_f[2] = 0;
        } break;
        case KinMethod_Rodas3: {
            r.name = "rodas3";
            r.stages = 4; r.order = 3;
            r.a[0] = 0.0; r.a[1] = 2.0; r.a[2] = 0.0; r.a[3] = 2.0; r.a[4] = 0.0; r.a[5] = 1.0;
            r.c[0] = 4.0; r.c[1] = 1.0; r.c[2] = -1.0; r.c[3] = 1.0; r.c[4] = -1.0; r.c[5] = -8.0 / 3.0;
            r.m[0] = 2.0; r.m[1] = 0.0; r.m[2] = 1.0; r.m[3] = 1.0;
            r.e[0] = 0.0; r.e[1] = 0.0; r.e[2] = 0.0; r.e[3] = 1.0;
            r.gamma0 = 0.5;
            r.new_f[0] = r.new_f[1] = r.new_f[2] = r.new_f[3] = 1;
        } break;
        default: break;
    }
    return r;
}

#endif // KIN_SOLVER_H
