// NFT hydraulics: pump -> supply riser -> manifold -> feed taps -> channels,
// plus a mixing eductor on the pump discharge and the gravity return.
// Scalar f64; solved when the network or the water temperature changes
// (hourly), not per step. Included by nft.cpp.
//
//   friction    Darcy-Weisbach. Laminar f = 64/Re (Re < 2300), turbulent
//               Colebrook-White solved by Newton iteration to machine
//               precision (Re > 4000), linear blend in between.
//   water       viscosity (Vogel) and density (Kell / Tanaka) vs temperature.
//   pump        H(Q) = H0 (1 - (Q/Qmax)^2), efficiency eta(Q) = eta_max 4x(1-x)
//               (x = Q/Qmax, floored); shaft power rho g Q H / eta ends up as
//               heat in the solution (submersible pump).
//   manifold    end-fed, dead-ended, one tap per channel at the channel pitch.
//               Each tap discharges to atmosphere through its feed tube (or
//               emitter): H_tap = (K + f L/d + 1) v^2 / 2g. Marched from the
//               dead end back to the inlet; outer bisection on the dead-end
//               head until the pump curve balances supply losses + lift.
//   eductor     nozzle on the pump discharge: Qm = Cd A sqrt(2 g H_d),
//               circulates Qm (1 + entrainment ratio) inside the tank.
//   return      gravity drain, full-pipe Manning capacity (n = 0.009).
//   film        per channel: film Reynolds q/nu picks laminar sheet flow
//               h = (3 nu q / (g S))^(1/3) (Re < 400) or Manning h = (n q / S^1/2)^(3/5)
//               (Re > 1600), blended in log space between.

#define NFT_PVC_ROUGHNESS_M 1.5e-6
#define NFT_MANNING_PVC 0.009

// PVC pressure pipe, ASTM D1785 inside diameters (inches -> m).
enum NftPipeSize { NftPipe_1_2, NftPipe_3_4, NftPipe_1, NftPipe_1_1_4, NftPipe_1_1_2, NftPipe_2, NftPipe_2_1_2,
                   NftPipe_3, NftPipe_4, NftPipe_Count };
global const char *const nft_pipe_names[NftPipe_Count] = { "1/2\"", "3/4\"", "1\"", "1-1/4\"", "1-1/2\"", "2\"", "2-1/2\"", "3\"", "4\"" };
global const f64 nft_pipe_id_in[2][NftPipe_Count] = {
    { 0.622, 0.824, 1.049, 1.380, 1.610, 2.067, 2.469, 3.068, 4.026 },   // Schedule 40
    { 0.546, 0.742, 0.957, 1.278, 1.500, 1.939, 2.323, 2.900, 3.826 },   // Schedule 80
};
internal f64 nft_pipe_id_m(u32 schedule80, u32 size) { return nft_pipe_id_in[schedule80 ? 1 : 0][size % NftPipe_Count] * 0.0254; }

struct NftPumpModel { const char *name; f32 h0_m, qmax_L_min, eta_max; };
enum NftPumpId { NftPump_400, NftPump_800, NftPump_1200, NftPump_2000, NftPump_3200, NftPump_Count };
global const NftPumpModel nft_pumps[NftPump_Count] = {   // typical submersible curves (GPH at zero head)
    { "400 GPH submersible",  2.4f,  25.0f, 0.25f },
    { "800 GPH submersible",  3.6f,  50.0f, 0.30f },
    { "1200 GPH submersible", 4.6f,  76.0f, 0.33f },
    { "2000 GPH submersible", 6.0f, 126.0f, 0.38f },
    { "3200 GPH submersible", 7.0f, 200.0f, 0.42f },
};

// Standard NFT channel profiles: wetted floor width and outer perimeter.
struct NftProfile { const char *name; f32 width_m, perimeter_m, height_m; };
enum NftProfileId { NftProfile_70x40, NftProfile_100x50, NftProfile_120x60, NftProfile_Count };
global const NftProfile nft_profiles[NftProfile_Count] = {
    { "70 x 40 mm",  0.070f, 0.22f, 0.040f },
    { "100 x 50 mm", 0.100f, 0.30f, 0.050f },
    { "120 x 60 mm", 0.120f, 0.36f, 0.060f },
};

struct NftHydraulicParams {
    u32 pump;                    // NftPumpId
    u32 schedule80;              // 0: Sch 40, 1: Sch 80
    u32 supply_size;             // NftPipeSize
    f32 supply_length_m;
    f32 supply_k;                // sum of fitting losses (elbows, valve, check valve, tee run)
    f32 lift_m;                  // manifold above the tank water level
    u32 manifold_size;
    f32 channel_pitch_m;         // tap spacing along the manifold
    f32 tap_id_mm;               // feed tube / emitter bore
    f32 tap_length_m;
    f32 tap_k;                   // tee branch + entrance + valve
    b32 balance_taps;            // per-tap restrictor sized for equal flow
    b32 eductor_on;
    f32 eductor_nozzle_mm, eductor_cd, eductor_ratio;   // entrained : motive
    u32 return_size;
    f32 return_slope;
    u32 profile;                 // NftProfileId
    f32 slope[2];                // channel grade per bank (m/m)
};

internal NftHydraulicParams nft_hydraulics_defaults() {
    NftHydraulicParams h = {};
    h.pump = NftPump_800;
    h.schedule80 = 0;
    h.supply_size = NftPipe_3_4;
    h.supply_length_m = 2.0f;
    h.supply_k = 4 * 0.9f + 0.2f + 2.0f;   // 4 elbows, ball valve, check valve
    h.lift_m = 1.0f;
    h.manifold_size = NftPipe_3_4;
    h.channel_pitch_m = 0.15f;
    h.tap_id_mm = 4.3f;                    // 1/4" OD drip tube
    h.tap_length_m = 0.4f;
    h.tap_k = 1.8f + 0.5f + 1.0f;          // tee branch, entrance, barb
    h.balance_taps = 0;
    h.eductor_on = 1;
    h.eductor_nozzle_mm = 6.0f;
    h.eductor_cd = 0.90f;
    h.eductor_ratio = 4.0f;                // standard 1:4 (5x total circulation)
    h.return_size = NftPipe_2;
    h.return_slope = 0.02f;
    h.profile = NftProfile_100x50;
    h.slope[0] = h.slope[1] = 0.02f;
    return h;
}

// --- water --------------------------------------------------------------------------

internal f64 nft_water_mu(f64 T_C) {                     // Pa s (Vogel)
    return 2.414e-5 * pow(10.0, 247.8 / (T_C + 273.15 - 140.0));
}
internal f64 nft_water_rho(f64 T_C) {                    // kg/m^3 (Tanaka et al. 2001)
    f64 d = T_C - 3.983035;
    return 999.974950 * (1.0 - (T_C + 301.797) / (522528.9 * (T_C + 69.34881)) * d * d);
}

// --- friction ---------------------------------------------------------------------

internal f64 nft_colebrook(f64 Re, f64 rel_rough) {
    // Swamee-Jain start, then Newton on g(x) = x + 2 log10(e/3.7 + 2.51 x / Re), x = 1/sqrt(f)
    f64 sj = 0.25 / pow(log10(rel_rough / 3.7 + 5.74 / pow(Re, 0.9)), 2.0);
    f64 x = 1.0 / sqrt(sj);
    for (u32 it = 0; it < 30; ++it) {
        f64 a = rel_rough / 3.7 + 2.51 * x / Re;
        f64 g = x + 2.0 * log10(a);
        f64 dg = 1.0 + 2.0 / (a * 2.302585092994046) * (2.51 / Re);
        f64 step = g / dg;
        x -= step;
        if (fabs(step) < 1e-15 * x) break;
    }
    return 1.0 / (x * x);
}

internal f64 nft_friction(f64 Re, f64 d) {
    if (Re <= 0.0) return 0.0;
    f64 rel = NFT_PVC_ROUGHNESS_M / d;
    if (Re < 2300.0) return 64.0 / Re;
    if (Re > 4000.0) return nft_colebrook(Re, rel);
    f64 t = (Re - 2300.0) / 1700.0;
    return (1.0 - t) * (64.0 / 2300.0) + t * nft_colebrook(4000.0, rel);
}

// Head loss (m) for flow q (m^3/s) through a pipe of length L, bore d, minor K.
internal f64 nft_pipe_loss(f64 q, f64 d, f64 L, f64 K, f64 nu) {
    if (q <= 0.0) return 0.0;
    f64 A = 0.25 * 3.14159265358979 * d * d;
    f64 v = q / A;
    f64 f = nft_friction(v * d / nu, d);
    return (f * L / d + K) * v * v / (2.0 * NFT_G);
}

// Flow through a tap discharging to atmosphere at head H (m):
// H = (K + 1 + f L/d) v^2 / 2g. Laminar (f = 64 nu / (v d)) is a quadratic in
// v, solved exactly; otherwise fixed-point on f(Re(v)), which contracts fast
// because f varies slowly with v.
internal f64 nft_tap_flow(f64 H, f64 d, f64 L, f64 K, f64 nu) {
    if (H <= 0.0) return 0.0;
    f64 A = 0.25 * 3.14159265358979 * d * d;
    f64 a = K + 1.0, b = 64.0 * nu * L / (d * d), c = -2.0 * NFT_G * H;
    f64 v = (-b + sqrt(b * b - 4.0 * a * c)) / (2.0 * a);       // laminar root
    if (v * d / nu < 2300.0) return v * A;
    for (u32 it = 0; it < 50; ++it) {
        f64 f = nft_friction(v * d / nu, d);
        f64 vn = sqrt(2.0 * NFT_G * H / (K + 1.0 + f * L / d));
        if (fabs(vn - v) < 1e-15 * vn) { v = vn; break; }
        v = vn;
    }
    return v * A;
}

internal f64 nft_pump_head(const NftPumpModel *p, f64 q_m3s) {
    f64 x = q_m3s * 60000.0 / p->qmax_L_min;
    return p->h0_m * (1.0 - x * x);
}

internal f64 nft_pump_eta(const NftPumpModel *p, f64 q_m3s) {
    f64 x = Clamp(q_m3s * 60000.0 / p->qmax_L_min, 0.0, 1.0);
    return Maximum(p->eta_max * 4.0 * x * (1.0 - x), 0.05);
}

// --- network ----------------------------------------------------------------------

struct NftHydraulics {
    b32 ok;                      // the pump reaches the manifold
    f64 nu, rho;
    f64 q_pump, q_manifold, q_eductor_motive, q_eductor_total;   // m^3/s
    f64 head_pump, head_manifold;                                  // m
    f64 pump_power_W, pump_heat_W, pump_eta;
    f64 q_channel[8192];         // m^3/s per channel
    f64 head_tap[8192];          // head at each tap (m)
    f64 q_min, q_max, q_mean;
    f64 return_capacity, return_use;   // m^3/s, fraction
    f64 tap_extra_k[8192];       // balancing restrictors (when balance_taps)
    f64 tank_turnover_per_h;
    u32 iterations;
};

// March the manifold from the dead end: given the dead-end tap head, returns
// the manifold inlet head and fills per-tap flows; returns the inlet flow.
internal f64 nft_manifold_march(NftHydraulicParams *hp, NftHydraulics *h, u32 taps, f64 H_end, f64 *H_inlet) {
    f64 d_m = nft_pipe_id_m(hp->schedule80, hp->manifold_size);
    f64 d_t = hp->tap_id_mm * 1e-3;
    f64 H = H_end, q_carry = 0.0;
    for (u32 i = taps; i-- > 0;) {
        f64 k = hp->tap_k + (hp->balance_taps ? h->tap_extra_k[i] : 0.0);
        f64 q = nft_tap_flow(H, d_t, hp->tap_length_m, k, h->nu);
        h->q_channel[i] = q;
        h->head_tap[i] = H;
        q_carry += q;
        if (i > 0) H += nft_pipe_loss(q_carry, d_m, hp->channel_pitch_m, 0.0, h->nu);   // segment toward the inlet (tee run ~0)
    }
    *H_inlet = H;
    return q_carry;
}

// Solve the operating point on the dead-end head: the residual
// H_pump(Q_total) - H_discharge decreases monotonically with it.
internal void nft_hydraulics_solve(NftHydraulicParams *hp, u32 taps, f64 T_C, NftHydraulics *h) {
    taps = Clamp(taps, 1u, 8192u);
    const NftPumpModel *pump = nft_pumps + (hp->pump % NftPump_Count);
    h->rho = nft_water_rho(T_C);
    h->nu = nft_water_mu(T_C) / h->rho;
    f64 d_s = nft_pipe_id_m(hp->schedule80, hp->supply_size);
    f64 a_e = 0.25 * 3.14159265358979 * (hp->eductor_nozzle_mm * 1e-3) * (hp->eductor_nozzle_mm * 1e-3);

    // Balancing: restrictors that make every tap see the last tap's resistance at
    // equal flow; computed once from an unbalanced solve's head profile.
    struct Eval { f64 residual, q_m, q_e, H_d, H_m; };
    auto evaluate = [&](f64 H_end) -> Eval {
        Eval e;
        e.q_m = nft_manifold_march(hp, h, taps, H_end, &e.H_m);
        e.H_d = hp->lift_m + nft_pipe_loss(e.q_m, d_s, hp->supply_length_m, hp->supply_k, h->nu) + e.H_m;
        e.q_e = hp->eductor_on ? hp->eductor_cd * a_e * sqrt(2.0 * NFT_G * Maximum(e.H_d, 0.0)) : 0.0;
        e.residual = nft_pump_head(pump, e.q_m + e.q_e) - e.H_d;
        return e;
    };
    Eval lo_e = evaluate(0.0);
    h->ok = lo_e.residual > 0.0;
    Eval e = lo_e;
    u32 it = 0;
    if (h->ok) {
        // Illinois regula falsi on the dead-end head: bracketed like bisection,
        // superlinear like secant.
        f64 a = 0.0, fa = lo_e.residual;
        f64 b = pump->h0_m;
        Eval eb = evaluate(b);
        f64 fb = eb.residual;
        int side = 0;
        for (it = 0; it < 100; ++it) {
            f64 x = (a * fb - b * fa) / (fb - fa);
            e = evaluate(x);
            f64 fx = e.residual;
            if (fabs(fx) < 1e-12 || fabs(b - a) < 1e-14 * Maximum(fabs(b), 1e-6)) break;
            if (fx * fb > 0.0) {
                b = x; fb = fx;
                if (side == -1) fa *= 0.5;
                side = -1;
            } else {
                a = x; fa = fx;
                if (side == 1) fb *= 0.5;
                side = 1;
            }
        }
    } else {
        for (u32 i = 0; i < taps; ++i) { h->q_channel[i] = 0.0; h->head_tap[i] = 0.0; }
        e.q_m = e.q_e = 0.0; e.H_d = 0.0; e.H_m = 0.0;
    }
    h->iterations = it;
    h->q_manifold = e.q_m;
    h->q_eductor_motive = e.q_e;
    h->q_eductor_total = e.q_e * (1.0 + hp->eductor_ratio);
    h->q_pump = e.q_m + e.q_e;
    h->head_pump = e.H_d;
    h->head_manifold = e.H_m;
    h->pump_eta = nft_pump_eta(pump, h->q_pump);
    h->pump_power_W = h->q_pump > 0.0 ? h->rho * NFT_G * h->q_pump * h->head_pump / h->pump_eta : 0.0;
    h->pump_heat_W = h->pump_power_W;   // submersible: motor + hydraulic losses all heat the solution
    h->q_min = 1e30; h->q_max = 0.0; h->q_mean = 0.0;
    for (u32 i = 0; i < taps; ++i) {
        h->q_min = Minimum(h->q_min, h->q_channel[i]);
        h->q_max = Maximum(h->q_max, h->q_channel[i]);
        h->q_mean += h->q_channel[i];
    }
    h->q_mean /= (f64)taps;
    f64 d_r = nft_pipe_id_m(hp->schedule80, hp->return_size);
    f64 A_r = 0.25 * 3.14159265358979 * d_r * d_r, R_h = d_r / 4.0;
    h->return_capacity = A_r * pow(R_h, 2.0 / 3.0) * sqrt(Maximum((f64)hp->return_slope, 1e-6)) / NFT_MANNING_PVC;
    h->return_use = h->return_capacity > 0.0 ? h->q_manifold / h->return_capacity : 0.0;
}

// Restrictor K per tap so every tap passes the mean flow at its own head
// (what a grower does with inline valves): K_extra = 2g H / v^2 - (K + 1 + fL/d).
internal void nft_hydraulics_balance(NftHydraulicParams *hp, u32 taps, f64 T_C, NftHydraulics *h) {
    b32 was = hp->balance_taps;
    hp->balance_taps = 0;
    nft_hydraulics_solve(hp, taps, T_C, h);
    hp->balance_taps = was;
    if (!was || !h->ok) return;
    // target: the weakest tap's flow (no restrictor there), others throttled to it
    f64 d_t = hp->tap_id_mm * 1e-3, A = 0.25 * 3.14159265358979 * d_t * d_t;
    for (u32 pass = 0; pass < 6; ++pass) {
        f64 target = h->q_min;
        for (u32 i = 0; i < taps; ++i) {
            f64 v = target / A;
            f64 f = nft_friction(v * d_t / h->nu, d_t);
            f64 k = 2.0 * NFT_G * h->head_tap[i] / (v * v) - (hp->tap_k + 1.0 + f * hp->tap_length_m / d_t);
            h->tap_extra_k[i] = Maximum(k, 0.0);
        }
        nft_hydraulics_solve(hp, taps, T_C, h);
    }
}

// Film depth (m) for flow q (m^3/s) on a channel of width w and grade S.
internal f64 nft_film_depth(f64 q, f64 w, f64 S, f64 nu) {
    f64 qu = q / w;                                     // per unit width
    f64 Re = qu / nu;
    f64 S_ = Maximum(S, 1e-5);
    f64 h_lam = cbrt(3.0 * nu * qu / (NFT_G * S_));
    f64 h_turb = pow(NFT_MANNING_PVC * qu / sqrt(S_), 0.6);
    if (Re <= 400.0) return h_lam;
    if (Re >= 1600.0) return h_turb;
    f64 t = log(Re / 400.0) / log(4.0);
    return exp((1.0 - t) * log(h_lam) + t * log(h_turb));
}
