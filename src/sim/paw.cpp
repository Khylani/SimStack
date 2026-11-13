// PAW skid: three coaxial DBD modules submerged in a batch drum, gas through a
// venturi on the recirculation loop, off-gas through a stage-2 bucket of the
// next batch's source water. Reference design: "DBD Reactor Design Review -
// 50 gal PAW" and the "PAW Plasma Skid" page (Nov 2026).
//
//   electrical  coaxial DBD = dielectric (C_d) in series with the gas gap (C_g);
//               Manley power 4 f C_d U_b (U_0 - U_b (C_d + C_g) / C_d) x a
//               measured/ideal factor, capped by the flyback and half-bridge
//   gas         SEI = P / air flow sets the regime: ozone below ~0.3 kJ/L,
//               NOx above ~1 kJ/L; NOx at an energy cost (MJ/mol), split into
//               N2O5 (-> 2 HNO3) and NO2 (2 NO2 + H2O -> HNO2 + HNO3)
//   drum        f64 well-mixed batch: nitrate, nitrite, dissolved O3, DIC
//               (CO2 exchanged with air by the venturi), untracked charge R
//               (the source water's counter-ions), temperature. pH from the
//               alkalinity by a closed-system speciation solve.
//   batch       treat to the target pH (plasma stops inside the window that
//               reaches it), hold until the ozone has decayed, then the NFT
//               reservoir tops off from the finished batch; refill from the
//               stage-2 bucket plus source water when the drum is empty.
//
// Everything that the reference design marks as an estimate (Manley factor,
// NOx energy cost, capture) is a parameter to calibrate against a Lissajous
// power measurement and nitrate strips. Included by nft.cpp.

#define PAW_ALUMINA_OD_MM 22.225     // 8746K family alumina, 7/8" x 5/8"
#define PAW_ALUMINA_ID_MM 15.875
#define PAW_EPS0          8.8541878128e-12
#define PAW_N_MG          14.007

global const char *const paw_pipe_names[2] = { "Sch 10 pipe, 2.8 mm gap", "Sch 5 pipe, 3.9 mm gap" };
global const f32 paw_gap_mm[2] = { 2.8f, 3.9f };

enum PawGasSource { PawGas_Design, PawGas_Kinetics, PawGas_Count };
global const char *const paw_gas_source_names[PawGas_Count] = { "design estimate (SEI regime + MJ/mol)", "M2 air kinetics (module annulus)" };

struct PawParams {
    b32 enabled;
    u32 gas_source;             // PawGasSource
    b32 gas_T_manual;           // 0: gap temperature from the module heat balance; 1: water + gas_dT_K
    f32 gas_dT_K;               // manual gap gas temperature above the drum water
    f32 reduced_field_Td;       // micro-discharge E/N for the Boltzmann G-values (kinetics source)
    f32 filament_mJ_cm3;        // micro-discharge energy density (kinetics source)
    f32 eps_alumina, eps_pipe;  // thermal emissivities across the gap
    f32 h_water_W_m2K;          // pipe outside to the stirred drum water
    // modules
    u32 modules;
    u32 pipe_sch5;              // 0: Sch 10 (2.8 mm gap), 1: Sch 5 (3.9 mm)
    f32 active_mm;              // pipe over alumina (both electrodes the same length)
    f32 eps_r;                  // alumina
    f32 bd_kV_per_mm, bd_offset_kV;   // gap breakdown (burning) voltage, uniform-field estimate
    f32 manley_factor;          // measured / ideal Manley power (Lissajous); 0.6-0.8 expected
    // drive
    f32 v_peak_kV, freq_kHz, bus_V;
    f32 burst_duty;             // average power control (bursts), 0..1
    f32 chain_eff;              // half-bridge + flyback
    f32 flyback_W_max, bus_A_max;
    // gas
    f32 air_L_min;              // total dry air through the annuli
    f32 nox_MJ_mol;             // energy cost of fixed N in NOx mode (15-30 literature)
    f32 o3_g_kWh;               // ozone-mode yield (SEI < 0.3 kJ/L)
    f32 o3_nox_g_kWh;           // ozone left in the NOx-mode effluent
    f32 n2o5_share;             // fraction of fixed N leaving as N2O5 (nitrate only)
    // capture
    f32 capture_lo, capture_hi; // venturi NOx capture at pH <= 6 / >= 8 (2 NO2 + 2 OH- absorbs faster)
    f32 o3_capture;             // venturi ozone transfer
    b32 stage2_on;
    f32 stage2_L, stage2_no2, stage2_o3;   // bucket of tomorrow's source water: its capture fractions
    // water
    f32 drum_L;
    f32 source_pH, source_kh_meq;          // municipal water: pH, carbonate alkalinity (KH)
    f32 source_T_C;
    f32 batch_pH;               // treat to this pH
    f32 hold_h, o3_ready_mg_L;  // hold before use: at least hold_h and until O3(aq) is below this
    f32 kla_co2_per_s;          // CO2 exchange with air in the drum (venturi bubbles)
    f32 heat_frac;              // share of plasma power that ends up in the drum water
    f32 pump_W, drum_ua_W_K;
};

internal PawParams paw_default_params() {
    PawParams p = {};
    p.enabled = 0;
    p.gas_source = PawGas_Design;
    p.gas_T_manual = 0;
    p.gas_dT_K = 30.0f;
    p.reduced_field_Td = 180.0f;   // air DBD filaments: 150-250 Td
    p.filament_mJ_cm3 = 30.0f;     // a few uJ in a ~100 um filament across the 2.8 mm gap
    p.eps_alumina = 0.7f;      // alumina, 400-600 K
    p.eps_pipe = 0.4f;         // 316, lightly oxidized inside
    p.h_water_W_m2K = 500.0f;  // recirculated drum
    p.modules = 3;
    p.pipe_sch5 = 0;
    p.active_mm = 200.0f;
    p.eps_r = 9.6f;
    p.bd_kV_per_mm = 3.0f;
    p.bd_offset_kV = 1.3f;
    p.manley_factor = 0.7f;
    p.v_peak_kV = 12.5f;
    p.freq_kHz = 6.5f;
    p.bus_V = 120.0f;
    p.burst_duty = 1.0f;
    p.chain_eff = 0.75f;
    p.flyback_W_max = 300.0f;
    p.bus_A_max = 5.0f;
    p.air_L_min = 6.0f;
    p.nox_MJ_mol = 22.0f;
    p.o3_g_kWh = 30.0f;
    p.o3_nox_g_kWh = 1.0f;
    p.n2o5_share = 0.6f;
    p.capture_lo = 0.75f;
    p.capture_hi = 0.85f;
    p.o3_capture = 0.6f;
    p.stage2_on = 1;
    p.stage2_L = 18.9f;        // 5 gal
    p.stage2_no2 = 0.7f;
    p.stage2_o3 = 0.3f;
    p.drum_L = 189.3f;         // 50 gal batch in the 55 gal drum
    p.source_pH = 10.0f;
    p.source_kh_meq = 1.5f;
    p.source_T_C = 18.0f;
    p.batch_pH = 5.5f;
    p.hold_h = 1.5f;
    p.o3_ready_mg_L = 0.05f;
    p.kla_co2_per_s = 2e-3f;
    p.heat_frac = 0.9f;
    p.pump_W = 60.0f;
    p.drum_ua_W_K = 8.0f;      // HDPE drum, ~2.4 m^2 at ~3.5 W/m^2K
    return p;
}

// --- electrical -------------------------------------------------------------------

struct PawElectrical {
    f64 Cd_F, Cg_F;            // per module
    f64 Vb_V, Vign_V;          // gap burning voltage; applied ignition voltage (peak)
    f64 P_ideal_W;             // Manley, per module
    f64 P_on_W;                // delivered while bursting, all modules (after factor and supply limits)
    f64 P_avg_W;               // x burst duty
    f64 area_cm2, W_cm2;       // per module, at the dielectric surface
    f64 I_bus_A;
    f64 annulus_cm3;           // per module
    f64 SEI_kJ_L;
    b32 flyback_limited, bus_limited;
};

internal f64 paw_pipe_id_mm(const PawParams *p) { return PAW_ALUMINA_OD_MM + 2.0 * paw_gap_mm[p->pipe_sch5 ? 1 : 0]; }

internal PawElectrical paw_electrical(const PawParams *p) {
    PawElectrical e = {};
    f64 L = (f64)p->active_mm * 1e-3;
    f64 gap = paw_gap_mm[p->pipe_sch5 ? 1 : 0];
    f64 pipe_id = paw_pipe_id_mm(p);
    e.Cd_F = 2.0 * 3.14159265358979 * PAW_EPS0 * p->eps_r * L / log(PAW_ALUMINA_OD_MM / PAW_ALUMINA_ID_MM);
    e.Cg_F = 2.0 * 3.14159265358979 * PAW_EPS0 * L / log(pipe_id / PAW_ALUMINA_OD_MM);
    e.Vb_V = ((f64)p->bd_kV_per_mm * gap + p->bd_offset_kV) * 1e3;
    e.Vign_V = e.Vb_V * (e.Cd_F + e.Cg_F) / e.Cd_F;
    f64 V = (f64)p->v_peak_kV * 1e3, f = (f64)p->freq_kHz * 1e3;
    e.P_ideal_W = Maximum(0.0, 4.0 * f * e.Cd_F * e.Vb_V * (V - e.Vign_V));
    f64 P = e.P_ideal_W * p->manley_factor * p->modules;
    f64 bus_cap = (f64)p->chain_eff * p->bus_V * p->bus_A_max;
    if (P > p->flyback_W_max) { P = p->flyback_W_max; e.flyback_limited = 1; }
    if (P > bus_cap) { P = bus_cap; e.bus_limited = 1; }
    e.P_on_W = P;
    e.P_avg_W = P * Clamp((f64)p->burst_duty, 0.0, 1.0);
    e.area_cm2 = 3.14159265358979 * PAW_ALUMINA_OD_MM * p->active_mm / 100.0;
    e.W_cm2 = p->modules ? P / p->modules / e.area_cm2 : 0.0;
    e.I_bus_A = p->bus_V > 0.0f ? P / (p->chain_eff * p->bus_V) : 0.0;
    e.annulus_cm3 = 3.14159265358979 * 0.25 * (pipe_id * pipe_id - PAW_ALUMINA_OD_MM * PAW_ALUMINA_OD_MM) * p->active_mm * 1e-3;
    e.SEI_kJ_L = p->air_L_min > 0.0f ? e.P_avg_W / ((f64)p->air_L_min / 60.0) * 1e-3 : 0.0;
    return e;
}

// --- module heat balance ---------------------------------------------------------------
//
// The rod and the alumina have no cooling path of their own: the rod's only
// exits are the alumina and a short stem in air. At steady state the
// discharge heat (heat_frac of the power, deposited in the gap and on the
// dielectric surface) crosses the 2.8 mm air gap to the water-cooled pipe by
// conduction (coaxial, air k(T)) and thermal radiation (gray coaxial
// cylinders). Solved for the alumina surface temperature; the gas sees the
// mean of the two walls (linear profile; filament heating in the volume
// raises it slightly). The pipe sits a film resistance above the water.
// Gas flow carries ~0.04 W/K out: negligible against ~0.5 W/K across the gap.

struct PawThermal {
    f64 T_pipe_C, T_alumina_C, T_gas_K;
    f64 q_cond_W, q_rad_W;        // per module
};

internal f64 paw_air_k(f64 T_K) { return 0.0241 * pow(T_K / 273.15, 0.81); }   // W/(m K), 250-800 K

internal PawThermal paw_thermal(const PawParams *p, const PawElectrical *e, f64 water_T_C) {
    PawThermal t = {};
    f64 L = (f64)p->active_mm * 1e-3;
    f64 r1 = 0.5e-3 * PAW_ALUMINA_OD_MM, r2 = 0.5e-3 * paw_pipe_id_mm(p);
    f64 q = p->modules ? (f64)p->heat_frac * e->P_avg_W / p->modules : 0.0;
    f64 A_out = 3.14159265358979 * 0.0334 * L;                       // 1" pipe OD 33.4 mm
    f64 Tw = water_T_C + 273.15 + q / Maximum((f64)p->h_water_W_m2K * A_out, 1e-9);
    f64 A1 = 2.0 * 3.14159265358979 * r1 * L;
    f64 rad_c = 5.670374419e-8 * A1 / (1.0 / p->eps_alumina + (r1 / r2) * (1.0 / p->eps_pipe - 1.0));
    f64 lo = Tw, hi = Tw + 3000.0;
    for (u32 it = 0; it < 200 && hi - lo > 1e-9; ++it) {
        f64 Ts = 0.5 * (lo + hi);
        f64 cond = 2.0 * 3.14159265358979 * paw_air_k(0.5 * (Ts + Tw)) * L * (Ts - Tw) / log(r2 / r1);
        f64 rad = rad_c * (Ts * Ts * Ts * Ts - Tw * Tw * Tw * Tw);
        if (cond + rad > q) hi = Ts; else lo = Ts;
    }
    f64 Ts = 0.5 * (lo + hi);
    t.q_cond_W = 2.0 * 3.14159265358979 * paw_air_k(0.5 * (Ts + Tw)) * L * (Ts - Tw) / log(r2 / r1);
    t.q_rad_W = rad_c * (Ts * Ts * Ts * Ts - Tw * Tw * Tw * Tw);
    t.T_pipe_C = Tw - 273.15;
    t.T_alumina_C = Ts - 273.15;
    t.T_gas_K = 0.5 * (Ts + Tw);
    return t;
}

internal f64 paw_gas_T_K(const PawParams *p, const PawElectrical *e, f64 water_T_C) {
    if (p->gas_T_manual) return water_T_C + 273.15 + p->gas_dT_K;
    return paw_thermal(p, e, water_T_C).T_gas_K;
}

// Share of the discharge energy in the NOx regime: 0 below 0.3 kJ/L, 1 above
// 1 kJ/L, smooth in between (the reference design's regime boundaries).
internal f64 paw_nox_regime(f64 sei_kJ_L) {
    f64 x = Clamp((sei_kJ_L - 0.3) / 0.7, 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

// --- gas from the M2 kinetics -----------------------------------------------------------
//
// One module's annulus as a plug-flow reactor (= a batch reactor run for the
// residence time) at the module's average power density, dry feed, gas
// temperature = drum water + gas_dT_K, pulsed at 2 f. Run on demand (init,
// preset load, "Run chemistry"), not per window: the operating point changes
// only when the drive, flow or water temperature does.

enum PawGasSpecies { PawS_O3, PawS_NO, PawS_NO2, PawS_NO3, PawS_N2O5, PawS_HNO2, PawS_HNO3, PawS_N2O, PawS_Count };
global const char *const paw_gas_names[PawS_Count] = { "O3", "NO", "NO2", "NO3", "N2O5", "HNO2", "HNO3", "N2O" };
global const f64 paw_gas_n_atoms[PawS_Count] = { 0, 1, 1, 1, 2, 1, 1, 2 };

struct PawGas {
    b32 valid;
    f64 T_K, power_W_cm3, residence_s, freq_Hz;   // the operating point it was computed at
    f64 P_avg_W, air_L_min;                       // ... and what that came from (stale when they change)
    f64 x[PawS_Count];                            // outlet mole fractions
    f64 nox_MJ_mol;                               // plasma energy per mol of soluble N (NO2, NO3, N2O5, HNO2, HNO3, NO paired with NO2)
    f64 o3_g_kWh;
};

struct PawChem {                                  // kinetics workspace, allocated once (nft_init)
    KinModel *model;
    AirSpecies sp;
    AirSourceModel src;
    DbdZones zones;
    b32 generated;
    u32 index[PawS_Count];
};

internal void paw_chem_alloc(PawChem *c, Arena *arena) {
    c->model = push_struct(arena, KinModel);
    air_mech_build(c->model, &c->sp);
    c->src = air_source_defaults();
    c->generated = dbd_generated_matches(c->model);
    dbd_zones_alloc(&c->zones, arena, c->model, 1);
    for (u32 i = 0; i < PawS_Count; ++i) c->index[i] = (u32)kin_find_species(c->model, paw_gas_names[i]);
}

// Moles per litre of feed air at 20 C, 1 atm (rotameters read at room conditions).
#define PAW_AIR_MOL_PER_L (101325.0 / (8.314462618 * 293.15) * 1e-3)

internal void paw_gas_compute(PawChem *c, PawGas *g, const PawParams *p, f64 water_T_C) {
    memset(g, 0, sizeof(*g));
    if (!c->model || !p->modules) return;
    PawElectrical e = paw_electrical(p);
    f64 flow_cm3_s = (f64)p->air_L_min * 1000.0 / 60.0 / p->modules;
    g->T_K = paw_gas_T_K(p, &e, water_T_C);
    g->power_W_cm3 = e.P_avg_W / p->modules / e.annulus_cm3;
    g->residence_s = e.annulus_cm3 / Maximum(flow_cm3_s, 1e-9);
    g->freq_Hz = (f64)p->freq_kHz * 1e3;
    g->P_avg_W = e.P_avg_W;
    g->air_L_min = p->air_L_min;
    // LoKI-B++ G-values at the filaments' reduced field, filament-resolved pulses
    // (validated against ChemPlasKin full kinetics at these conditions: docs/M2_KINETICS.md)
    c->src.boltzmann = 1;
    c->src.reduced_field_Td = p->reduced_field_Td;
    c->src.filament_J_cm3 = (f64)p->filament_mJ_cm3 * 1e-3;
    DbdZones *z = &c->zones;
    z->T_K[0] = g->T_K; z->p_Pa[0] = 101325.0; z->x_H2O[0] = 103.0 / 101325.0;   // dryer: -20 C dew point
    z->power_W_cm3[0] = g->power_W_cm3; z->freq_Hz[0] = g->freq_Hz;
    dbd_zones_fill_padding(z);
    dbd_zones_reset(z, &c->sp);
    dbd_zones_prepare_rates(z, c->model);
    KinMethod method = kin_method(KinMethod_Rodas3);
    KinControl ctl = dbd_default_control();
    f64 dt = 1e-3, res = Minimum(g->residence_s, 30.0);
    u32 steps = (u32)ceil(res / dt);
    for (u32 i = 0; i < steps; ++i) {
        f64 a = i * dt, b = Minimum((i + 1) * dt, res);
        if (c->generated) dbd_step_cpu<f64x4, AirGenOps>(z, c->model, &c->sp, &c->src, DbdDischarge_Pulsed, &method, &ctl, a, b);
        else dbd_step_cpu<f64x4>(z, c->model, &c->sp, &c->src, DbdDischarge_Pulsed, &method, &ctl, a, b);
    }
    f64 total = 0.0;
    for (u32 s = 0; s < z->n_species; ++s) total += z->y[s * z->padded];
    for (u32 i = 0; i < PawS_Count; ++i) g->x[i] = total > 0.0 ? z->y[c->index[i] * z->padded] / total : 0.0;
    f64 mol_air_s = (f64)p->air_L_min / 60.0 * PAW_AIR_MOL_PER_L;
    f64 soluble = (g->x[PawS_NO2] + Minimum(g->x[PawS_NO], g->x[PawS_NO2]) + g->x[PawS_NO3] + 2.0 * g->x[PawS_N2O5] +
                   g->x[PawS_HNO2] + g->x[PawS_HNO3]) * mol_air_s;   // NO counts with its NO2 partner (N2O3)
    g->nox_MJ_mol = soluble > 0.0 ? e.P_avg_W / soluble * 1e-6 : 0.0;
    g->o3_g_kWh = e.P_avg_W > 0.0 ? g->x[PawS_O3] * mol_air_s * 48.0 / (e.P_avg_W / 3.6e6) : 0.0;
    g->valid = 1;
}

internal b32 paw_gas_stale(const PawGas *g, const PawParams *p) {
    if (!g->valid) return 1;
    PawElectrical e = paw_electrical(p);
    return fabs(e.P_avg_W - g->P_avg_W) > 1e-9 * Maximum(e.P_avg_W, 1.0) || fabs((f64)p->air_L_min - g->air_L_min) > 1e-9 ||
           fabs((f64)p->freq_kHz * 1e3 - g->freq_Hz) > 1e-6;
}

// --- water --------------------------------------------------------------------------

struct PawWater {              // a well-mixed batch (drum) or the stage-2 bucket
    f64 V_L, T_C;
    f64 no3_mmol, no2_mmol, o3_mmol;
    f64 dic_mmol;              // dissolved inorganic carbon
    f64 R_meq;                 // untracked charge (the source water's counter-ions)
    f64 pH;
};

// HNO2 = H+ + NO2-, pKa 3.25 at 25 C (Park & Lee 1988), van't Hoff dH = +14.6 kJ/mol.
internal f64 paw_ka_hno2(f64 T_C) { return pow(10.0, nft_vant_hoff_log(-3.25, 14.6e3, T_C + 273.15)); }

// Alkalinity (mol/L) at pH of a closed carbonate system with total DIC (mol/L),
// nitrite counted below its zero level (-[HNO2]).
internal f64 paw_alk_closed(const NftAcidBase *a, f64 ka_no2, f64 pH, f64 dic, f64 no2) {
    f64 aH = pow(10.0, -pH);
    f64 r1 = a->K1 / (aH * a->g1);                  // [HCO3-]/[CO2*]
    f64 r2 = a->K2 * a->g1 / (aH * a->g2);          // [CO3--]/[HCO3-]
    f64 co2 = dic / (1.0 + r1 * (1.0 + r2));
    f64 hco3 = co2 * r1, co3 = hco3 * r2;
    f64 oh = a->Kw / (aH * a->g1), h = aH / a->g1;
    f64 rn = ka_no2 / (aH * a->g1);                 // [NO2-]/[HNO2]
    f64 hno2 = no2 / (1.0 + rn);
    return hco3 + 2.0 * co3 + oh - h - hno2;
}

// TA of the batch (meq/L): untracked charge minus the strong anions.
internal f64 paw_ta_meq(const PawWater *w) { return w->V_L > 0.0 ? (w->R_meq - w->no3_mmol - w->no2_mmol) / w->V_L : 0.0; }

internal f64 paw_ionic_strength(const PawWater *w) {
    if (w->V_L <= 0.0) return 0.0;
    return 0.5e-3 * (fabs(w->R_meq) + w->no3_mmol + w->no2_mmol + w->dic_mmol) / w->V_L;   // monovalent, ~2x the anions
}

internal f64 paw_solve_ph(PawWater *w) {
    if (w->V_L <= 0.0) return w->pH;
    NftAcidBase a = nft_acid_base(w->T_C, paw_ionic_strength(w));
    f64 ka = paw_ka_hno2(w->T_C);
    f64 ta = paw_ta_meq(w) * 1e-3, dic = w->dic_mmol / w->V_L * 1e-3, no2 = w->no2_mmol / w->V_L * 1e-3;
    f64 lo = 0.0, hi = 14.0, x = w->pH > 0.0 && w->pH < 14.0 ? w->pH : 7.0;
    for (u32 it = 0; it < 80; ++it) {
        f64 f = paw_alk_closed(&a, ka, x, dic, no2) - ta;
        if (f > 0.0) hi = x; else lo = x;
        if (hi - lo < 1e-10) break;
        f64 d = (paw_alk_closed(&a, ka, x + 1e-6, dic, no2) - ta - f) * 1e6;
        f64 xn = d > 0.0 ? x - f / d : 0.5 * (lo + hi);
        if (!(xn > lo && xn < hi)) xn = 0.5 * (lo + hi);
        if (fabs(xn - x) < 1e-11) { x = xn; break; }
        x = xn;
    }
    w->pH = x;
    return x;
}

// Fresh source water: carbonate alkalinity (KH) at the source pH, closed system.
internal void paw_fill_source(PawWater *w, const PawParams *p, f64 V_L) {
    memset(w, 0, sizeof(*w));
    w->V_L = V_L;
    w->T_C = p->source_T_C;
    w->pH = p->source_pH;
    if (V_L <= 0.0) return;
    // the activity coefficients depend on the ionic strength, which depends on
    // the result: iterate to the fixed point (the solver then reproduces the pH)
    f64 I = 0.5e-3 * 2.0 * p->source_kh_meq;
    for (u32 it = 0; it < 8; ++it) {
        NftAcidBase a = nft_acid_base(w->T_C, I);
        f64 aH = pow(10.0, -(f64)p->source_pH);
        f64 r1 = a.K1 / (aH * a.g1), r2 = a.K2 * a.g1 / (aH * a.g2);
        f64 a1 = r1 / (1.0 + r1 * (1.0 + r2)), a2 = a1 * r2;
        f64 dic = (f64)p->source_kh_meq * 1e-3 / (a1 + 2.0 * a2);                 // mol/L
        f64 ta = (f64)p->source_kh_meq * 1e-3 + a.Kw / (aH * a.g1) - aH / a.g1;  // + hydroxide - H+
        w->dic_mmol = dic * 1e3 * V_L;
        w->R_meq = ta * 1e3 * V_L;
        I = paw_ionic_strength(w);
    }
}

// Mixes `add` (taken whole) into `w`.
internal void paw_mix(PawWater *w, const PawWater *add) {
    f64 V = w->V_L + add->V_L;
    if (V <= 0.0) return;
    w->T_C = (w->T_C * w->V_L + add->T_C * add->V_L) / V;
    w->no3_mmol += add->no3_mmol; w->no2_mmol += add->no2_mmol; w->o3_mmol += add->o3_mmol;
    w->dic_mmol += add->dic_mmol; w->R_meq += add->R_meq;
    w->V_L = V;
}

// Takes `V_L` out of `w` into `out` (same composition).
internal void paw_take(PawWater *w, f64 V_L, PawWater *out) {
    f64 f = w->V_L > 0.0 ? Clamp(V_L / w->V_L, 0.0, 1.0) : 0.0;
    *out = *w;
    out->V_L = w->V_L * f;
    out->no3_mmol = w->no3_mmol * f; out->no2_mmol = w->no2_mmol * f; out->o3_mmol = w->o3_mmol * f;
    out->dic_mmol = w->dic_mmol * f; out->R_meq = w->R_meq * f;
    w->V_L -= out->V_L;
    w->no3_mmol -= out->no3_mmol; w->no2_mmol -= out->no2_mmol; w->o3_mmol -= out->o3_mmol;
    w->dic_mmol -= out->dic_mmol; w->R_meq -= out->R_meq;
}

// --- skid ---------------------------------------------------------------------------

enum PawState { PawState_Treat, PawState_Hold, PawState_Ready };
global const char *const paw_state_names[3] = { "treating", "holding (ozone decay)", "ready" };

struct PawAccounts {
    f64 plasma_J, supply_J, pump_J;
    f64 nox_N_mmol, o3_mmol;               // made by the plasma (N: every oxidized N species, N2O included)
    f64 n2o_N_mmol;                        // of which N2O (insoluble, vented; a greenhouse gas)
    f64 vented_N_mmol, vented_o3_mmol;     // through both water stages
    f64 delivered_L, delivered_N_mmol, delivered_no2_mmol;
    f64 source_L;                          // fresh source water drawn
    f64 batches, last_batch_h, last_batch_kWh;
    f64 nitrite_frac_last;                 // NO2- / (NO3- + NO2-) at the end of the last treatment
};

struct PawSkid {
    PawWater drum, bucket;
    u32 state;
    f64 state_t;                           // s in the current state
    f64 batch_t, batch_J;                  // treatment time and plasma energy of the current batch
    PawElectrical el;
    PawThermal th;                         // module heat balance (alumina, pipe, gap gas)
    PawGas gas;                            // M2 kinetics outlet of one module (gas_source = kinetics, and for display)
    f64 capture;                           // current venturi NOx capture
    f64 plasma_on;                         // fraction of the last window with the plasma on
    PawAccounts acc;
};

internal void paw_init(PawSkid *k, const PawParams *p) {
    memset(k, 0, sizeof(*k));
    if (!p->enabled) return;
    paw_fill_source(&k->drum, p, p->drum_L);
    paw_fill_source(&k->bucket, p, p->stage2_on ? p->stage2_L : 0.0);
    k->acc.source_L = k->drum.V_L + k->bucket.V_L;
    k->state = PawState_Treat;
    k->el = paw_electrical(p);
    k->th = paw_thermal(p, &k->el, k->drum.T_C);
    paw_solve_ph(&k->drum);
    paw_solve_ph(&k->bucket);
}

// Empty drum: refill with the bucket (yesterday's off-gas already in it) plus
// fresh source water, refill the bucket, start treating.
internal void paw_refill(PawSkid *k, const PawParams *p) {
    PawWater fresh;
    if (p->stage2_on && k->bucket.V_L > 0.0) {
        PawWater b;
        paw_take(&k->bucket, k->bucket.V_L, &b);
        paw_mix(&k->drum, &b);
    }
    f64 need = Maximum((f64)p->drum_L - k->drum.V_L, 0.0);
    paw_fill_source(&fresh, p, need);
    paw_mix(&k->drum, &fresh);
    k->acc.source_L += need;
    if (p->stage2_on) {
        paw_fill_source(&fresh, p, p->stage2_L);
        paw_mix(&k->bucket, &fresh);
        k->acc.source_L += p->stage2_L;
    }
    paw_solve_ph(&k->drum);
    paw_solve_ph(&k->bucket);
    k->state = PawState_Treat;
    k->state_t = 0.0;
    k->batch_t = 0.0;
    k->batch_J = 0.0;
}

// CO2 exchange toward air equilibrium (implicit over dt), at the water's pH.
internal void paw_co2_exchange(PawWater *w, f64 kla, f64 pco2_atm, f64 dt) {
    if (w->V_L <= 0.0) return;
    NftAcidBase a = nft_acid_base(w->T_C, paw_ionic_strength(w));
    f64 aH = pow(10.0, -w->pH);
    f64 r1 = a.K1 / (aH * a.g1), r2 = a.K2 * a.g1 / (aH * a.g2);
    f64 a0 = 1.0 / (1.0 + r1 * (1.0 + r2));                  // CO2* share of DIC
    f64 sat = a.K0 * pco2_atm * 1e3 * w->V_L;                   // mmol of CO2* at equilibrium
    f64 co2 = w->dic_mmol * a0;
    f64 co2_new = (co2 + kla * dt * sat) / (1.0 + kla * dt);   // CO2* relaxes; the carbonate re-speciates
    w->dic_mmol = Maximum(w->dic_mmol + (co2_new - co2), 0.0);
}

// Liquid chemistry of one water over dt with NOx / O3 delivered at constant
// rates (mmol/s): NO2 hydrolysis (1:1 nitrite:nitrate), N2O5 -> nitrate,
// nitrite + O3 -> nitrate (k = 3.7e5 /M/s, Hoigne & Bader 1985), O3 decay
// k0 + k_OH [OH-] (empirical; seconds at pH 10, ~20 min near neutral).
internal void paw_liquid(PawWater *w, f64 nitrate_in, f64 nitrite_in, f64 o3_in, f64 dt) {
    if (w->V_L <= 0.0) return;
    u32 sub = (u32)Maximum(1.0, ceil(dt / 1.0));
    f64 h = dt / sub;
    f64 oh = pow(10.0, w->pH - 14.0);
    f64 k_dec = 5e-4 + 1.0e3 * oh;
    for (u32 i = 0; i < sub; ++i) {
        w->no3_mmol += nitrate_in * h;
        w->no2_mmol += nitrite_in * h;
        f64 no2_M = w->no2_mmol / w->V_L * 1e-3;
        f64 k_ox = 3.7e5 * no2_M;                                 // 1/s on O3
        f64 o3 = (w->o3_mmol + o3_in * h) / (1.0 + (k_dec + k_ox) * h);   // implicit
        f64 ox = Minimum(k_ox * o3 * h, w->no2_mmol);             // mmol nitrite -> nitrate
        w->no2_mmol -= ox;
        w->no3_mmol += ox;
        w->o3_mmol = o3;
    }
}

internal f64 paw_capture(const PawParams *p, f64 pH) {
    f64 x = Clamp((pH - 6.0) / 2.0, 0.0, 1.0);
    return (f64)p->capture_lo + ((f64)p->capture_hi - p->capture_lo) * x;
}

// One coupling window (dt s) of the skid.
internal void paw_window(PawSkid *k, const PawParams *p, f64 dt, f64 T_amb, f64 pco2_atm) {
    if (!p->enabled) return;
    k->el = paw_electrical(p);
    PawWater *d = &k->drum;
    k->th = paw_thermal(p, &k->el, d->T_C);
    f64 on = 0.0;
    if (k->state == PawState_Treat && d->V_L > 0.0) {
        // acid still needed to reach the batch pH at the current DIC (exact
        // stop inside the window: production is constant over it)
        NftAcidBase a = nft_acid_base(d->T_C, paw_ionic_strength(d));
        f64 ta_target = paw_alk_closed(&a, paw_ka_hno2(d->T_C), p->batch_pH, d->dic_mmol / d->V_L * 1e-3,
                                       d->no2_mmol / d->V_L * 1e-3) * 1e3 * d->V_L;     // meq
        f64 need = d->R_meq - d->no3_mmol - d->no2_mmol - ta_target;
        f64 cap = paw_capture(p, d->pH);
        f64 nox_rate;                      // soluble N, mmol / s at full on-time
        if (p->gas_source == PawGas_Kinetics && k->gas.valid) {
            const f64 *x = k->gas.x;
            nox_rate = (x[PawS_NO2] + Minimum(x[PawS_NO], x[PawS_NO2]) + x[PawS_NO3] + 2.0 * x[PawS_N2O5] + x[PawS_HNO2] +
                        x[PawS_HNO3]) * (f64)p->air_L_min / 60.0 * PAW_AIR_MOL_PER_L * 1e3;
        } else {
            nox_rate = k->el.P_avg_W * paw_nox_regime(k->el.SEI_kJ_L) / ((f64)p->nox_MJ_mol * 1e6) * 1e3;
        }
        f64 acid_per_s = nox_rate * cap;   // every captured N is one strong anion (nitrate or nitrite)
        on = acid_per_s > 0.0 ? Clamp(need / (acid_per_s * dt), 0.0, 1.0) : 0.0;
        if (need <= 0.0) on = 0.0;
    }
    k->plasma_on = on;
    f64 P = k->el.P_avg_W * on;
    k->capture = paw_capture(p, d->pH);
    // gas leaving the modules (mmol/s): nitrate-forming N, nitrite-forming N
    // (from NO2 half and half, HNO2 all nitrite), insoluble N (NO alone, N2O), O3
    f64 to_nitrate, to_nitrite, insoluble, n2o, o3;
    if (p->gas_source == PawGas_Kinetics && k->gas.valid) {
        f64 mol = (f64)p->air_L_min / 60.0 * PAW_AIR_MOL_PER_L * on * 1e3;   // mmol air / s
        const f64 *x = k->gas.x;
        // NO is insoluble alone, but NO + NO2 absorb together as N2O3
        // (NO + NO2 + 2 OH- -> 2 NO2- + H2O, fast in alkaline water): the
        // equimolar part goes to nitrite, the NO2 left over hydrolyses 1:1
        f64 pair = Minimum(x[PawS_NO], x[PawS_NO2]);
        f64 no2_left = x[PawS_NO2] - pair;
        to_nitrate = (x[PawS_NO3] + 2.0 * x[PawS_N2O5] + x[PawS_HNO3] + 0.5 * no2_left) * mol;
        to_nitrite = (x[PawS_HNO2] + 2.0 * pair + 0.5 * no2_left) * mol;
        n2o = 2.0 * x[PawS_N2O] * mol;
        insoluble = (x[PawS_NO] - pair) * mol + n2o;
        o3 = x[PawS_O3] * mol;
    } else {
        f64 regime = paw_nox_regime(k->el.SEI_kJ_L);
        f64 nox = P * regime / ((f64)p->nox_MJ_mol * 1e6) * 1e3;                     // mmol N / s
        f64 o3_g_kWh = (f64)p->o3_g_kWh * (1.0 - regime) + (f64)p->o3_nox_g_kWh * regime;
        o3 = P * o3_g_kWh / 3.6e6 / 48.0 * 1e3;                                      // mmol / s
        f64 n2o5 = Clamp((f64)p->n2o5_share, 0.0, 1.0);
        to_nitrate = nox * (n2o5 + 0.5 * (1.0 - n2o5));
        to_nitrite = nox * 0.5 * (1.0 - n2o5);
        insoluble = n2o = 0.0;
    }
    f64 soluble = to_nitrate + to_nitrite;
    f64 capN = soluble * k->capture, capO3 = o3 * p->o3_capture;
    f64 escN = soluble - capN, escO3 = o3 - capO3;
    paw_liquid(d, to_nitrate * k->capture, to_nitrite * k->capture, capO3, dt);
    f64 b2N = 0.0, b2O3 = 0.0;   // what escapes is mostly NO2: half and half in the bucket
    if (p->stage2_on && k->bucket.V_L > 0.0) { b2N = escN * p->stage2_no2; b2O3 = escO3 * p->stage2_o3; }
    paw_liquid(&k->bucket, 0.5 * b2N, 0.5 * b2N, b2O3, dt);
    k->acc.nox_N_mmol += (soluble + insoluble) * dt;
    k->acc.n2o_N_mmol += n2o * dt;
    k->acc.o3_mmol += o3 * dt;
    k->acc.vented_N_mmol += (escN - b2N + insoluble) * dt;
    k->acc.vented_o3_mmol += (escO3 - b2O3) * dt;
    // carbon dioxide: the venturi bubbles strip it toward air equilibrium; the bucket sparger less so
    paw_co2_exchange(d, p->kla_co2_per_s, pco2_atm, dt);
    if (p->stage2_on) paw_co2_exchange(&k->bucket, 0.25 * p->kla_co2_per_s, pco2_atm, dt);
    // heat: plasma (most of it reaches the water through the submerged pipes), pump, drum walls
    if (d->V_L > 0.0) {
        f64 q = (f64)p->heat_frac * P + p->pump_W + p->drum_ua_W_K * (T_amb - d->T_C);
        d->T_C += q * dt / (NFT_RHO_CP * 1e-3 * d->V_L);
    }
    k->acc.plasma_J += P * dt;
    k->acc.supply_J += (p->chain_eff > 0.0f ? P / p->chain_eff : 0.0) * dt;
    k->acc.pump_J += p->pump_W * dt;
    k->batch_J += P * dt;
    paw_solve_ph(d);
    if (p->stage2_on) paw_solve_ph(&k->bucket);

    k->state_t += dt;
    if (k->state == PawState_Treat) {
        k->batch_t += on * dt;
        if (d->pH <= p->batch_pH + 1e-6 && d->V_L > 0.0) {
            k->state = PawState_Hold;
            k->state_t = 0.0;
            k->acc.batches += 1.0;
            k->acc.last_batch_h = k->batch_t / 3600.0;
            k->acc.last_batch_kWh = k->batch_J / 3.6e6;
            f64 n = d->no3_mmol + d->no2_mmol;
            k->acc.nitrite_frac_last = n > 0.0 ? d->no2_mmol / n : 0.0;
        }
    } else if (k->state == PawState_Hold) {
        f64 o3_mg_L = d->V_L > 0.0 ? d->o3_mmol * 48.0 / d->V_L : 0.0;
        if (k->state_t >= (f64)p->hold_h * 3600.0 && o3_mg_L <= p->o3_ready_mg_L) { k->state = PawState_Ready; k->state_t = 0.0; }
    }
    if (k->state == PawState_Ready && d->V_L < 0.02 * p->drum_L) paw_refill(k, p);
}

// Finished PAW available for top-off (L).
internal f64 paw_available_L(PawSkid *k, const PawParams *p) {
    return p->enabled && k->state == PawState_Ready ? k->drum.V_L : 0.0;
}

// The DBD reactor scene's operating point for one module of the skid: dry air
// through the 2.8 mm annulus, plug flow at the module's residence time, the
// module's delivered power over the annulus volume.
struct PawModuleOperating { f64 power_W_cm3, freq_Hz, residence_s, x_H2O, T_K; };
internal PawModuleOperating paw_module_operating(const PawParams *p, f64 water_T_C) {
    PawElectrical e = paw_electrical(p);
    PawModuleOperating o;
    f64 per_module_W = p->modules ? e.P_on_W / p->modules : 0.0;
    f64 flow_cm3_s = p->modules ? (f64)p->air_L_min * 1000.0 / 60.0 / p->modules : 1.0;
    o.power_W_cm3 = per_module_W / e.annulus_cm3;
    o.freq_Hz = (f64)p->freq_kHz * 1e3;
    o.residence_s = e.annulus_cm3 / flow_cm3_s;
    o.x_H2O = 103.0 / 101325.0;                      // dryer: dew point -20 C (e_s = 103 Pa)
    o.T_K = paw_gas_T_K(p, &e, water_T_C);           // module heat balance (see paw_thermal)
    return o;
}
