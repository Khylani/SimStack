// NFT (nutrient film technique) system: rails, reservoir, chiller, plants.
//
//   rails      sloped gullies carrying a thin laminar film. Along each rail the
//              film carries temperature, dissolved O2 and seven nutrient ions;
//              heat enters through the wall (U-value) and from radiant load on
//              the top (absorptivity); O2 re-aerates toward saturation; plants
//              at fixed sites take up ions and O2.
//   reservoir  one well-mixed tank (f64): return flow from every rail, wall
//              exchange with ambient, pump heat, a chiller with hysteresis,
//              fresh-water top-off and an EC-controlled dosing pump.
//   plants     one per site, logistic growth limited by root-zone temperature,
//              dissolved O2 and nutrients (Liebig minimum over Michaelis-Menten
//              uptake capacity). Uptake = tissue composition x actual growth,
//              so nutrient mass balances exactly; transpiration is tied to
//              growth. Harvest at the crop's cycle length, then replant.
//
// Memory layout (the point of this file): lanes are rails. Eight rails form a
// group, the width of an f32x8. Each group's state is one contiguous block:
//
//   NftGroup header (lane parameters, crop parameters, outflow sums)
//   rows[cells]   : NftRow  { f32 f[field][8] }      288 B per cell row
//   sites[n]      : NftSiteRow { W[8], age[8], drop[field][8], ... }
//
// An 80-cell group is ~23 KB of rows: it stays in L1 while one thread runs it
// for a whole coupling window (hundreds of substeps). Upwind advection reads
// the previous row: aligned 32-byte loads, no gathers, no rail-boundary
// branches (row 0's upstream is the reservoir). Groups are 64-byte aligned
// and owned by one thread per window, so there is no false sharing.
//
// Rail cells hold *deviations* from a per-window reference (the f32 inlet):
// films differ from the tank by tenths of a percent, so deltas are ~1000x
// smaller than the absolute values and every f32 rounding is ~1000x smaller.
// Without this, f32 advection of near-uniform 150 mg/L fields loses ~1e-4 of
// the mass per month to systematically signed rounding. Read absolute values
// with nft_abs().
//
// The kernels are templates over the lane type: V = f32x8 (production) or f32
// (reference, one lane at a time); the test checks they are bit-identical.
// Included by simstack.cpp and the test TU.

#define NFT_LANES        8
#define NFT_MAX_SITES    1024        // plants per channel
#define NFT_MAX_CELLS    8192        // cells per channel (>= 2 per plant)
#define NFT_RHO_CP       4.186e6     // J / (m^3 K), water
#define NFT_G            9.81

#include "nft_hydraulics.cpp"

enum NftField {
    NftF_T,       // C
    NftF_DO,      // mg/L dissolved O2
    NftF_NO3,     // mg/L as N
    NftF_NH4,     // mg/L as N
    NftF_P,       // mg/L
    NftF_K,
    NftF_Ca,
    NftF_Mg,
    NftF_S,       // mg/L as S
    NftF_Count,
};
#define NFT_ION0 NftF_NO3          // fields NO3..S are ions
#define NFT_IONS (NftF_Count - NFT_ION0)

global const char *const nft_field_names[NftF_Count] = { "T", "DO", "NO3-N", "NH4-N", "P", "K", "Ca", "Mg", "S" };

// Plant nutrients (demand side): N is met from NO3 and NH4.
enum NftNutrient { NftN_N, NftN_P, NftN_K, NftN_Ca, NftN_Mg, NftN_S, NftN_Count };
global const char *const nft_nutrient_names[NftN_Count] = { "N", "P", "K", "Ca", "Mg", "S" };

// Michaelis-Menten half-saturation (mg/L) per ion field (NO3..S).
global const f32 nft_km[NFT_IONS] = { 1.0f, 0.3f, 0.2f, 1.5f, 3.0f, 1.5f, 1.0f };

// ---------------------------------------------------------------------------
// Crops. Illustrative literature-range values (dry mass, temperature response,
// tissue composition); every number is a per-cultivar tunable.

struct NftCrop {
    const char *name;
    f32 cycle_days;          // transplant -> harvest
    f32 w0_g, wmax_g;        // dry mass per plant at transplant / asymptote
    f32 rgr_per_day;         // maximum relative growth rate
    f32 t_min, t_opt_lo, t_opt_hi, t_max;   // root-zone temperature response, C
    f32 root_frac;           // root share of dry mass
    f32 transp_g_per_g;      // water transpired per g dry mass produced
    f32 tissue[NftN_Count];  // g / g dry mass: N P K Ca Mg S
    f32 o2_mg_per_g_h;       // root respiration at 20 C, mg O2 / (g root DM h)
    f32 do_crit;             // mg/L: growth falls to 0 at do_crit/2, full at do_crit
    f32 t_bolt;              // air temperature above which heat accumulates toward bolting, C
    f32 bolt_dd;             // degree-days above t_bolt that mark a plant at bolting risk
};

enum NftCropId {
    NftCrop_Cilantro, NftCrop_Arugula, NftCrop_Lettuce, NftCrop_Basil, NftCrop_Kale, NftCrop_Spinach, NftCrop_Strawberry,
    NftCrop_Count,
};

// Cilantro (Coriandrum sativum) and arugula (Eruca sativa) are cool-season
// leaf crops that bolt under heat and long days: root-zone optimum ~17-22 C,
// so the chiller and the rail wrap matter. Arugula is a brassica (high Ca, S).
global const NftCrop nft_crops[NftCrop_Count] = {
    // name                     cycle  w0     wmax  rgr     tmin lo  hi  tmax root   transp  N       P       K       Ca      Mg       S       o2    DOc   tbolt dd
    { "Cilantro",                 40, 0.06f,  4.0f, 0.22f,   5, 17, 22, 29, 0.14f, 230, {0.045f, 0.005f, 0.055f, 0.013f, 0.0035f, 0.003f}, 1.6f, 4.0f, 25, 12 },
    { "Arugula",                  28, 0.05f,  3.5f, 0.30f,   4, 18, 22, 28, 0.12f, 220, {0.050f, 0.006f, 0.060f, 0.025f, 0.0040f, 0.010f}, 1.6f, 4.0f, 26, 10 },
    { "Lettuce (butterhead)",     35, 0.12f, 12.0f, 0.30f,   4, 18, 22, 30, 0.12f, 230, {0.045f, 0.006f, 0.070f, 0.012f, 0.0035f, 0.003f}, 1.5f, 4.0f, 27, 15 },
    { "Basil",                    42, 0.15f, 18.0f, 0.24f,  10, 22, 27, 34, 0.12f, 250, {0.040f, 0.006f, 0.050f, 0.018f, 0.0060f, 0.004f}, 1.8f, 4.0f, 32, 30 },
    { "Kale",                     50, 0.15f, 28.0f, 0.20f,   4, 16, 21, 28, 0.15f, 220, {0.045f, 0.006f, 0.045f, 0.025f, 0.0040f, 0.008f}, 1.5f, 4.0f, 27, 20 },
    { "Spinach",                  30, 0.10f,  9.0f, 0.30f,   3, 15, 20, 26, 0.12f, 220, {0.045f, 0.005f, 0.075f, 0.010f, 0.0090f, 0.003f}, 1.6f, 4.5f, 24, 8 },
    { "Strawberry (day-neutral)",120, 1.00f, 70.0f, 0.08f,   6, 16, 22, 28, 0.20f, 300, {0.025f, 0.004f, 0.030f, 0.012f, 0.0040f, 0.002f}, 1.2f, 4.0f, 30, 40 },
};

// Rail wrap / surface presets: U (W/m^2/K, outer surface) and top absorptivity.
enum NftWrapId { NftWrap_BarePvc, NftWrap_BubbleMylar, NftWrap_Count };
struct NftWrap { const char *name; f32 u_w_m2k; f32 absorptivity; };
global const NftWrap nft_wraps[NftWrap_Count] = {
    { "Bare white PVC",   6.0f, 0.35f },
    { "Bubble mylar wrap", 2.0f, 0.06f },
};

// ---------------------------------------------------------------------------
// Layout

struct alignas(32) NftRow {
    f32 f[NftF_Count][NFT_LANES];
};

struct alignas(32) NftSiteRow {
    f32 W[NFT_LANES];                       // g dry mass
    f32 age[NFT_LANES];                     // s since transplant
    f32 drop[NftF_Count][NFT_LANES];        // concentration removed per substep (mg/L); [T] unused
    f32 uptake[NFT_IONS][NFT_LANES];        // mg/s this window (accounting)
    f32 o2_rate[NFT_LANES];                 // mg/s
    f32 transp[NFT_LANES];                  // g water / s
    f32 fT[NFT_LANES], fDO[NFT_LANES], fN[NFT_LANES];   // limitation factors (diagnostics / UI)
    f32 harvested_g[NFT_LANES];             // cumulative dry mass harvested at this site
    f32 harvests[NFT_LANES];
    f32 heat_dd[NFT_LANES];                 // air degree-days above t_bolt since transplant
    f32 bolted[NFT_LANES];                  // cumulative harvests that had crossed bolt_dd
    f32 removed[NftF_Count][NFT_LANES];     // concentration actually removed this window (sum of exact
                                            // x_before - x_after, Sterbenz), x cell volume = booked uptake
};

// Per-lane (per-rail) constants, recomputed when parameters change.
struct NftLaneParams {
    f32 courant[NFT_LANES];     // v dt / dx
    f32 k_wall[NFT_LANES];      // UA_cell dt / C_cell      (per substep)
    f32 k_rad[NFT_LANES];       // a w dx dt / C_cell       (per substep, x radiant W/m^2)
    f32 k_o2[NFT_LANES];        // kLa dt                   (per substep)
    f32 inv_vcell_L[NFT_LANES]; // 1 / cell volume in litres
    f32 flow_L_s[NFT_LANES];    // 0 for padding lanes
    f32 depth_m[NFT_LANES], velocity_m_s[NFT_LANES];
};

struct NftCropLanes {
    f32 rgr_s[NFT_LANES], w0[NFT_LANES], wmax[NFT_LANES], cycle_s[NFT_LANES];
    f32 t_min[NFT_LANES], t_lo[NFT_LANES], t_hi[NFT_LANES], t_max[NFT_LANES];
    f32 root_frac[NFT_LANES], transp[NFT_LANES], o2_s[NFT_LANES], do_crit[NFT_LANES];
    f32 t_bolt[NFT_LANES], bolt_dd[NFT_LANES];
    f32 tissue[NftN_Count][NFT_LANES];    // g/g
    f32 imax_s[NFT_IONS][NFT_LANES];      // mg / (g root s) at saturation, per ion field
};

struct alignas(64) NftGroup {   // one cache-line-aligned block per 8-rail group; owned by one thread per window
    NftLaneParams lp;
    NftCropLanes crop;
    f32 out_sum[NftF_Count][NFT_LANES];   // sum over the window's substeps of (outlet - inlet): small, so f32 sums stay exact
    NftRow *rows;                          // cells
    NftSiteRow *sites;                     // site rows
};

// Every lane array must start on a 32-byte boundary for aligned f32x8 loads.
static_assert(sizeof(NftRow) % 32 == 0, "NftRow lane arrays");
static_assert(sizeof(NftSiteRow) % 32 == 0, "NftSiteRow lane arrays");
static_assert(sizeof(NftLaneParams) % 32 == 0 && sizeof(NftCropLanes) % 32 == 0, "lane parameter arrays");
static_assert(offsetof(NftGroup, crop) % 32 == 0 && offsetof(NftGroup, out_sum) % 32 == 0, "NftGroup lane arrays");

// ---------------------------------------------------------------------------
// User parameters (edited live in the UI; structural ones need a rebuild)

struct NftBank {                // half of the rails: crop + wrap + planting stagger
    u32 crop;                   // NftCropId
    u32 wrap;                   // NftWrapId
    f32 stagger_days;           // site ages are spread over this many days (0 = all planted at once)
};

struct NftParams {
    // structure (rebuild): the system scales with these three
    u32 rails;                  // channels
    f32 rail_length_m;          // channel length
    u32 plants_per_rail;        // plants per channel (evenly spaced: spacing = length / plants)
    b32 auto_size;              // size reservoir, flow and chiller from the structure
    // hydraulics: PVC network, pump, eductor, channel profile and grade (nft_hydraulics.cpp)
    NftHydraulicParams hyd;
    f32 target_flow_L_min;      // per channel, for auto-sizing the pump and pipes
    b32 pump_submersible;       // 1: all pump power heats the solution; 0: external pump, only hydraulic power
    // environment
    f32 ambient_mean_C, ambient_amp_C;
    f32 photoperiod_h, lights_on_h;
    f32 radiant_W_m2;           // on rail tops while lights are on
    f32 kla_film_per_s;
    // reservoir
    f32 reservoir_L, tank_ua_W_K, tank_kla_per_s;
    b32 chiller_on;
    f32 chiller_set_C, chiller_band_C, chiller_W, chiller_cop;
    b32 dosing_on, topoff_on;
    b32 dose_matched;           // replenishment stock in the crops' uptake ratios (else the fill recipe)
    f32 ec_set_mS, dose_L_per_h_max;
    f32 dump_days;              // full solution change every N days (0 = never)
    f32 recipe[NFT_IONS];       // mg/L of the fill solution (stock = 100x)
    f32 fill_T_C;               // fresh water temperature
    NftBank bank[2];            // rails [0, rails/2) bank A, the rest bank B
};

internal NftParams nft_default_params() {
    NftParams p = {};
    p.rails = 16;
    p.rail_length_m = 4.0f;
    p.plants_per_rail = 26;     // 15 cm spacing
    p.auto_size = 1;
    p.hyd = nft_hydraulics_defaults();
    p.target_flow_L_min = 1.5f;
    p.pump_submersible = 1;
    p.ambient_mean_C = 22.0f;   // indoor herb room
    p.ambient_amp_C = 3.0f;
    p.photoperiod_h = 16.0f;
    p.lights_on_h = 6.0f;
    p.radiant_W_m2 = 150.0f;
    p.kla_film_per_s = 0.01f;
    p.reservoir_L = 200.0f;
    p.tank_ua_W_K = 3.0f;
    p.tank_kla_per_s = 0.002f;
    p.chiller_on = 1;
    p.chiller_set_C = 20.0f;
    p.chiller_band_C = 0.3f;
    p.chiller_W = 800.0f;
    p.chiller_cop = 3.0f;
    p.dosing_on = 1;
    p.topoff_on = 1;
    p.dose_matched = 1;
    p.dump_days = 0.0f;
    p.ec_set_mS = 1.6f;
    p.dose_L_per_h_max = 0.5f;
    f32 recipe[NFT_IONS] = { 140.0f, 10.0f, 40.0f, 200.0f, 160.0f, 40.0f, 60.0f };   // lettuce-type working solution
    for (u32 i = 0; i < NFT_IONS; ++i) p.recipe[i] = recipe[i];
    p.fill_T_C = 18.0f;
    p.bank[0].crop = NftCrop_Cilantro;
    p.bank[0].wrap = NftWrap_BubbleMylar;
    p.bank[0].stagger_days = 35.0f;
    p.bank[1].crop = NftCrop_Arugula;
    p.bank[1].wrap = NftWrap_BubbleMylar;
    p.bank[1].stagger_days = 24.0f;
    return p;
}

// ---------------------------------------------------------------------------
// System

struct NftReservoir {
    f64 V_L;
    f64 T_C;
    f64 mass_mg[NftF_Count];    // [DO..S]: mg in the tank ([T] unused)
    b32 chiller_running;
};

struct NftAccounts {            // cumulative since reset
    f64 chiller_heat_J, chiller_elec_J, pump_J;
    f64 uptake_mg[NFT_IONS];
    f64 dosed_mg[NFT_IONS];
    f64 dosed_L, topoff_L, transpired_L;
    f64 dumped_L, dumped_mg[NFT_IONS];
    f64 last_dump_t;
    f64 harvest_g[2], harvests[2];     // per bank
    f64 bolted[2];                     // harvests that had crossed the crop's bolting degree-days
    f64 chiller_on_s;
};

struct NftSystem {
    NftParams params;
    u32 groups, cells, sites;
    u32 site_offset, site_stride;      // site row j sits on cell site_offset + j * site_stride
    f64 cell_m;                        // dx = spacing / site_stride
    NftHydraulics hyd;                 // operating point of the network (re-solved hourly)
    f64 hyd_solved_hour;
    u32 rails;
    NftGroup *group;                   // array, each pointing at its own contiguous block
    NftReservoir res;
    NftAccounts acc;
    f64 t;                             // s
    f64 window_s;                      // coupling window (reservoir / plant sync)
    u32 substeps;                      // per window
    f64 dt_sub;
    // environment at the current window
    f32 T_amb, light;
    f32 inlet[NftF_Count];             // exactly what the rails received this window (f32)
    f32 ref[NftF_Count];               // rail cells store x - ref (deltas): ref = this window's inlet
};

internal f64 nft_reservoir_conc(NftSystem *s, u32 field) {
    return s->res.V_L > 0.0 ? s->res.mass_mg[field] / s->res.V_L : 0.0;
}

// EC from ion concentrations: 0.1 x cation meq/L (mS/cm), the usual rule of thumb.
internal f64 nft_ec_mS(const f64 *c) {
    f64 meq = c[NftF_NH4] / 14.007 + c[NftF_K] / 39.098 + c[NftF_Ca] / 20.039 + c[NftF_Mg] / 12.153;
    return 0.1 * meq;
}

// O2 saturation in fresh water at 1 atm, mg/L (polynomial fit, 0-40 C).
template <class V>
SIM_HD V nft_do_sat(V T) {
    V a = lane_splat((V *)0, -0.000077774), b = lane_splat((V *)0, 0.007991);
    V c = lane_splat((V *)0, -0.41022), d = lane_splat((V *)0, 14.652);
    return lane_fmadd(lane_fmadd(lane_fmadd(a, T, b), T, c), T, d);
}
internal f64 nft_do_sat_f64(f64 T) { return ((-0.000077774 * T + 0.007991) * T - 0.41022) * T + 14.652; }

internal b32 nft_rail_bank(NftSystem *s, u32 rail) { return rail >= (s->rails + 1) / 2; }

// Network operating point, then per-channel film and per-substep coefficients.
// Called on rebuild, on any parameter change, and hourly (water viscosity
// follows the tank temperature).
internal void nft_update_lane_params(NftSystem *s) {
    NftParams *p = &s->params;
    f64 T = s->res.V_L > 0.0 ? s->res.T_C : (f64)p->fill_T_C;
    if (p->hyd.balance_taps) nft_hydraulics_balance(&p->hyd, s->rails, T, &s->hyd);
    else nft_hydraulics_solve(&p->hyd, s->rails, T, &s->hyd);
    const NftProfile *prof = nft_profiles + (p->hyd.profile % NftProfile_Count);
    f64 w = prof->width_m, perim = prof->perimeter_m;
    f64 dx = s->cell_m;
    f64 depth[8192], vel[8192];
    f64 v_max = 1e-6;
    for (u32 r = 0; r < s->rails; ++r) {
        f64 q = s->hyd.q_channel[r];
        f64 S = p->hyd.slope[nft_rail_bank(s, r) ? 1 : 0];
        depth[r] = q > 0.0 ? nft_film_depth(q, w, S, s->hyd.nu) : 1e-4;
        vel[r] = q > 0.0 ? q / (w * depth[r]) : 0.0;
        v_max = Maximum(v_max, vel[r]);
    }
    // substep: Courant 0.9 on the fastest channel, an integer count per window
    s->window_s = 60.0;
    s->substeps = (u32)ceil(s->window_s / (0.9 * dx / v_max));
    if (s->substeps < 1) s->substeps = 1;
    s->dt_sub = s->window_s / (f64)s->substeps;
    for (u32 g = 0; g < s->groups; ++g) {
        NftLaneParams *lp = &s->group[g].lp;
        for (u32 l = 0; l < NFT_LANES; ++l) {
            u32 rail = g * NFT_LANES + l;
            b32 live = rail < s->rails;
            u32 rr = live ? rail : 0;
            const NftWrap *wrap = nft_wraps + p->bank[live && nft_rail_bank(s, rail) ? 1 : 0].wrap;
            f64 vcell = w * depth[rr] * dx;                                   // m^3
            f64 ccell = NFT_RHO_CP * vcell;                                   // J/K
            lp->courant[l] = live ? (f32)(vel[rr] * s->dt_sub / dx) : 0.0f;
            lp->k_wall[l] = (f32)(wrap->u_w_m2k * perim * dx * s->dt_sub / ccell);
            lp->k_rad[l] = (f32)(wrap->absorptivity * w * dx * s->dt_sub / ccell);
            lp->k_o2[l] = (f32)(p->kla_film_per_s * s->dt_sub);
            lp->inv_vcell_L[l] = (f32)(1.0 / (vcell * 1000.0));
            lp->flow_L_s[l] = live ? (f32)(s->hyd.q_channel[rr] * 1000.0) : 0.0f;
            lp->depth_m[l] = (f32)depth[rr];
            lp->velocity_m_s[l] = (f32)vel[rr];
        }
    }
}

internal void nft_update_crop_lanes(NftSystem *s) {
    for (u32 g = 0; g < s->groups; ++g) {
        NftCropLanes *c = &s->group[g].crop;
        for (u32 l = 0; l < NFT_LANES; ++l) {
            u32 rail = g * NFT_LANES + l;
            const NftCrop *crop = nft_crops + s->params.bank[nft_rail_bank(s, rail) ? 1 : 0].crop;
            f32 rgr_s = crop->rgr_per_day / 86400.0f;
            c->rgr_s[l] = rgr_s;
            c->w0[l] = crop->w0_g;
            c->wmax[l] = crop->wmax_g;
            c->cycle_s[l] = crop->cycle_days * 86400.0f;
            c->t_min[l] = crop->t_min; c->t_lo[l] = crop->t_opt_lo; c->t_hi[l] = crop->t_opt_hi; c->t_max[l] = crop->t_max;
            c->root_frac[l] = crop->root_frac;
            c->transp[l] = crop->transp_g_per_g;
            c->o2_s[l] = crop->o2_mg_per_g_h / 3600.0f;
            c->do_crit[l] = crop->do_crit;
            c->t_bolt[l] = crop->t_bolt;
            c->bolt_dd[l] = crop->bolt_dd;
            for (u32 n = 0; n < NftN_Count; ++n) c->tissue[n][l] = crop->tissue[n];
            // Uptake capacity at saturation: 2.5x the demand of maximum growth, per g root.
            f32 cap_scale = 2.5f * rgr_s / crop->root_frac * 1000.0f;   // mg / (g root s) per (g/g tissue)
            c->imax_s[NftF_NO3 - NFT_ION0][l] = cap_scale * crop->tissue[NftN_N];
            c->imax_s[NftF_NH4 - NFT_ION0][l] = cap_scale * crop->tissue[NftN_N] * 0.3f;
            c->imax_s[NftF_P - NFT_ION0][l] = cap_scale * crop->tissue[NftN_P];
            c->imax_s[NftF_K - NFT_ION0][l] = cap_scale * crop->tissue[NftN_K];
            c->imax_s[NftF_Ca - NFT_ION0][l] = cap_scale * crop->tissue[NftN_Ca];
            c->imax_s[NftF_Mg - NFT_ION0][l] = cap_scale * crop->tissue[NftN_Mg];
            c->imax_s[NftF_S - NFT_ION0][l] = cap_scale * crop->tissue[NftN_S];
        }
    }
}

// Allocates every group as one contiguous, 64-byte aligned block and fills
// rails and reservoir with the working solution at the fill temperature.
// Grid from the structure: every plant gets `site_stride` cells of ~5 cm
// (at least 2), so sites sit on a uniform stride (no per-row lookups in the
// kernel) and the channel is exactly plants x spacing long.
internal void nft_layout(NftSystem *s, NftParams *p) {
    s->rails = Clamp(p->rails, 1u, 8192u);
    s->groups = (s->rails + NFT_LANES - 1) / NFT_LANES;
    s->sites = Clamp(p->plants_per_rail, 1u, (u32)NFT_MAX_SITES);
    f64 length = Clamp((f64)p->rail_length_m, 0.2, 200.0);
    f64 spacing = length / (f64)s->sites;
    u32 stride = Maximum((u32)(spacing / 0.05 + 0.5), 2u);
    while (stride > 2 && s->sites * stride > NFT_MAX_CELLS) --stride;
    s->site_stride = stride;
    s->site_offset = stride / 2;
    s->cells = s->sites * stride;
    s->cell_m = spacing / (f64)stride;
}

// Peak sensible heat into the solution (W) for chiller sizing: channel walls
// at the warmest ambient, radiant load on the channel tops, tank walls, pump.
internal f64 nft_peak_heat_W(NftSystem *s, NftParams *p, f64 pump_heat_W) {
    f64 dT = Maximum((f64)p->ambient_mean_C + p->ambient_amp_C - p->chiller_set_C, 0.0);
    const NftProfile *prof = nft_profiles + (p->hyd.profile % NftProfile_Count);
    f64 w = 0.0;
    for (u32 b = 0; b < 2; ++b) {
        u32 rails = b == 0 ? s->rails - s->rails / 2 : s->rails / 2;
        const NftWrap *wrap = nft_wraps + p->bank[b].wrap;
        w += rails * p->rail_length_m * (wrap->u_w_m2k * prof->perimeter_m * dT + wrap->absorptivity * prof->width_m * p->radiant_W_m2);
    }
    return w + p->tank_ua_W_K * dT + pump_heat_W;
}

internal f64 nft_pump_heat_W(NftParams *p, NftHydraulics *h) {
    return p->pump_submersible ? h->pump_power_W : h->rho * NFT_G * h->q_pump * h->head_pump;
}

// Auto-size from the structure: pipes for <= 1.5 m/s, the smallest pump
// preset that delivers the target flow to the weakest channel, ~0.75 L of
// reservoir per plant (>= 50 L), chiller 1.5x the peak heat load (>= 200 W).
internal void nft_auto_size(NftSystem *s, NftParams *p) {
    if (!p->auto_size) return;
    f64 plants = (f64)s->rails * (f64)s->sites;
    p->reservoir_L = (f32)Maximum(50.0, 0.75 * plants);
    p->tank_ua_W_K = (f32)(0.5 * pow((f64)p->reservoir_L / 1000.0, 2.0 / 3.0) * 6.0 * 1.8);   // ~cube tank, U 1.8 W/m2K
    f64 q_target = (f64)p->target_flow_L_min / 60000.0 * s->rails;
    u32 size = 0;
    while (size + 1 < NftPipe_Count &&
           q_target / (0.25 * 3.14159265358979 * nft_pipe_id_m(p->hyd.schedule80, size) * nft_pipe_id_m(p->hyd.schedule80, size)) > 1.5) ++size;
    p->hyd.supply_size = p->hyd.manifold_size = size;
    p->hyd.return_size = Minimum(size + 2, (u32)NftPipe_Count - 1);
    for (u32 pump = 0; pump < NftPump_Count; ++pump) {
        p->hyd.pump = pump;
        nft_hydraulics_solve(&p->hyd, s->rails, p->fill_T_C, &s->hyd);
        if (s->hyd.ok && s->hyd.q_min * 60000.0 >= p->target_flow_L_min) break;
    }
    p->chiller_W = (f32)Maximum(200.0, 1.5 * nft_peak_heat_W(s, p, nft_pump_heat_W(p, &s->hyd)));
}

internal void nft_init(NftSystem *s, Arena *arena, NftParams params) {
    memset(s, 0, sizeof(*s));
    nft_layout(s, &params);
    nft_auto_size(s, &params);
    s->params = params;
    s->group = push_array_aligned(arena, NftGroup, s->groups, 64);
    for (u32 g = 0; g < s->groups; ++g) {
        s->group[g].rows = push_array_aligned(arena, NftRow, s->cells, 64);
        s->group[g].sites = push_array_aligned(arena, NftSiteRow, s->sites, 64);
    }
    s->res.V_L = params.reservoir_L;
    s->res.T_C = params.fill_T_C;
    nft_update_lane_params(s);
    nft_update_crop_lanes(s);
    s->hyd_solved_hour = 0.0;

    f64 c[NftF_Count] = {};
    c[NftF_DO] = nft_do_sat_f64(params.fill_T_C);
    for (u32 i = 0; i < NFT_IONS; ++i) c[NFT_ION0 + i] = params.recipe[i];
    for (u32 k = 1; k < NftF_Count; ++k) s->res.mass_mg[k] = c[k] * s->res.V_L;
    s->ref[NftF_T] = (f32)params.fill_T_C;
    for (u32 k = 1; k < NftF_Count; ++k) s->ref[k] = (f32)c[k];
    CopyArray(s->inlet, s->ref, NftF_Count);
    for (u32 g = 0; g < s->groups; ++g) {
        NftGroup *grp = s->group + g;   // rows are zero deltas (arena zero-init): films start equal to the tank
        for (u32 j = 0; j < s->sites; ++j)
            for (u32 l = 0; l < NFT_LANES; ++l) {
                u32 rail = g * NFT_LANES + l;
                NftBank *bank = &params.bank[nft_rail_bank(s, rail) ? 1 : 0];
                // Staggered planting: site ages spread evenly over stagger_days (continuous harvest).
                f32 frac = s->sites > 1 ? (f32)((j * 7 + rail * 3) % s->sites) / (f32)s->sites : 0.0f;
                f32 age = frac * bank->stagger_days * 86400.0f;
                age = Minimum(age, grp->crop.cycle_s[l] * 0.95f);
                // a plant of that age on the logistic curve at full rate
                f32 w0 = grp->crop.w0[l], wmax = grp->crop.wmax[l];
                f32 e = expf(grp->crop.rgr_s[l] * age);
                grp->sites[j].W[l] = wmax * w0 * e / (wmax + w0 * (e - 1.0f));
                grp->sites[j].age[l] = age;
            }
    }
}

// ---------------------------------------------------------------------------
// Kernels (lane templates: V = f32x8 for 8 rails, or f32 for one rail)

// Plants: limitation factors, growth, uptake/transpiration rates for the next
// window, harvest + replant. Reads the site cells' current state.
template <class V>
SIM_HD void nft_plants(NftGroup *grp, u32 lane, u32 sites, u32 site_offset, u32 site_stride, f32 dt_window,
                       f32 light_gain, f32 dt_sub, f32 T_air, const f32 *ref) {
    typedef typename LaneTraits<V>::Mask M;
    const V zero = lane_splat((V *)0, 0.0), one = lane_splat((V *)0, 1.0);
    const NftCropLanes *c = &grp->crop;
    V rgr = lane_load_f32(c->rgr_s + lane, (V *)0) * lane_splat((V *)0, light_gain);
    V wmax = lane_load_f32(c->wmax + lane, (V *)0), w0 = lane_load_f32(c->w0 + lane, (V *)0);
    V cycle = lane_load_f32(c->cycle_s + lane, (V *)0);
    V tmin = lane_load_f32(c->t_min + lane, (V *)0), tlo = lane_load_f32(c->t_lo + lane, (V *)0);
    V thi = lane_load_f32(c->t_hi + lane, (V *)0), tmax = lane_load_f32(c->t_max + lane, (V *)0);
    V root_frac = lane_load_f32(c->root_frac + lane, (V *)0), transp = lane_load_f32(c->transp + lane, (V *)0);
    V o2s = lane_load_f32(c->o2_s + lane, (V *)0), docrit = lane_load_f32(c->do_crit + lane, (V *)0);
    V inv_vcell = lane_load_f32(grp->lp.inv_vcell_L + lane, (V *)0);
    V dtw = lane_splat((V *)0, dt_window), dts = lane_splat((V *)0, dt_sub);
    V tiny = lane_splat((V *)0, 1e-20);
    // heat toward bolting: air degree-days above the crop's threshold this window
    V tbolt = lane_load_f32(c->t_bolt + lane, (V *)0), bolt_dd = lane_load_f32(c->bolt_dd + lane, (V *)0);
    V heat = lane_max(lane_splat((V *)0, T_air) - tbolt, zero) * lane_splat((V *)0, dt_window / 86400.0);

    for (u32 j = 0; j < sites; ++j) {
        NftSiteRow *site = grp->sites + j;
        NftRow *cell = grp->rows + site_offset + j * site_stride;
        V T = lane_load_f32(cell->f[NftF_T] + lane, (V *)0) + lane_splat((V *)0, ref[NftF_T]);
        V DO = lane_load_f32(cell->f[NftF_DO] + lane, (V *)0) + lane_splat((V *)0, ref[NftF_DO]);
        V W = lane_load_f32(site->W + lane, (V *)0);
        V age = lane_load_f32(site->age + lane, (V *)0);

        // root-zone temperature: trapezoid 0 -> 1 -> 1 -> 0
        V rise = (T - tmin) / (tlo - tmin), fall = (tmax - T) / (tmax - thi);
        V fT = lane_min(lane_max(lane_min(rise, fall), zero), one);
        // dissolved O2: 0 at do_crit/2, 1 at do_crit
        V half = docrit * lane_splat((V *)0, 0.5);
        V fDO = lane_min(lane_max((DO - half) / half, zero), one);
        V gpot = rgr * fT * fDO * W * lane_max(one - W / wmax, zero);     // g DM / s
        V root = root_frac * W;

        // demand-limited, capacity-capped uptake (Liebig over nutrients)
        V cap[NFT_IONS];
        for (u32 i = 0; i < NFT_IONS; ++i) {
            V C = lane_max(lane_load_f32(cell->f[NFT_ION0 + i] + lane, (V *)0) + lane_splat((V *)0, ref[NFT_ION0 + i]), zero);
            V km = lane_splat((V *)0, (f64)nft_km[i]);
            cap[i] = lane_load_f32(c->imax_s[i] + lane, (V *)0) * root * fT * (C / (km + C));   // mg/s
        }
        V thousand = lane_splat((V *)0, 1000.0);
        V capN = cap[NftF_NO3 - NFT_ION0] + cap[NftF_NH4 - NFT_ION0];
        V caps[NftN_Count] = { capN, cap[NftF_P - NFT_ION0], cap[NftF_K - NFT_ION0], cap[NftF_Ca - NFT_ION0],
                               cap[NftF_Mg - NFT_ION0], cap[NftF_S - NFT_ION0] };
        V fN = one;
        V tissue[NftN_Count];
        for (u32 n = 0; n < NftN_Count; ++n) {
            tissue[n] = lane_load_f32(c->tissue[n] + lane, (V *)0);
            V demand = tissue[n] * gpot * thousand;                       // mg/s
            fN = lane_min(fN, caps[n] / lane_max(demand, tiny));
        }
        fN = lane_min(lane_max(fN, zero), one);
        V growth = gpot * fN;
        // uptake = tissue x actual growth; N split between NO3 and NH4 by capacity
        V upN = tissue[NftN_N] * growth * thousand;
        V nh4_share = cap[NftF_NH4 - NFT_ION0] / lane_max(capN, tiny);
        V up[NFT_IONS];
        up[NftF_NO3 - NFT_ION0] = upN * (one - nh4_share);
        up[NftF_NH4 - NFT_ION0] = upN * nh4_share;
        up[NftF_P - NFT_ION0] = tissue[NftN_P] * growth * thousand;
        up[NftF_K - NFT_ION0] = tissue[NftN_K] * growth * thousand;
        up[NftF_Ca - NFT_ION0] = tissue[NftN_Ca] * growth * thousand;
        up[NftF_Mg - NFT_ION0] = tissue[NftN_Mg] * growth * thousand;
        up[NftF_S - NFT_ION0] = tissue[NftN_S] * growth * thousand;
        // root respiration, Q10 = 2 around 20 C (quadratic of 2^(x/10), no libm)
        V x = T - lane_splat((V *)0, 20.0);
        V q10 = lane_fmadd(lane_fmadd(lane_splat((V *)0, 0.0024), x, lane_splat((V *)0, 0.0693)), x, one);
        V o2 = o2s * root * lane_max(q10, zero);

        for (u32 i = 0; i < NFT_IONS; ++i) {
            lane_store_f32(site->uptake[i] + lane, up[i]);
            lane_store_f32(site->drop[NFT_ION0 + i] + lane, up[i] * dts * inv_vcell);
        }
        for (u32 k = 0; k < NftF_Count; ++k) lane_store_f32(site->removed[k] + lane, zero);
        lane_store_f32(site->o2_rate + lane, o2);
        lane_store_f32(site->drop[NftF_DO] + lane, o2 * dts * inv_vcell);
        lane_store_f32(site->transp + lane, transp * growth);
        lane_store_f32(site->fT + lane, fT);
        lane_store_f32(site->fDO + lane, fDO);
        lane_store_f32(site->fN + lane, fN);

        // grow, age, harvest at maturity, replant
        W = lane_fmadd(growth, dtw, W);
        age = age + dtw;
        M ripe = lane_ge(age, cycle);
        V harvested = lane_load_f32(site->harvested_g + lane, (V *)0);
        V count = lane_load_f32(site->harvests + lane, (V *)0);
        V dd = lane_load_f32(site->heat_dd + lane, (V *)0) + heat;
        V bolted = lane_load_f32(site->bolted + lane, (V *)0);
        M was_bolting = lane_ge(dd, bolt_dd);
        lane_store_f32(site->harvested_g + lane, harvested + lane_select(ripe, W, zero));
        lane_store_f32(site->harvests + lane, count + lane_select(ripe, one, zero));
        lane_store_f32(site->bolted + lane, bolted + lane_select(lane_and(ripe, was_bolting), one, zero));
        lane_store_f32(site->heat_dd + lane, lane_select(ripe, zero, dd));
        lane_store_f32(site->W + lane, lane_select(ripe, w0, W));
        lane_store_f32(site->age + lane, lane_select(ripe, zero, age));
    }
}

// Transport: `substeps` upwind advection + wall/radiant heat + O2 exchange +
// site uptake, all rows of one group, in deltas from `ref` (the inlet, so the
// inflow delta is 0). Rows are swept outlet -> inlet so each row reads its
// upstream neighbour's old value. Outlet deltas are summed for the return flow.
template <class V>
SIM_HD void nft_transport(NftGroup *grp, u32 lane, u32 cells, u32 sites, u32 site_offset, u32 site_stride,
                          const f32 *ref, f32 T_amb, f32 radiant_now, u32 substeps) {
    const V zero = lane_splat((V *)0, 0.0);
    V c = lane_load_f32(grp->lp.courant + lane, (V *)0);
    V kw = lane_load_f32(grp->lp.k_wall + lane, (V *)0);
    V kr = lane_load_f32(grp->lp.k_rad + lane, (V *)0) * lane_splat((V *)0, radiant_now);
    V ko = lane_load_f32(grp->lp.k_o2 + lane, (V *)0);
    V tamb_rel = lane_splat((V *)0, T_amb - ref[NftF_T]);      // ambient minus reference, in delta space
    V refT = lane_splat((V *)0, ref[NftF_T]), refDO = lane_splat((V *)0, ref[NftF_DO]);
    V floor_[NftF_Count];                                       // delta at zero concentration
    for (u32 k = 0; k < NftF_Count; ++k) floor_[k] = lane_splat((V *)0, -ref[k]);
    V out[NftF_Count];
    for (u32 k = 0; k < NftF_Count; ++k) out[k] = zero;
    u32 last_site_row = site_offset + (sites ? (sites - 1) * site_stride : 0);

    for (u32 s = 0; s < substeps; ++s) {
        NftRow *last = grp->rows + cells - 1;
        for (u32 k = 0; k < NftF_Count; ++k) out[k] = out[k] + lane_load_f32(last->f[k] + lane, (V *)0);
        for (u32 j = cells; j-- > 0;) {
            NftRow *row = grp->rows + j;
            V x[NftF_Count];
            if (j > 0) {
                NftRow *up = row - 1;
                for (u32 k = 0; k < NftF_Count; ++k) {
                    V v = lane_load_f32(row->f[k] + lane, (V *)0);
                    x[k] = lane_fmadd(c, lane_load_f32(up->f[k] + lane, (V *)0) - v, v);
                }
            } else {
                for (u32 k = 0; k < NftF_Count; ++k) {
                    V v = lane_load_f32(row->f[k] + lane, (V *)0);
                    x[k] = lane_fnmadd(c, v, v);                         // inflow delta is 0
                }
            }
            // heat: wall conduction to ambient + absorbed radiant load
            x[NftF_T] = lane_fmadd(kw, tamb_rel - x[NftF_T], x[NftF_T]) + kr;
            // O2 exchange with the air in the gully (saturation at the absolute temperature)
            x[NftF_DO] = lane_fmadd(ko, nft_do_sat(refT + x[NftF_T]) - (refDO + x[NftF_DO]), x[NftF_DO]);
            // plant site: uptake and root respiration, booked as what was actually removed
            if (j >= site_offset && j <= last_site_row && (j - site_offset) % site_stride == 0) {
                NftSiteRow *site = grp->sites + (j - site_offset) / site_stride;
                for (u32 k = 1; k < NftF_Count; ++k) {
                    V before = x[k];
                    x[k] = lane_max(x[k] - lane_load_f32(site->drop[k] + lane, (V *)0), floor_[k]);
                    lane_store_f32(site->removed[k] + lane, lane_load_f32(site->removed[k] + lane, (V *)0) + (before - x[k]));
                }
            }
            for (u32 k = 0; k < NftF_Count; ++k) lane_store_f32(row->f[k] + lane, x[k]);
        }
    }
    for (u32 k = 0; k < NftF_Count; ++k) lane_store_f32(grp->out_sum[k] + lane, out[k]);
}

// Moves a group's deltas to a new reference: x = ref + d = new_ref + (d + ref - new_ref).
template <class V>
SIM_HD void nft_rebase(NftGroup *grp, u32 lane, u32 cells, const f32 *shift) {
    V sh[NftF_Count];
    for (u32 k = 0; k < NftF_Count; ++k) sh[k] = lane_splat((V *)0, shift[k]);
    for (u32 j = 0; j < cells; ++j)
        for (u32 k = 0; k < NftF_Count; ++k)
            lane_store_f32(grp->rows[j].f[k] + lane, lane_load_f32(grp->rows[j].f[k] + lane, (V *)0) + sh[k]);
}

// One whole window for one group: plants, then transport. The unit of work
// handed to a thread (temporal blocking: the group stays in that core's L1).
// `shift` = old reference - new reference (all zero: no rebase pass).
template <class V>
SIM_HD void nft_group_window(NftSystem *s, u32 g, const f32 *ref, const f32 *shift, b32 rebase, f32 light_gain) {
    NftGroup *grp = s->group + g;
    for (u32 lane = 0; lane < NFT_LANES; lane += LaneTraits<V>::width) {
        if (rebase) nft_rebase<V>(grp, lane, s->cells, shift);
        nft_plants<V>(grp, lane, s->sites, s->site_offset, s->site_stride, (f32)s->window_s, light_gain, (f32)s->dt_sub,
                      s->T_amb, ref);
        nft_transport<V>(grp, lane, s->cells, s->sites, s->site_offset, s->site_stride, ref, s->T_amb,
                         s->light * s->params.radiant_W_m2, s->substeps);
    }
}

// ---------------------------------------------------------------------------
// Environment and reservoir

internal void nft_environment(NftSystem *s, f64 t) {
    NftParams *p = &s->params;
    f64 hour = fmod(t / 3600.0, 24.0);
    s->T_amb = (f32)(p->ambient_mean_C + p->ambient_amp_C * sin(2.0 * 3.14159265358979 * (hour - 9.0) / 24.0));
    f64 since_on = fmod(hour - p->lights_on_h + 24.0, 24.0);
    s->light = since_on < p->photoperiod_h ? 1.0f : 0.0f;
}

// Growth multiplier while lights are on / off so the daily mean is 1:
// photosynthesis by day, a little (reserves) by night.
internal f32 nft_light_gain(NftSystem *s) {
    f32 day = s->params.photoperiod_h / 24.0f, dark_gain = 0.1f;
    f32 day_gain = day > 0.0f ? (1.0f - (1.0f - day) * dark_gain) / day : 0.0f;
    return s->light > 0.5f ? day_gain : dark_gain;
}

// Dosing stock (mg/L), 100x concentrate. Uptake-matched: ion ratios from the
// crops' tissue composition (both banks, weighted by rails), scaled to the
// fill recipe's EC; N supplied as NO3 / NH4 in the recipe's proportion.
internal void nft_stock_composition(NftSystem *s, f64 *stock) {
    NftParams *p = &s->params;
    for (u32 k = 0; k < NftF_Count; ++k) stock[k] = 0.0;
    f64 recipe_c[NftF_Count] = {};
    for (u32 i = 0; i < NFT_IONS; ++i) recipe_c[NFT_ION0 + i] = p->recipe[i];
    if (!p->dose_matched) {
        for (u32 i = 0; i < NFT_IONS; ++i) stock[NFT_ION0 + i] = 100.0 * p->recipe[i];
        return;
    }
    f64 tissue[NftN_Count] = {};
    u32 rails_b = s->rails / 2, rails_a = s->rails - rails_b;
    for (u32 n = 0; n < NftN_Count; ++n)
        tissue[n] = (rails_a * (f64)nft_crops[p->bank[0].crop].tissue[n] + rails_b * (f64)nft_crops[p->bank[1].crop].tissue[n]) / (f64)s->rails;
    f64 nh4_share = (p->recipe[1] + p->recipe[0]) > 0 ? (f64)p->recipe[1] / ((f64)p->recipe[0] + p->recipe[1]) : 0.0;
    f64 c[NftF_Count] = {};
    c[NftF_NO3] = tissue[NftN_N] * (1.0 - nh4_share);
    c[NftF_NH4] = tissue[NftN_N] * nh4_share;
    c[NftF_P] = tissue[NftN_P]; c[NftF_K] = tissue[NftN_K]; c[NftF_Ca] = tissue[NftN_Ca];
    c[NftF_Mg] = tissue[NftN_Mg]; c[NftF_S] = tissue[NftN_S];
    f64 scale = 100.0 * nft_ec_mS(recipe_c) / Maximum(nft_ec_mS(c), 1e-30);
    for (u32 k = NFT_ION0; k < NftF_Count; ++k) stock[k] = c[k] * scale;
}

// Mixes the window's return flow into the tank and applies its heat budget,
// chiller, aeration, transpiration, top-off and dosing. f64 throughout.
internal void nft_reservoir_window(NftSystem *s) {
    NftParams *p = &s->params;
    NftReservoir *r = &s->res;
    f64 dt = s->window_s;
    f64 inv_sub = 1.0 / (f64)s->substeps;

    // return flow (rails carried tank water out at the window's start conditions)
    f64 q_tot = 0.0, ret[NftF_Count] = {}, uptake_window[NFT_IONS] = {}, transp_g_s = 0.0;
    for (u32 g = 0; g < s->groups; ++g) {
        NftGroup *grp = s->group + g;
        for (u32 l = 0; l < NFT_LANES; ++l) {
            f64 q = grp->lp.flow_L_s[l];
            if (q <= 0.0) continue;
            q_tot += q;
            // what left the rails: inlet + mean deviation (L/s x mg/L, or x C)
            for (u32 k = 0; k < NftF_Count; ++k) ret[k] += q * ((f64)s->inlet[k] + (f64)grp->out_sum[k][l] * inv_sub);
            f64 vcell = 1.0 / (f64)grp->lp.inv_vcell_L[l];
            for (u32 j = 0; j < s->sites; ++j) {
                for (u32 i = 0; i < NFT_IONS; ++i) uptake_window[i] += (f64)grp->sites[j].removed[NFT_ION0 + i][l] * vcell;
                transp_g_s += (f64)grp->sites[j].transp[l];
            }
        }
    }
    // the tank is debited exactly what the rails received (the f32 inlet), so
    // the f32/f64 boundary cannot create or destroy mass
    for (u32 k = 1; k < NftF_Count; ++k) r->mass_mg[k] += dt * (ret[k] - q_tot * (f64)s->inlet[k]);
    for (u32 i = 0; i < NFT_IONS; ++i) s->acc.uptake_mg[i] += uptake_window[i];

    // heat: return water, tank walls, pump, chiller (hysteresis on the tank temperature)
    f64 heat_W = NFT_RHO_CP * 1e-3 * (ret[NftF_T] - q_tot * (f64)s->inlet[NftF_T])   // rho cp (L/s -> m^3/s) dT
                 + p->tank_ua_W_K * ((f64)s->T_amb - r->T_C) + nft_pump_heat_W(p, &s->hyd);
    if (p->chiller_on) {
        if (r->T_C > p->chiller_set_C + p->chiller_band_C) r->chiller_running = 1;
        if (r->T_C < p->chiller_set_C - p->chiller_band_C) r->chiller_running = 0;
    } else {
        r->chiller_running = 0;
    }
    if (r->chiller_running) {
        heat_W -= p->chiller_W;
        s->acc.chiller_heat_J += p->chiller_W * dt;
        s->acc.chiller_elec_J += p->chiller_W / p->chiller_cop * dt;
        s->acc.chiller_on_s += dt;
    }
    s->acc.pump_J += s->hyd.pump_power_W * dt;
    r->T_C += heat_W * dt / (NFT_RHO_CP * 1e-3 * r->V_L);

    // tank aeration toward saturation
    f64 do_c = nft_reservoir_conc(s, NftF_DO);
    r->mass_mg[NftF_DO] += dt * p->tank_kla_per_s * (nft_do_sat_f64(r->T_C) - do_c) * r->V_L;

    // water leaves through the plants
    f64 transp_L = transp_g_s * dt * 1e-3;
    r->V_L -= transp_L;
    s->acc.transpired_L += transp_L;

    // top-off with fresh water (no ions) back to the set volume
    if (p->topoff_on && r->V_L < 0.95 * p->reservoir_L) {
        f64 add = p->reservoir_L - r->V_L;
        r->T_C = (r->T_C * r->V_L + p->fill_T_C * add) / p->reservoir_L;
        r->mass_mg[NftF_DO] += nft_do_sat_f64(p->fill_T_C) * add;
        r->V_L = p->reservoir_L;
        s->acc.topoff_L += add;
    }
    // EC dosing, rate-limited. Stock ion ratios: either the fill recipe, or
    // matched to what the plants actually take up (their tissue composition),
    // so ions neither pile up (Ca, S) nor run out (K) in a recirculating system.
    if (p->dosing_on) {
        f64 c[NftF_Count];
        for (u32 k = 0; k < NftF_Count; ++k) c[k] = nft_reservoir_conc(s, k);
        f64 ec = nft_ec_mS(c);
        f64 stock[NftF_Count] = {};
        nft_stock_composition(s, stock);
        f64 ec_stock = nft_ec_mS(stock);
        if (ec < p->ec_set_mS && ec_stock > 0.0) {
            // stock volume that would close the gap (EC is linear in the doses), capped by the pump
            f64 need = (p->ec_set_mS - ec) * r->V_L / ec_stock;
            f64 dose = Minimum(need, (f64)p->dose_L_per_h_max / 3600.0 * dt);
            for (u32 i = 0; i < NFT_IONS; ++i) {
                f64 m = stock[NFT_ION0 + i] * dose;
                r->mass_mg[NFT_ION0 + i] += m;
                s->acc.dosed_mg[i] += m;
            }
            r->V_L += dose;
            s->acc.dosed_L += dose;
        }
    }
    // scheduled solution change: drain the tank, refill with fresh recipe
    if (p->dump_days > 0.0f && s->t + dt - s->acc.last_dump_t >= (f64)p->dump_days * 86400.0) {
        for (u32 i = 0; i < NFT_IONS; ++i) {
            s->acc.dumped_mg[i] += r->mass_mg[NFT_ION0 + i];
            r->mass_mg[NFT_ION0 + i] = (f64)p->recipe[i] * p->reservoir_L;
            s->acc.dosed_mg[i] += r->mass_mg[NFT_ION0 + i];
        }
        s->acc.dumped_L += r->V_L;
        s->acc.topoff_L += p->reservoir_L;
        r->V_L = p->reservoir_L;
        r->T_C = p->fill_T_C;
        r->mass_mg[NftF_DO] = nft_do_sat_f64(p->fill_T_C) * p->reservoir_L;
        s->acc.last_dump_t = s->t + dt;
    }
}

// Accumulates harvests per bank (sites keep running totals; the system keeps
// the bank sums so the UI does not have to walk every site).
internal void nft_collect_harvest(NftSystem *s) {
    f64 g_bank[2] = {}, n_bank[2] = {}, b_bank[2] = {};
    for (u32 g = 0; g < s->groups; ++g)
        for (u32 l = 0; l < NFT_LANES; ++l) {
            u32 rail = g * NFT_LANES + l;
            if (rail >= s->rails) continue;
            u32 b = nft_rail_bank(s, rail) ? 1 : 0;
            for (u32 j = 0; j < s->sites; ++j) {
                g_bank[b] += s->group[g].sites[j].harvested_g[l];
                n_bank[b] += s->group[g].sites[j].harvests[l];
                b_bank[b] += s->group[g].sites[j].bolted[l];
            }
        }
    for (u32 b = 0; b < 2; ++b) { s->acc.harvest_g[b] = g_bank[b]; s->acc.harvests[b] = n_bank[b]; s->acc.bolted[b] = b_bank[b]; }
}

// Inlet values for the rails this window: the tank as it is now.
// This window's inlet (the tank now) becomes the rails' new reference.
// Returns whether the deltas must be rebased, with shift = old ref - new ref.
internal b32 nft_inlet(NftSystem *s, f32 *shift) {
    f32 inlet[NftF_Count];
    inlet[NftF_T] = (f32)s->res.T_C;
    for (u32 k = 1; k < NftF_Count; ++k) inlet[k] = (f32)nft_reservoir_conc(s, k);
    b32 rebase = 0;
    for (u32 k = 0; k < NftF_Count; ++k) {
        shift[k] = s->ref[k] - inlet[k];
        rebase |= shift[k] != 0.0f;
    }
    CopyArray(s->inlet, inlet, NftF_Count);   // the reservoir debits exactly this
    CopyArray(s->ref, inlet, NftF_Count);
    return rebase;
}

// Absolute value of a rail cell.
internal f32 nft_abs(NftSystem *s, u32 g, u32 j, u32 field, u32 lane) {
    return s->ref[field] + s->group[g].rows[j].f[field][lane];
}

// ---------------------------------------------------------------------------
// Driver: one window = [every group in parallel: plants + transport] + reservoir.

struct NftWork {
    NftSystem *s;
    u32 first, end;            // groups
    f32 shift[NftF_Count];
    b32 rebase;
    f32 light_gain;
};

internal PLATFORM_WORK_QUEUE_CALLBACK(nft_work_window) {
    NftWork *w = (NftWork *)data;
    for (u32 g = w->first; g < w->end; ++g) nft_group_window<f32x8>(w->s, g, w->s->ref, w->shift, w->rebase, w->light_gain);
}

// Water viscosity follows the tank temperature: re-solve the network hourly.
internal void nft_maybe_resolve(NftSystem *s) {
    f64 hour = floor(s->t / 3600.0);
    if (hour != s->hyd_solved_hour) {
        s->hyd_solved_hour = hour;
        nft_update_lane_params(s);
    }
}

internal void nft_window(NftSystem *s, PlatformAPI *api, PlatformWorkQueue *queue, u32 threads, Arena *frame) {
    TIMED_FUNCTION();
    nft_maybe_resolve(s);
    nft_environment(s, s->t);
    f32 shift[NftF_Count];
    b32 rebase = nft_inlet(s, shift);
    f32 light_gain = nft_light_gain(s);
    if (!queue || threads <= 1 || s->groups <= 1) {
        for (u32 g = 0; g < s->groups; ++g) nft_group_window<f32x8>(s, g, s->ref, shift, rebase, light_gain);
    } else {
        u32 chunks = Minimum(s->groups, threads * 2);
        u32 per = (s->groups + chunks - 1) / chunks;
        TempMemory temp = begin_temp(frame);
        NftWork *work = push_array(frame, NftWork, chunks);
        u32 n = 0;
        for (u32 g = 0; g < s->groups; g += per) {
            NftWork *w = work + n++;
            w->s = s;
            w->first = g;
            w->end = Minimum(s->groups, g + per);
            CopyArray(w->shift, shift, NftF_Count);
            w->rebase = rebase;
            w->light_gain = light_gain;
            api->add_work_entry(queue, nft_work_window, w);
        }
        api->complete_all_work(queue);
        end_temp(temp);
    }
    nft_reservoir_window(s);
    s->t += s->window_s;
}

// --- diagnostics ------------------------------------------------------------------

// Mass of a field (mg) held in the rail films.
internal f64 nft_rail_mass_mg(NftSystem *s, u32 field) {
    f64 m = 0.0;
    for (u32 g = 0; g < s->groups; ++g)
        for (u32 l = 0; l < NFT_LANES; ++l) {
            if (g * NFT_LANES + l >= s->rails) continue;
            f64 v = 1.0 / (f64)s->group[g].lp.inv_vcell_L[l];
            for (u32 j = 0; j < s->cells; ++j) m += ((f64)s->ref[field] + (f64)s->group[g].rows[j].f[field][l]) * v;
        }
    return m;
}

internal f64 nft_rail_volume_L(NftSystem *s) {
    if (!s->groups) return 0.0;
    return (f64)s->rails * (f64)s->cells / (f64)s->group[0].lp.inv_vcell_L[0];
}

// Mean of a field over the rails of a bank at cell j.
internal f64 nft_bank_cell_mean(NftSystem *s, u32 bank, u32 field, u32 j) {
    f64 sum = 0.0; u32 n = 0;
    for (u32 r = 0; r < s->rails; ++r) {
        if ((nft_rail_bank(s, r) ? 1u : 0u) != bank) continue;
        sum += nft_abs(s, r / NFT_LANES, j, field, r % NFT_LANES);
        ++n;
    }
    return n ? sum / n : 0.0;
}

// Mean plant dry mass and limitation factors of a bank.
internal void nft_bank_plants(NftSystem *s, u32 bank, f64 *mean_W, f64 *mean_fT, f64 *mean_fN, f64 *mean_fDO) {
    f64 w = 0, ft = 0, fn = 0, fd = 0; u32 n = 0;
    for (u32 r = 0; r < s->rails; ++r) {
        if ((nft_rail_bank(s, r) ? 1u : 0u) != bank) continue;
        NftGroup *g = s->group + r / NFT_LANES;
        u32 l = r % NFT_LANES;
        for (u32 j = 0; j < s->sites; ++j) {
            w += g->sites[j].W[l]; ft += g->sites[j].fT[l]; fn += g->sites[j].fN[l]; fd += g->sites[j].fDO[l];
            ++n;
        }
    }
    f64 inv = n ? 1.0 / n : 0.0;
    *mean_W = w * inv; *mean_fT = ft * inv; *mean_fN = fn * inv; *mean_fDO = fd * inv;
}
