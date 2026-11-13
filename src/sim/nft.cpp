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

// Charge bookkeeping for alkalinity (nft_ph.cpp): mg per meq of each ion at its
// alkalinity reference level (NO3-, NH4+ total, P total as 1 per P (H2PO4-
// is the zero level), K+, Ca++, Mg++, SO4--) and its sign (+ cation, - anion).
global const f64 nft_meq_mg[NFT_IONS] = { 14.007, 14.007, 30.974, 39.098, 20.039, 12.1525, 16.03 };
global const f64 nft_charge_sign[NFT_IONS] = { -1, +1, -1, +1, +1, +1, -1 };

// Plant nutrients (demand side): N is met from NO3 and NH4.
enum NftNutrient { NftN_N, NftN_P, NftN_K, NftN_Ca, NftN_Mg, NftN_S, NftN_Count };
global const char *const nft_nutrient_names[NftN_Count] = { "N", "P", "K", "Ca", "Mg", "S" };

// Michaelis-Menten half-saturation (mg/L) per ion field (NO3..S).
global const f32 nft_km[NFT_IONS] = { 1.0f, 0.3f, 0.2f, 1.5f, 3.0f, 1.5f, 1.0f };

#include "nft_water.cpp"
#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

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
    f32 sla_m2_g;            // specific leaf area of the shoot, m^2 per g dry mass
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
    // name                     cycle  w0     wmax  rgr     tmin lo  hi  tmax root   SLA     N       P       K       Ca      Mg       S       o2    DOc   tbolt dd
    { "Cilantro",                 40, 0.06f,  4.0f, 0.22f,   5, 17, 22, 29, 0.14f, 0.035f, {0.045f, 0.005f, 0.055f, 0.013f, 0.0035f, 0.003f}, 1.6f, 4.0f, 25, 12 },
    { "Arugula",                  28, 0.05f,  3.5f, 0.30f,   4, 18, 22, 28, 0.12f, 0.040f, {0.050f, 0.006f, 0.060f, 0.025f, 0.0040f, 0.010f}, 1.6f, 4.0f, 26, 10 },
    { "Lettuce (butterhead)",     35, 0.12f, 12.0f, 0.30f,   4, 18, 22, 30, 0.12f, 0.045f, {0.045f, 0.006f, 0.070f, 0.012f, 0.0035f, 0.003f}, 1.5f, 4.0f, 27, 15 },
    { "Basil",                    42, 0.15f, 18.0f, 0.24f,  10, 22, 27, 34, 0.12f, 0.035f, {0.040f, 0.006f, 0.050f, 0.018f, 0.0060f, 0.004f}, 1.8f, 4.0f, 32, 30 },
    { "Kale",                     50, 0.15f, 28.0f, 0.20f,   4, 16, 21, 28, 0.15f, 0.025f, {0.045f, 0.006f, 0.045f, 0.025f, 0.0040f, 0.008f}, 1.5f, 4.0f, 27, 20 },
    { "Spinach",                  30, 0.10f,  9.0f, 0.30f,   3, 15, 20, 26, 0.12f, 0.030f, {0.045f, 0.005f, 0.075f, 0.010f, 0.0090f, 0.003f}, 1.6f, 4.5f, 24, 8 },
    { "Strawberry (day-neutral)",120, 1.00f, 70.0f, 0.08f,   6, 16, 22, 28, 0.20f, 0.015f, {0.025f, 0.004f, 0.030f, 0.012f, 0.0040f, 0.002f}, 1.2f, 4.0f, 30, 40 },
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

// Per-cell coefficients (lanes = channels). Root mats make the film deeper
// and slower cell by cell, so volume, Courant number, heat capacity and O2
// transfer are per cell. Split by use: the transport loop reads only the hot
// row (128 B per cell next to the 288 B field row); the rest is touched once
// per window or when roots are re-solved (nft_group_cells).
struct alignas(32) NftCellHot {
    f32 c[NFT_LANES];           // Q dt / V      upwind Courant number (flux form: Q is the same through every cell)
    f32 kT[NFT_LANES];          // dT = -kT T + bT per substep (wall + latent, linearized per window)
    f32 bT[NFT_LANES];
    f32 ko[NFT_LANES];          // kL A dt / V   O2 transfer per substep
};
struct alignas(32) NftCellCold {
    f64 vol_L[NFT_LANES];       // water volume the books use: exactly Q dt / c (f64 of the f32 c), so the
                                // f32 upwind fluxes telescope exactly: V_j c_j = Q dt in every cell
    f32 inv_v[NFT_LANES];       // 1 / water volume (L^-1), for the kernels
    f32 kw[NFT_LANES];          // U P dx dt / (rho cp V)          wall exchange per substep
    f32 kr[NFT_LANES];          // a w dx dt / (rho cp V)          per substep per W/m^2 radiant
    f32 ge[NFT_LANES];          // lambda ev / (rho cp V)          latent cooling per substep per Pa (K/Pa)
    f32 ev[NFT_LANES];          // evaporation per substep per Pa of vapour deficit (kg/Pa)
    f32 depth[NFT_LANES];       // film depth incl. roots, m
    f32 phi[NFT_LANES];         // root volume fraction of the wetted layer
    f32 vel[NFT_LANES];         // pore velocity, m/s
};

// Per-lane (per-channel) constants.
struct NftLaneParams {
    f32 flow_L_s[NFT_LANES];    // 0 for padding lanes
    f32 depth_m[NFT_LANES], velocity_m_s[NFT_LANES];   // root-free film
    f32 evap_kg[NFT_LANES];     // evaporated from the film this window (kernel output)
};

struct NftCropLanes {
    f32 rgr_s[NFT_LANES], w0[NFT_LANES], wmax[NFT_LANES], cycle_s[NFT_LANES];
    f32 t_min[NFT_LANES], t_lo[NFT_LANES], t_hi[NFT_LANES], t_max[NFT_LANES];
    f32 root_frac[NFT_LANES], sla[NFT_LANES], o2_s[NFT_LANES], do_crit[NFT_LANES];
    f32 t_bolt[NFT_LANES], bolt_dd[NFT_LANES];
    f32 tissue[NftN_Count][NFT_LANES];    // g/g
    f32 imax_s[NFT_IONS][NFT_LANES];      // mg / (g root s) at saturation, per ion field
};

struct alignas(64) NftGroup {   // one cache-line-aligned block per 8-rail group; owned by one thread per window
    NftLaneParams lp;
    NftCropLanes crop;
    f32 out_sum[NftF_Count][NFT_LANES];   // sum over the window's substeps of (outlet - inlet): small, so f32 sums stay exact
    NftRow *rows;                          // cells
    NftCellHot *hot;                       // per-cell coefficients, parallel to rows
    NftCellCold *cold;
    NftSiteRow *sites;                     // site rows
    // root-mat holdup booked since the last reservoir step: water moved from the
    // tank into the channels (L) and what it carried (L x mg/L, or L x C for T)
    f64 hold_dV, hold_dm[NftF_Count];
    u32 overflow_cells;                    // cells whose film tops the channel wall
    f32 max_depth_m;
};

// Every lane array must start on a 32-byte boundary for aligned f32x8 loads.
static_assert(sizeof(NftRow) % 32 == 0, "NftRow lane arrays");
static_assert(sizeof(NftCellHot) % 32 == 0 && sizeof(NftCellCold) % 32 == 0, "per-cell lane arrays");
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
    f32 radiant_W_m2;           // on rail tops / canopy while lights are on
    f32 rh;                     // ambient relative humidity (0..1)
    f32 co2_ppm;                // air CO2 the films equilibrate with
    f32 kla_film_per_s;         // O2 transfer of the root-free film (scales as 1/depth)
    // roots and water
    f32 root_diam_mm;           // fine-root diameter for the mat permeability
    f32 root_fresh_dry;         // fresh / dry mass of roots (volume ~ fresh mass at 1 g/cm^3)
    f32 root_phi_max;           // root volume fraction of a packed mat
    f32 channel_pitch_m;        // channel centre spacing: ground area per plant = pitch x plant spacing
    f32 tissue_water;           // g water per g dry mass grown (held in the shoot)
    f32 channel_open_frac;      // film area open to air (planting holes, gaps)
    f32 tank_open_frac;         // reservoir surface not covered by the lid
    f32 h_conv_W_m2K;           // convective coefficient over exposed water (evaporation via Lewis)
    f32 baille_A, baille_B;     // transpiration: A (radiation, -) and B (W/m^2 per kPa per LAI)
    f32 night_stomata;          // B-term fraction with lights off
    // pH
    b32 ph_control;
    f32 ph_set, ph_band;
    f32 acid_mmol_per_h_max;    // HNO3 (1 M stock)
    f32 tap_alk_meq;            // alkalinity of top-off / fill water
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
    p.rh = 0.60f;
    p.co2_ppm = 420.0f;
    p.kla_film_per_s = 0.01f;
    p.root_diam_mm = 0.4f;
    p.root_fresh_dry = 15.0f;
    p.root_phi_max = 0.35f;
    p.channel_pitch_m = 0.20f;
    p.tissue_water = 14.0f;
    p.channel_open_frac = 0.08f;
    p.tank_open_frac = 0.10f;
    p.h_conv_W_m2K = 5.0f;
    p.baille_A = 0.30f;       // Baille et al. 1994 (greenhouse crops): ~0.3 of absorbed radiation to latent heat
    p.baille_B = 16.0f;       //   and ~15-20 W m^-2 kPa^-1 per unit LAI
    p.night_stomata = 0.25f;
    p.ph_control = 1;
    p.ph_set = 5.8f;
    p.ph_band = 0.2f;
    p.acid_mmol_per_h_max = 50.0f;
    p.tap_alk_meq = 1.5f;     // moderately hard tap water (75 mg/L as CaCO3)
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
    f64 evap_channel_L, evap_tank_L, evap_heat_J;
    f64 acid_mmol, base_mmol, acid_L, base_L;   // 1 M HNO3 / 1 M KOH
    f64 holdup_L;                       // water held in the channels (follows the root mats)
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
    NftHydraulics hyd;                 // operating point of the network
    f64 hyd_solved_T;                  // tank temperature it was solved at
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
    f32 rh;                            // ambient relative humidity this window
    f64 R_meq;                         // untracked charge (Na, Cl, tap bicarbonate...), meq over tank + channels
    f64 pH;                            // tank pH (solved every window)
    b32 acid_running, base_running;    // pH controller (hysteresis: start outside the band, stop at the set point)
    u64 window_index;
    b32 cells_force;                   // re-solve every cell next window (hydraulics changed)
    u32 overflow_cells;                // cells whose film tops the channel wall (last solve)
    f32 max_depth_m;
    f64 evap_channel_L_s, evap_tank_L_s, transp_L_s;   // rates over the last window
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

// Network operating point, then per-channel root-free film and the substep.
// Called on rebuild, on any parameter change, and when the tank has moved 1 K
// (water viscosity). Cells are re-solved by their group at the next window.
internal void nft_update_lane_params(NftSystem *s) {
    NftParams *p = &s->params;
    f64 T = s->res.V_L > 0.0 ? s->res.T_C : (f64)p->fill_T_C;
    s->hyd_solved_T = T;
    if (p->hyd.balance_taps) nft_hydraulics_balance(&p->hyd, s->rails, T, &s->hyd);
    else nft_hydraulics_solve(&p->hyd, s->rails, T, &s->hyd);
    const NftProfile *prof = nft_profiles + (p->hyd.profile % NftProfile_Count);
    f64 w = prof->width_m;
    f64 dx = s->cell_m;
    f64 v_max = 1e-6;
    for (u32 g = 0; g < s->groups; ++g) {
        NftLaneParams *lp = &s->group[g].lp;
        for (u32 l = 0; l < NFT_LANES; ++l) {
            u32 rail = g * NFT_LANES + l;
            b32 live = rail < s->rails;
            u32 rr = live ? rail : 0;
            f64 q = s->hyd.q_channel[rr];
            f64 S = p->hyd.slope[nft_rail_bank(s, rr) ? 1 : 0];
            f64 depth = q > 0.0 ? nft_film_depth(q, w, S, s->hyd.nu) : 1e-4;
            f64 vel = q > 0.0 ? q / (w * depth) : 0.0;
            if (live) v_max = Maximum(v_max, vel);
            lp->flow_L_s[l] = live ? (f32)(q * 1000.0) : 0.0f;
            lp->depth_m[l] = (f32)depth;
            lp->velocity_m_s[l] = (f32)vel;
        }
    }
    // substep: Courant 0.9 on the fastest root-free channel (roots only slow a
    // cell down, nft_root_film), an integer count per window
    s->window_s = 60.0;
    s->substeps = (u32)ceil(s->window_s / (0.9 * dx / v_max));
    if (s->substeps < 1) s->substeps = 1;
    s->dt_sub = s->window_s / (f64)s->substeps;
    s->cells_force = 1;
}

// Per-cell film of one group from its plants' current root mass: depth, root
// fraction, water volume and every per-substep coefficient. Scalar f64, run by
// the group's own thread at the start of a window (nft_cells_due). With `book`, the change
// of each cell's water volume is booked against the tank (hold_dV, hold_dm):
// water drawn into a thickening mat leaves the tank with the tank-side value
// of the cell, so mass, water and heat stay closed.
internal void nft_group_cells(NftSystem *s, u32 g, const f32 *ref, b32 book) {
    NftParams *p = &s->params;
    NftGroup *grp = s->group + g;
    const NftProfile *prof = nft_profiles + (p->hyd.profile % NftProfile_Count);
    f64 w = prof->width_m, perim = prof->perimeter_m, dx = s->cell_m, dt = s->dt_sub;
    f64 nu = s->hyd.nu;
    f64 d_root = Maximum((f64)p->root_diam_mm, 0.01) * 1e-3;
    f64 phi_max = Clamp((f64)p->root_phi_max, 0.01, 0.9);
    f64 lam = nft_latent(s->res.T_C);
    f64 ev_coef = nft_evap_coef(p->h_conv_W_m2K) * Clamp((f64)p->channel_open_frac, 0.0, 1.0) * w * dx * dt;   // kg/Pa per substep
    f64 root_scale = (f64)p->root_fresh_dry * 1e-6 / ((f64)s->site_stride * w * dx);   // g dry root -> m of root per floor area
    f64 wall = prof->height_m;
    u32 overflow = 0;
    f64 h_max = 0.0;
    for (u32 l = 0; l < NFT_LANES; ++l) {
        u32 rail = g * NFT_LANES + l;
        b32 live = rail < s->rails;
        u32 rr = live ? rail : 0;
        u32 bank = nft_rail_bank(s, rr) ? 1 : 0;
        const NftWrap *wrap = nft_wraps + p->bank[bank].wrap;
        f64 q = live ? s->hyd.q_channel[rr] : 0.0;
        f64 S = p->hyd.slope[bank];
        f64 h0 = grp->lp.depth_m[l];
        f64 kL = (f64)p->kla_film_per_s * h0;           // m/s: the root-free film's kLa = kL / h0
        f64 rf = grp->crop.root_frac[l];
        for (u32 j = 0; j < s->cells; ++j) {
            NftCellHot *hot = grp->hot + j;
            NftCellCold *cold = grp->cold + j;
            u32 site = j / s->site_stride;              // the plant whose roots spread over this cell
            f64 tr = rf * (f64)grp->sites[site].W[l] * root_scale;
            f64 h, phi;
            nft_root_film(q / w, S, nu, h0, tr, d_root, phi_max, cold->depth[l], &h, &phi);
            f64 hw = h * (1.0 - phi);                   // water depth
            f64 V = w * dx * hw;                        // m^3 of water
            f64 QdtL = (f64)grp->lp.flow_L_s[l] * dt;   // L per substep, from the same f32 flow the reservoir books
            f32 c = (f32)(QdtL / (V * 1000.0));
            f64 vol = live && c > 0.0f ? QdtL / (f64)c : V * 1000.0;
            if (book && live && cold->vol_L[l] > 0.0) {
                f64 dV = vol - cold->vol_L[l];
                grp->hold_dV += dV;
                for (u32 k = 0; k < NftF_Count; ++k) grp->hold_dm[k] += dV * ((f64)ref[k] + (f64)grp->rows[j].f[k][l]);
            }
            f64 Vx = vol * 1e-3;                        // m^3
            f64 C = NFT_RHO_CP * Vx;
            f64 ev = live ? ev_coef : 0.0;
            hot->c[l] = c;
            hot->ko[l] = (f32)(kL * w * dx * dt / Vx);
            cold->vol_L[l] = vol;
            cold->inv_v[l] = (f32)(1.0 / vol);
            cold->kw[l] = (f32)(wrap->u_w_m2k * perim * dx * dt / C);
            cold->kr[l] = (f32)(wrap->absorptivity * w * dx * dt / C);
            cold->ev[l] = (f32)ev;
            cold->ge[l] = (f32)(lam * ev / C);
            cold->depth[l] = (f32)h;
            cold->phi[l] = (f32)phi;
            cold->vel[l] = (f32)(q / (w * hw));
            if (live) {
                if (h > 0.8 * wall) ++overflow;
                h_max = Maximum(h_max, h);
            }
        }
    }
    grp->overflow_cells = overflow;
    grp->max_depth_m = (f32)h_max;
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
            c->sla[l] = crop->sla_m2_g;
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

// Water (L) and mass of a field (mg, or L x C for T) held in the channel films.
internal f64 nft_rail_volume_L(NftSystem *s) {
    f64 v = 0.0;
    for (u32 g = 0; g < s->groups; ++g)
        for (u32 l = 0; l < NFT_LANES; ++l) {
            if (g * NFT_LANES + l >= s->rails) continue;
            for (u32 j = 0; j < s->cells; ++j) v += s->group[g].cold[j].vol_L[l];
        }
    return v;
}

internal f64 nft_rail_mass_mg(NftSystem *s, u32 field) {
    f64 m = 0.0;
    for (u32 g = 0; g < s->groups; ++g)
        for (u32 l = 0; l < NFT_LANES; ++l) {
            if (g * NFT_LANES + l >= s->rails) continue;
            for (u32 j = 0; j < s->cells; ++j)
                m += ((f64)s->ref[field] + (f64)s->group[g].rows[j].f[field][l]) * s->group[g].cold[j].vol_L[l];
        }
    return m;
}

// --- tank chemistry: alkalinity, pH, controller -------------------------------------

internal f64 nft_pco2_atm(NftParams *p) { return (f64)p->co2_ppm * 1e-6; }

// Untracked charge per litre: R is spread over the tank and the channels (it
// has no sources in the channels, which exchange their volume in minutes).
internal f64 nft_R_conc(NftSystem *s) {
    f64 v = s->res.V_L + s->acc.holdup_L;
    return v > 0.0 ? s->R_meq / v : 0.0;
}

// Charge of an ion vector (mg in `m`, NftField order), meq: + cations, - anions.
internal f64 nft_ion_charge_meq(const f64 *m) {
    f64 z = 0.0;
    for (u32 i = 0; i < NFT_IONS; ++i) z += nft_charge_sign[i] * m[NFT_ION0 + i] / nft_meq_mg[i];
    return z;
}

internal f64 nft_tank_ph(NftSystem *s) {
    f64 c[NftF_Count];
    for (u32 k = 0; k < NftF_Count; ++k) c[k] = nft_reservoir_conc(s, k);
    return nft_ph(c, s->res.T_C, nft_R_conc(s), nft_pco2_atm(&s->params), s->pH > 0.0 ? s->pH : 6.0);
}

// pH of a channel cell: its own ions and temperature, the system's R.
internal f64 nft_cell_ph(NftSystem *s, u32 g, u32 j, u32 l, f64 guess) {
    f64 c[NftF_Count];
    for (u32 k = 0; k < NftF_Count; ++k) c[k] = (f64)s->ref[k] + (f64)s->group[g].rows[j].f[k][l];
    return nft_ph(c, c[NftF_T], nft_R_conc(s), nft_pco2_atm(&s->params), guess);
}

// pH controller on the tank: 1 M HNO3 (adds NO3-N) or 1 M KOH (adds K), dosed
// toward the set point once pH leaves the band, rate-limited (unless `makeup`:
// the fill is adjusted at once, as a grower does when mixing a batch). Acid and
// base change pH through the strong ions they add; R is untouched.
internal void nft_ph_control(NftSystem *s, f64 dt, b32 makeup) {
    NftParams *p = &s->params;
    NftReservoir *r = &s->res;
    s->pH = nft_tank_ph(s);
    if (!p->ph_control) { s->acid_running = s->base_running = 0; return; }
    if (s->pH > p->ph_set + p->ph_band || makeup) s->acid_running = s->pH > p->ph_set;
    if (s->pH < p->ph_set - p->ph_band || makeup) s->base_running = s->pH < p->ph_set;
    if (s->acid_running && s->pH <= p->ph_set) s->acid_running = 0;
    if (s->base_running && s->pH >= p->ph_set) s->base_running = 0;
    if (!s->acid_running && !s->base_running) return;
    f64 c[NftF_Count];
    for (u32 k = 0; k < NftF_Count; ++k) c[k] = nft_reservoir_conc(s, k);
    NftWaterChem w = nft_water_chem(c, nft_R_conc(s));
    NftAcidBase ab = nft_acid_base(r->T_C, w.I);
    f64 ta_set = nft_alk_species(&ab, p->ph_set, nft_pco2_atm(p), w.PT, w.NT) * 1e3;   // meq/L
    f64 need = (w.TA_meq - ta_set) * r->V_L;           // meq of acid (> 0) or base (< 0)
    f64 cap = makeup ? 1e30 : (f64)p->acid_mmol_per_h_max / 3600.0 * dt;
    f64 mmol = Minimum(fabs(need), cap);
    if (need > 0.0 && s->acid_running) {
        f64 mg = mmol * nft_meq_mg[NftF_NO3 - NFT_ION0];
        r->mass_mg[NftF_NO3] += mg;
        s->acc.dosed_mg[NftF_NO3 - NFT_ION0] += mg;
        s->acc.acid_mmol += mmol;
        s->acc.acid_L += mmol * 1e-3;
        r->V_L += mmol * 1e-3;
    } else if (need < 0.0 && s->base_running) {
        f64 mg = mmol * nft_meq_mg[NftF_K - NFT_ION0];
        r->mass_mg[NftF_K] += mg;
        s->acc.dosed_mg[NftF_K - NFT_ION0] += mg;
        s->acc.base_mmol += mmol;
        s->acc.base_L += mmol * 1e-3;
        r->V_L += mmol * 1e-3;
    }
    s->pH = nft_tank_ph(s);
}

// Fresh solution: the recipe made up in tap water. The salts are neutral, so
// the batch carries the tap water's alkalinity; R is whatever balances that.
internal f64 nft_fill_R_conc(NftParams *p) {
    f64 m[NftF_Count] = {};
    for (u32 i = 0; i < NFT_IONS; ++i) m[NFT_ION0 + i] = p->recipe[i];
    return (f64)p->tap_alk_meq - nft_ion_charge_meq(m);
}

internal void nft_init(NftSystem *s, Arena *arena, NftParams params) {
    memset(s, 0, sizeof(*s));
    nft_layout(s, &params);
    nft_auto_size(s, &params);
    s->params = params;
    s->group = push_array_aligned(arena, NftGroup, s->groups, 64);
    for (u32 g = 0; g < s->groups; ++g) {
        s->group[g].rows = push_array_aligned(arena, NftRow, s->cells, 64);
        s->group[g].hot = push_array_aligned(arena, NftCellHot, s->cells, 64);
        s->group[g].cold = push_array_aligned(arena, NftCellCold, s->cells, 64);
        s->group[g].sites = push_array_aligned(arena, NftSiteRow, s->sites, 64);
    }
    s->res.V_L = params.reservoir_L;
    s->res.T_C = params.fill_T_C;
    nft_update_lane_params(s);
    nft_update_crop_lanes(s);

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
        // films through the initial root mats; the channels are filled on top of the tank volume
        nft_group_cells(s, g, s->ref, 0);
    }
    s->cells_force = 0;
    for (u32 g = 0; g < s->groups; ++g) {
        s->overflow_cells += s->group[g].overflow_cells;
        s->max_depth_m = Maximum(s->max_depth_m, s->group[g].max_depth_m);
    }
    s->acc.holdup_L = nft_rail_volume_L(s);
    s->R_meq = nft_fill_R_conc(&params) * (s->res.V_L + s->acc.holdup_L);
    nft_ph_control(s, 0.0, 1);   // make-up pH adjustment of the fresh batch
    f64 holdup = s->acc.holdup_L;
    memset(&s->acc, 0, sizeof(s->acc));   // the books start from the made-up batch
    s->acc.holdup_L = holdup;
    // the channels hold the same adjusted solution as the tank
    for (u32 k = 1; k < NftF_Count; ++k) s->ref[k] = (f32)nft_reservoir_conc(s, k);
    CopyArray(s->inlet, s->ref, NftF_Count);
}

// ---------------------------------------------------------------------------
// Kernels (lane templates: V = f32x8 for 8 rails, or f32 for one rail)

// Per-window inputs shared by every group.
struct NftEnv {
    f32 light_gain;          // growth multiplier (day / night, daily mean 1)
    f32 T_amb, radiant;      // C; W/m^2 on channel tops and canopy now
    f32 e0, es_slope;        // film evaporation: vapour deficit at the inlet temperature (Pa), d e_s / dT (Pa/K)
    f32 fpH;                 // growth factor of the tank pH
    f32 et_rad, et_vpd;      // transpiration, g/s per plant: et_rad (1 - exp(-k LAI)) + et_vpd LAI
    f32 inv_ground;          // 1 / ground area per plant (m^-2)
    f32 tissue_water;        // g water per g dry mass grown
};

#define NFT_K_EXT 0.7        // canopy light extinction coefficient (leafy crops)

// exp(-x) for x in [0, 16] without libm: Taylor on x/16 (error < 6e-7 for
// k LAI < 7), squared four times. Same operations in every lane type.
template <class V>
SIM_HD V nft_exp_neg(V x) {
    V u = lane_min(lane_max(x, lane_splat((V *)0, 0.0)), lane_splat((V *)0, 16.0)) * lane_splat((V *)0, 1.0 / 16.0);
    V e = lane_splat((V *)0, 1.0 / 720.0);
    e = lane_fnmadd(e, u, lane_splat((V *)0, 1.0 / 120.0));
    e = lane_fnmadd(e, u, lane_splat((V *)0, 1.0 / 24.0));
    e = lane_fnmadd(e, u, lane_splat((V *)0, 1.0 / 6.0));
    e = lane_fnmadd(e, u, lane_splat((V *)0, 0.5));
    e = lane_fnmadd(e, u, lane_splat((V *)0, 1.0));
    e = lane_fnmadd(e, u, lane_splat((V *)0, 1.0));      // 1 - u (1 - u (1/2 - u (1/6 - ...)))
    e = e * e; e = e * e; e = e * e; e = e * e;
    return e;
}

// Plants: limitation factors, growth, uptake/transpiration rates for the next
// window, harvest + replant. Reads the site cells' current state.
template <class V>
SIM_HD void nft_plants(NftGroup *grp, u32 lane, u32 sites, u32 site_offset, u32 site_stride, f32 dt_window,
                       f32 dt_sub, const NftEnv *env, const f32 *ref) {
    typedef typename LaneTraits<V>::Mask M;
    const V zero = lane_splat((V *)0, 0.0), one = lane_splat((V *)0, 1.0);
    const NftCropLanes *c = &grp->crop;
    V rgr = lane_load_f32(c->rgr_s + lane, (V *)0) * lane_splat((V *)0, env->light_gain * env->fpH);
    V wmax = lane_load_f32(c->wmax + lane, (V *)0), w0 = lane_load_f32(c->w0 + lane, (V *)0);
    V cycle = lane_load_f32(c->cycle_s + lane, (V *)0);
    V tmin = lane_load_f32(c->t_min + lane, (V *)0), tlo = lane_load_f32(c->t_lo + lane, (V *)0);
    V thi = lane_load_f32(c->t_hi + lane, (V *)0), tmax = lane_load_f32(c->t_max + lane, (V *)0);
    V root_frac = lane_load_f32(c->root_frac + lane, (V *)0);
    V lai_per_g = lane_load_f32(c->sla + lane, (V *)0) * (one - root_frac) * lane_splat((V *)0, env->inv_ground);
    V et_rad = lane_splat((V *)0, env->et_rad), et_vpd = lane_splat((V *)0, env->et_vpd);
    V kext = lane_splat((V *)0, NFT_K_EXT), tissue_water = lane_splat((V *)0, env->tissue_water);
    V o2s = lane_load_f32(c->o2_s + lane, (V *)0), docrit = lane_load_f32(c->do_crit + lane, (V *)0);
    V dtw = lane_splat((V *)0, dt_window), dts = lane_splat((V *)0, dt_sub);
    V tiny = lane_splat((V *)0, 1e-20);
    // heat toward bolting: air degree-days above the crop's threshold this window
    V tbolt = lane_load_f32(c->t_bolt + lane, (V *)0), bolt_dd = lane_load_f32(c->bolt_dd + lane, (V *)0);
    V heat = lane_max(lane_splat((V *)0, env->T_amb) - tbolt, zero) * lane_splat((V *)0, dt_window / 86400.0);

    for (u32 j = 0; j < sites; ++j) {
        NftSiteRow *site = grp->sites + j;
        u32 row = site_offset + j * site_stride;
        NftRow *cell = grp->rows + row;
        V inv_vcell = lane_load_f32(grp->cold[row].inv_v + lane, (V *)0);
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
        // water: canopy transpiration (Baille et al. 1994: radiation term on the
        // intercepted fraction, vapour-deficit term per unit leaf area) plus the
        // water held in the new tissue
        V lai = lai_per_g * W;
        V et = lane_fmadd(et_rad, one - nft_exp_neg(kext * lai), et_vpd * lai);
        V water = lane_fmadd(tissue_water, growth, et);                   // g/s

        for (u32 i = 0; i < NFT_IONS; ++i) {
            lane_store_f32(site->uptake[i] + lane, up[i]);
            lane_store_f32(site->drop[NFT_ION0 + i] + lane, up[i] * dts * inv_vcell);
        }
        for (u32 k = 0; k < NftF_Count; ++k) lane_store_f32(site->removed[k] + lane, zero);
        lane_store_f32(site->o2_rate + lane, o2);
        lane_store_f32(site->drop[NftF_DO] + lane, o2 * dts * inv_vcell);
        lane_store_f32(site->transp + lane, water);
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

// Transport: `substeps` upwind advection + wall/radiant/latent heat + O2
// exchange + site uptake, all rows of one group, in deltas from `ref` (the
// inlet, so the inflow delta is 0). Outlet deltas are summed for the return
// flow. Coefficients are per cell (root mats): the flux form
// x_j += (Q dt / V_j)(x_{j-1} - x_j) conserves mass with variable V_j.
template <class V>
SIM_HD void nft_transport(NftGroup *grp, u32 lane, u32 cells, u32 sites, u32 site_offset, u32 site_stride,
                          const f32 *ref, const NftEnv *env, u32 substeps) {
    const V zero = lane_splat((V *)0, 0.0);
    V refT = lane_splat((V *)0, ref[NftF_T]), refDO = lane_splat((V *)0, ref[NftF_DO]);
    V e0 = lane_splat((V *)0, env->e0), es_slope = lane_splat((V *)0, env->es_slope);
    // This window's heat terms, linear in the cell temperature delta x:
    //   wall   kw (T_amb - T)      radiant kr R
    //   latent -ge (e_s(T) - e_air) ~ -ge (e0 + s x)   (e_s linearized at the inlet temperature)
    // => x += -kT x + bT, kT = kw + ge s, bT = kw (T_amb - ref) + kr R - ge e0
    {
        V tamb_rel = lane_splat((V *)0, env->T_amb - ref[NftF_T]);
        V rad = lane_splat((V *)0, env->radiant);
        for (u32 j = 0; j < cells; ++j) {
            V kw = lane_load_f32(grp->cold[j].kw + lane, (V *)0);
            V ge = lane_load_f32(grp->cold[j].ge + lane, (V *)0);
            V kr = lane_load_f32(grp->cold[j].kr + lane, (V *)0);
            lane_store_f32(grp->hot[j].kT + lane, lane_fmadd(ge, es_slope, kw));
            lane_store_f32(grp->hot[j].bT + lane, lane_fnmadd(ge, e0, lane_fmadd(kw, tamb_rel, kr * rad)));
        }
    }
    V floor_[NftF_Count];                                       // delta at zero concentration
    for (u32 k = 0; k < NftF_Count; ++k) floor_[k] = lane_splat((V *)0, -ref[k]);
    V out[NftF_Count];
    for (u32 k = 0; k < NftF_Count; ++k) out[k] = zero;
    u32 last_site_row = site_offset + (sites ? (sites - 1) * site_stride : 0);

    for (u32 s = 0; s < substeps; ++s) {
        NftRow *last = grp->rows + cells - 1;
        for (u32 k = 0; k < NftF_Count; ++k) out[k] = out[k] + lane_load_f32(last->f[k] + lane, (V *)0);
        // Sweep outlet -> inlet so each row reads its upstream neighbour's old
        // value from memory (carrying it in registers instead spills: measured).
        // Sites are counted down with the sweep, no divide in the row loop.
        s32 next_site = sites ? (s32)sites - 1 : -1;
        u32 next_site_row = last_site_row;
        for (u32 j = cells; j-- > 0;) {
            NftRow *row = grp->rows + j;
            const NftCellHot *h = grp->hot + j;
            V c = lane_load_f32(h->c + lane, (V *)0);
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
            // heat: walls, radiant load, evaporative cooling of the open surface
            x[NftF_T] = lane_fnmadd(lane_load_f32(h->kT + lane, (V *)0), x[NftF_T], x[NftF_T]) + lane_load_f32(h->bT + lane, (V *)0);
            // O2 exchange with the air in the gully (saturation at the absolute temperature)
            x[NftF_DO] = lane_fmadd(lane_load_f32(h->ko + lane, (V *)0), nft_do_sat(refT + x[NftF_T]) - (refDO + x[NftF_DO]), x[NftF_DO]);
            // plant site: uptake and root respiration, booked as what was actually removed
            if (next_site >= 0 && j == next_site_row) {
                NftSiteRow *site = grp->sites + next_site;
                --next_site;
                next_site_row -= site_stride;
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
    // water evaporated from the film this window (end-of-window temperatures;
    // the water books close on this number, the latent heat was taken per substep)
    V evap = zero;
    for (u32 j = 0; j < cells; ++j)
        evap = lane_fmadd(lane_load_f32(grp->cold[j].ev + lane, (V *)0),
                          lane_fmadd(es_slope, lane_load_f32(grp->rows[j].f[NftF_T] + lane, (V *)0), e0), evap);
    lane_store_f32(grp->lp.evap_kg + lane, evap * lane_splat((V *)0, (f64)substeps));
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

// Re-solve a group's root-mat cells this window? Every NFT_CELL_WINDOWS windows
// per group (30 min: roots change over hours), staggered so the scalar solve
// is spread evenly over windows; always after a hydraulic change.
#define NFT_CELL_WINDOWS 30
internal b32 nft_cells_due(NftSystem *s, u32 g) { return s->cells_force || (s->window_index + g) % NFT_CELL_WINDOWS == 0; }

// f32 deltas near the inlet decay geometrically toward the inflow's 0 after
// every rebase; in root cells (Courant ~0.1-0.3) they spend ~100 substeps in
// the subnormal range, where each operation costs ~100+ cycles (measured: the
// whole transport 1.5x slower). Values below 1.2e-38 mg/L carry no physics, so
// the group's work runs with flush-to-zero / denormals-are-zero, and restores
// the caller's MXCSR (the f64 kinetics on the same workers keep IEEE).
struct NftFlushDenormals {
#if defined(__x86_64__) && !defined(__CUDA_ARCH__)
    u32 saved;
    NftFlushDenormals() { saved = _mm_getcsr(); _mm_setcsr(saved | 0x8040); }
    ~NftFlushDenormals() { _mm_setcsr(saved); }
#endif
};

// One whole window for one group: rebase, root-mat cells (when due), plants,
// then transport. The unit of work handed to a thread (temporal blocking: the
// group stays in that core's L1/L2). `shift` = old reference - new reference.
template <class V>
SIM_HD void nft_group_window(NftSystem *s, u32 g, const f32 *ref, const f32 *shift, b32 rebase, const NftEnv *env) {
    NftGroup *grp = s->group + g;
    NftFlushDenormals ftz;
    if (rebase)
        for (u32 lane = 0; lane < NFT_LANES; lane += LaneTraits<V>::width) nft_rebase<V>(grp, lane, s->cells, shift);
    if (nft_cells_due(s, g)) nft_group_cells(s, g, ref, 1);
    for (u32 lane = 0; lane < NFT_LANES; lane += LaneTraits<V>::width) {
        nft_plants<V>(grp, lane, s->sites, s->site_offset, s->site_stride, (f32)s->window_s, (f32)s->dt_sub, env, ref);
        nft_transport<V>(grp, lane, s->cells, s->sites, s->site_offset, s->site_stride, ref, env, s->substeps);
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
    s->rh = Clamp(p->rh, 0.0f, 1.0f);
}

// Growth multiplier while lights are on / off so the daily mean is 1:
// photosynthesis by day, a little (reserves) by night.
internal f32 nft_light_gain(NftSystem *s) {
    f32 day = s->params.photoperiod_h / 24.0f, dark_gain = 0.1f;
    f32 day_gain = day > 0.0f ? (1.0f - (1.0f - day) * dark_gain) / day : 0.0f;
    return s->light > 0.5f ? day_gain : dark_gain;
}

// Growth response to the root-zone pH: 1 over 5.0-7.0 (the usual hydroponic
// range), falling linearly to 0 at 3.5 (root damage) and 9.5 (P, Fe, Mn
// unavailable). An empirical envelope, not a speciation model of micronutrients.
internal f32 nft_ph_factor(f64 pH) {
    f64 f = pH < 5.0 ? (pH - 3.5) / 1.5 : pH > 7.0 ? (9.5 - pH) / 2.5 : 1.0;
    return (f32)Clamp(f, 0.0, 1.0);
}

internal f64 nft_ground_per_plant_m2(NftSystem *s) {
    return Maximum((f64)s->params.channel_pitch_m, 0.01) * Clamp((f64)s->params.rail_length_m, 0.2, 200.0) / (f64)s->sites;
}

internal NftEnv nft_env(NftSystem *s) {
    NftParams *p = &s->params;
    NftEnv e;
    e.light_gain = nft_light_gain(s);
    e.T_amb = s->T_amb;
    e.radiant = s->light * p->radiant_W_m2;
    f64 Tw = s->ref[NftF_T], Ta = s->T_amb;
    f64 e_air = s->rh * nft_es_Pa(Ta);
    e.e0 = (f32)(nft_es_Pa(Tw) - e_air);
    e.es_slope = (f32)nft_es_slope(Tw);
    e.fpH = nft_ph_factor(s->pH);
    // Baille: lambda E = A (1 - exp(-k LAI)) Rn + B LAI VPD per ground area;
    // stomata mostly close in the dark (B x night_stomata, no radiation term)
    f64 ground = nft_ground_per_plant_m2(s);
    f64 vpd_kPa = Maximum(nft_es_Pa(Ta) - e_air, 0.0) * 1e-3;
    f64 g_per_J = 1000.0 / nft_latent(Ta);
    f64 B = (f64)p->baille_B * (s->light > 0.5f ? 1.0 : (f64)p->night_stomata);
    e.et_rad = (f32)((f64)p->baille_A * e.radiant * ground * g_per_J);
    e.et_vpd = (f32)(B * vpd_kPa * ground * g_per_J);
    e.inv_ground = (f32)(1.0 / ground);
    e.tissue_water = p->tissue_water;
    return e;
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

// Tank surface open to the room: a ~cube tank's top x the uncovered fraction.
internal f64 nft_tank_open_m2(NftParams *p) {
    return pow(Maximum((f64)p->reservoir_L, 1.0) / 1000.0, 2.0 / 3.0) * Clamp((f64)p->tank_open_frac, 0.0, 1.0);
}

// Mixes the window's return flow into the tank and applies its heat budget,
// chiller, aeration, root-mat holdup, evaporation, plant water, top-off,
// dosing and pH control. f64 throughout.
internal void nft_reservoir_window(NftSystem *s) {
    NftParams *p = &s->params;
    NftReservoir *r = &s->res;
    f64 dt = s->window_s;
    f64 inv_sub = 1.0 / (f64)s->substeps;

    // water drawn into (or released by) the root mats this window, with what it carried
    f64 hold_dV = 0.0, hold_dm[NftF_Count] = {};
    u32 overflow = 0;
    f32 hmax = 0.0f;
    for (u32 g = 0; g < s->groups; ++g) {
        NftGroup *grp = s->group + g;
        hold_dV += grp->hold_dV;
        for (u32 k = 0; k < NftF_Count; ++k) hold_dm[k] += grp->hold_dm[k];
        grp->hold_dV = 0.0;
        for (u32 k = 0; k < NftF_Count; ++k) grp->hold_dm[k] = 0.0;
        overflow += grp->overflow_cells;
        hmax = Maximum(hmax, grp->max_depth_m);
    }
    s->overflow_cells = overflow;
    s->max_depth_m = hmax;
    if (hold_dV != 0.0) {
        f64 v_new = r->V_L - hold_dV;
        r->T_C = (r->T_C * r->V_L - hold_dm[NftF_T]) / v_new;   // heat content: L x C
        for (u32 k = 1; k < NftF_Count; ++k) r->mass_mg[k] -= hold_dm[k];
        r->V_L = v_new;
        s->acc.holdup_L += hold_dV;
    }

    // return flow (rails carried tank water out at the window's start conditions)
    f64 q_tot = 0.0, ret[NftF_Count] = {}, uptake_window[NFT_IONS] = {}, water_g_s = 0.0, evap_kg = 0.0;
    for (u32 g = 0; g < s->groups; ++g) {
        NftGroup *grp = s->group + g;
        for (u32 l = 0; l < NFT_LANES; ++l) {
            f64 q = grp->lp.flow_L_s[l];
            if (q <= 0.0) continue;
            q_tot += q;
            // what left the rails: inlet + mean deviation (L/s x mg/L, or x C)
            for (u32 k = 0; k < NftF_Count; ++k) ret[k] += q * ((f64)s->inlet[k] + (f64)grp->out_sum[k][l] * inv_sub);
            evap_kg += (f64)grp->lp.evap_kg[l];
            for (u32 j = 0; j < s->sites; ++j) {
                f64 vcell = grp->cold[s->site_offset + j * s->site_stride].vol_L[l];
                for (u32 i = 0; i < NFT_IONS; ++i) uptake_window[i] += (f64)grp->sites[j].removed[NFT_ION0 + i][l] * vcell;
                water_g_s += (f64)grp->sites[j].transp[l];
            }
        }
    }
    // the tank is debited exactly what the rails received (the f32 inlet), so
    // the f32/f64 boundary cannot create or destroy mass
    for (u32 k = 1; k < NftF_Count; ++k) r->mass_mg[k] += dt * (ret[k] - q_tot * (f64)s->inlet[k]);
    for (u32 i = 0; i < NFT_IONS; ++i) s->acc.uptake_mg[i] += uptake_window[i];

    // heat: return water (already cooled by film evaporation), tank walls, pump,
    // tank-surface evaporation, chiller (hysteresis on the tank temperature)
    f64 e_air = s->rh * nft_es_Pa(s->T_amb);
    f64 evap_tank_kg_s = nft_evap_coef(p->h_conv_W_m2K) * nft_tank_open_m2(p) * (nft_es_Pa(r->T_C) - e_air);
    f64 heat_W = NFT_RHO_CP * 1e-3 * (ret[NftF_T] - q_tot * (f64)s->inlet[NftF_T])   // rho cp (L/s -> m^3/s) dT
                 + p->tank_ua_W_K * ((f64)s->T_amb - r->T_C) + nft_pump_heat_W(p, &s->hyd)
                 - nft_latent(r->T_C) * evap_tank_kg_s;
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
    s->acc.evap_heat_J += (nft_latent(r->T_C) * evap_tank_kg_s * dt + nft_latent(s->ref[NftF_T]) * evap_kg);
    r->T_C += heat_W * dt / (NFT_RHO_CP * 1e-3 * r->V_L);

    // tank aeration toward saturation
    f64 do_c = nft_reservoir_conc(s, NftF_DO);
    r->mass_mg[NftF_DO] += dt * p->tank_kla_per_s * (nft_do_sat_f64(r->T_C) - do_c) * r->V_L;

    // water leaves through the plants and the open surfaces (1 kg = 1 L); the
    // ions stay, so the solution concentrates until top-off
    f64 water_L = water_g_s * dt * 1e-3;
    f64 evap_tank_L = evap_tank_kg_s * dt;
    r->V_L -= water_L + evap_kg + evap_tank_L;
    s->acc.transpired_L += water_L;
    s->acc.evap_channel_L += evap_kg;
    s->acc.evap_tank_L += evap_tank_L;
    s->transp_L_s = water_L / dt;
    s->evap_channel_L_s = evap_kg / dt;
    s->evap_tank_L_s = evap_tank_L / dt;

    // top-off with tap water (no tracked ions; its alkalinity goes into R) back to the set volume
    if (p->topoff_on && r->V_L < 0.95 * p->reservoir_L) {
        f64 add = p->reservoir_L - r->V_L;
        r->T_C = (r->T_C * r->V_L + p->fill_T_C * add) / p->reservoir_L;
        r->mass_mg[NftF_DO] += nft_do_sat_f64(p->fill_T_C) * add;
        r->V_L = p->reservoir_L;
        s->acc.topoff_L += add;
        s->R_meq += (f64)p->tap_alk_meq * add;
    }
    // EC dosing, rate-limited. Stock ion ratios: either the fill recipe, or
    // matched to what the plants actually take up (their tissue composition),
    // so ions neither pile up (Ca, S) nor run out (K) in a recirculating system.
    // The stock is made of neutral salts: its counter-ions enter R, so dosing
    // itself does not move the alkalinity.
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
            f64 m[NftF_Count] = {};
            for (u32 i = 0; i < NFT_IONS; ++i) {
                m[NFT_ION0 + i] = stock[NFT_ION0 + i] * dose;
                r->mass_mg[NFT_ION0 + i] += m[NFT_ION0 + i];
                s->acc.dosed_mg[i] += m[NFT_ION0 + i];
            }
            s->R_meq -= nft_ion_charge_meq(m);
            r->V_L += dose;
            s->acc.dosed_L += dose;
        }
    }
    // scheduled solution change: drain the tank, refill with fresh recipe
    b32 refill = 0;
    if (p->dump_days > 0.0f && s->t + dt - s->acc.last_dump_t >= (f64)p->dump_days * 86400.0) {
        f64 R_channels = nft_R_conc(s) * s->acc.holdup_L;
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
        s->R_meq = R_channels + nft_fill_R_conc(p) * p->reservoir_L;
        s->acc.last_dump_t = s->t + dt;
        refill = 1;
    }
    nft_ph_control(s, dt, refill);
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
    const NftEnv *env;
};

internal PLATFORM_WORK_QUEUE_CALLBACK(nft_work_window) {
    NftWork *w = (NftWork *)data;
    for (u32 g = w->first; g < w->end; ++g) nft_group_window<f32x8>(w->s, g, w->s->ref, w->shift, w->rebase, w->env);
}

// Water viscosity follows the tank temperature (~1.2 %/K, film depth ~0.4 %/K):
// re-solve the network when the tank has moved 1 K since the last solve.
internal void nft_maybe_resolve(NftSystem *s) {
    if (fabs(s->res.T_C - s->hyd_solved_T) > 1.0) nft_update_lane_params(s);
}

// Window prologue / epilogue around the group work (shared with the tests).
internal NftEnv nft_window_begin(NftSystem *s, f32 *shift, b32 *rebase) {
    nft_maybe_resolve(s);
    nft_environment(s, s->t);
    *rebase = nft_inlet(s, shift);
    return nft_env(s);
}

internal void nft_window_end(NftSystem *s) {
    s->cells_force = 0;
    nft_reservoir_window(s);
    s->t += s->window_s;
    ++s->window_index;
}

internal void nft_window(NftSystem *s, PlatformAPI *api, PlatformWorkQueue *queue, u32 threads, Arena *frame) {
    TIMED_FUNCTION();
    f32 shift[NftF_Count];
    b32 rebase;
    NftEnv env = nft_window_begin(s, shift, &rebase);
    // Small systems: the whole window is a fraction of a millisecond; dispatching
    // it costs more than it saves. ~1.5 ns per row-substep per group measured.
    f64 work_rows = (f64)s->groups * s->cells * s->substeps;
    if (!queue || threads <= 1 || s->groups <= 1 || work_rows < 2.0e5) {
        for (u32 g = 0; g < s->groups; ++g) nft_group_window<f32x8>(s, g, s->ref, shift, rebase, &env);
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
            w->env = &env;
            api->add_work_entry(queue, nft_work_window, w);
        }
        api->complete_all_work(queue);
        end_temp(temp);
    }
    nft_window_end(s);
}

// --- diagnostics ------------------------------------------------------------------

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
