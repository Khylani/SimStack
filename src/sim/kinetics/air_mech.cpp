// Humid-air neutral chemistry for a DBD reactor and its afterglow, plus the
// surrogate electron-impact source (G-values) that stands in for the
// Boltzmann/plasma part until the M2 reference solver (ChemPlasKin) and the
// generated reduction tables exist.
//
// 17 species, 44 reactions. Rate data: JPL 19-5 / IUPAC evaluations at the
// low-pressure-limit / fall-off forms given there, rounded to two digits.
//
// Three metastables carry the chemistry that ground-state radicals alone
// cannot: N(2D) + O2 -> NO + O is the main NO source of an air discharge at
// room temperature (N(4S) + O2 is 1e5 times slower), N2(A) dissociates O2 and
// makes NO with O atoms, and O(1D) makes OH from water and NO from N2O. They
// live 2 ns (O(1D)) to 50 ns (N(2D), N2(A)) at 1 atm, thousands of times
// shorter than any other time scale here, so they are treated quasi-steady:
// at each discharge pulse each one is distributed over its reaction partners
// in proportion to k [partner] (air_source_delta). Same rate constants (Herron,
// J. Phys. Chem. Ref. Data 28, 1999, for N(2D); Kossyi et al., Plasma Sources
// Sci. Technol. 1, 1992, for N2(A); JPL 19-5 for O(1D)), exact atom balance,
// and no nanosecond stiffness for the solver.
// Ions, N2(v) and surface chemistry are not modelled; vibrational NO
// production (N2(v) + O -> NO + N) matters above ~400 K gas temperature.

struct AirSpecies {
    // Indices into KinModel::mech species order, resolved by name at build.
    u32 O, O3, N, NO, NO2, NO3, N2O5, N2O, HNO2, HNO3, OH, HO2, H2O2, H, N2, O2, H2O;
};

internal void air_mech_build(KinModel *model, AirSpecies *sp) {
    memset(model, 0, sizeof(*model));
    sp->O    = kin_add_species(model, "O");
    sp->O3   = kin_add_species(model, "O3");
    sp->N    = kin_add_species(model, "N");
    sp->NO   = kin_add_species(model, "NO");
    sp->NO2  = kin_add_species(model, "NO2");
    sp->NO3  = kin_add_species(model, "NO3");
    sp->N2O5 = kin_add_species(model, "N2O5");
    sp->N2O  = kin_add_species(model, "N2O");
    sp->HNO2 = kin_add_species(model, "HNO2");
    sp->HNO3 = kin_add_species(model, "HNO3");
    sp->OH   = kin_add_species(model, "OH");
    sp->HO2  = kin_add_species(model, "HO2");
    sp->H2O2 = kin_add_species(model, "H2O2");
    sp->H    = kin_add_species(model, "H");
    sp->N2   = kin_add_species(model, "N2");
    sp->O2   = kin_add_species(model, "O2");
    sp->H2O  = kin_add_species(model, "H2O");

    // Ox
    kin_add_reaction(model, "O + O2 + M -> O3 + M",        kin_m(6.0e-34, -2.4));
    kin_add_reaction(model, "O + O3 -> O2 + O2",           kin_arr(8.0e-12, 0.0, 2060.0));
    kin_add_reaction(model, "O + O + M -> O2 + M",         kin_m(5.2e-35, 0.0, -900.0));
    // NOx
    kin_add_reaction(model, "O + NO + M -> NO2 + M",       kin_troe(9.0e-32, -1.5, 3.0e-11, 0.0));
    kin_add_reaction(model, "O + NO2 -> NO + O2",          kin_arr(5.1e-12, 0.0, -210.0));
    kin_add_reaction(model, "O + NO2 + M -> NO3 + M",      kin_troe(2.5e-31, -1.8, 2.2e-11, -0.7));
    kin_add_reaction(model, "O3 + NO -> NO2 + O2",         kin_arr(3.0e-12, 0.0, 1500.0));
    kin_add_reaction(model, "O3 + NO2 -> NO3 + O2",        kin_arr(1.2e-13, 0.0, 2450.0));
    kin_add_reaction(model, "NO3 + NO -> NO2 + NO2",       kin_arr(1.5e-11, 0.0, -170.0));
    u32 r_n2o5 =
    kin_add_reaction(model, "NO2 + NO3 + M -> N2O5 + M",   kin_troe(2.0e-30, -4.4, 1.4e-12, -0.7));
    kin_add_reaction(model, "N2O5 + M -> NO2 + NO3 + M",   kin_eq_reverse(r_n2o5, 2.7e-27, 11000.0));
    kin_add_reaction(model, "NO + NO + O2 -> NO2 + NO2",   kin_arr(3.3e-39, 0.0, -530.0));
    kin_add_reaction(model, "N2O5 + H2O -> HNO3 + HNO3",   kin_arr(2.5e-22));
    // N atoms
    kin_add_reaction(model, "N + NO -> N2 + O",            kin_arr(2.1e-11, 0.0, -100.0));
    kin_add_reaction(model, "N + O2 -> NO + O",            kin_arr(1.5e-11, 0.0, 3600.0));
    // N + NO2: JPL total, Kossyi 1992 branching (N2O + O 0.43, 2 NO 0.33, N2 + 2 O 0.13, N2 + O2 0.10)
    kin_add_reaction(model, "N + NO2 -> N2O + O",          kin_arr(0.434 * 5.8e-12, 0.0, -220.0));
    kin_add_reaction(model, "N + NO2 -> NO + NO",          kin_arr(0.333 * 5.8e-12, 0.0, -220.0));
    kin_add_reaction(model, "N + NO2 -> N2 + O + O",       kin_arr(0.132 * 5.8e-12, 0.0, -220.0));
    kin_add_reaction(model, "N + NO2 -> N2 + O2",          kin_arr(0.101 * 5.8e-12, 0.0, -220.0));
    kin_add_reaction(model, "N + O + M -> NO + M",         kin_m(5.5e-33, 0.0, -155.0));
    kin_add_reaction(model, "N + N + M -> N2 + M",         kin_m(8.3e-34, 0.0, -500.0));
    kin_add_reaction(model, "N + OH -> NO + H",            kin_arr(4.7e-11));
    // HOx
    kin_add_reaction(model, "OH + OH + M -> H2O2 + M",     kin_troe(6.9e-31, -1.0, 2.6e-11, 0.0));
    kin_add_reaction(model, "OH + OH -> H2O + O",          kin_arr(1.8e-12));
    kin_add_reaction(model, "OH + O -> H + O2",            kin_arr(1.8e-11, 0.0, -180.0));
    kin_add_reaction(model, "OH + O3 -> HO2 + O2",         kin_arr(1.7e-12, 0.0, 940.0));
    kin_add_reaction(model, "OH + HO2 -> H2O + O2",        kin_arr(4.8e-11, 0.0, -250.0));
    kin_add_reaction(model, "OH + H2O2 -> HO2 + H2O",      kin_arr(1.8e-12));
    kin_add_reaction(model, "HO2 + HO2 -> H2O2 + O2",      kin_arr(3.0e-13, 0.0, -460.0));
    kin_add_reaction(model, "HO2 + O -> OH + O2",          kin_arr(3.0e-11, 0.0, -200.0));
    kin_add_reaction(model, "HO2 + O3 -> OH + O2 + O2",    kin_arr(1.0e-14, 0.0, 490.0));
    kin_add_reaction(model, "H + O2 + M -> HO2 + M",       kin_troe(4.4e-32, -1.3, 7.5e-11, 0.2));
    kin_add_reaction(model, "H + O3 -> OH + O2",           kin_arr(1.4e-10, 0.0, 470.0));
    kin_add_reaction(model, "H + HO2 -> OH + OH",          kin_arr(7.2e-11));
    kin_add_reaction(model, "O + H2O2 -> OH + HO2",        kin_arr(1.4e-12, 0.0, 2000.0));
    // HOx-NOx coupling, nitrous and nitric acid
    kin_add_reaction(model, "OH + NO + M -> HNO2 + M",     kin_troe(7.0e-31, -2.6, 3.6e-11, -0.1));
    kin_add_reaction(model, "OH + NO2 + M -> HNO3 + M",    kin_troe(1.8e-30, -3.0, 2.8e-11, 0.0));
    kin_add_reaction(model, "OH + HNO2 -> NO2 + H2O",      kin_arr(1.8e-11, 0.0, 390.0));
    kin_add_reaction(model, "OH + HNO3 -> NO3 + H2O",      kin_arr(1.5e-13));
    kin_add_reaction(model, "HO2 + NO -> OH + NO2",        kin_arr(3.3e-12, 0.0, -270.0));
    kin_add_reaction(model, "H + NO2 -> OH + NO",          kin_arr(4.0e-10, 0.0, 340.0));
    kin_add_reaction(model, "OH + NO3 -> HO2 + NO2",       kin_arr(2.2e-11));
    kin_add_reaction(model, "HO2 + NO3 -> OH + NO2 + O2",  kin_arr(3.5e-12));
    // thermal ozone decomposition (matters above ~400 K)
    kin_add_reaction(model, "O3 + M -> O2 + O + M",        kin_m(7.2e-10, 0.0, 11400.0));

    kin_mech_finalize(model);
}

// ----------------------------------------------------------------------------
// Discharge source. Each filamentary micro-discharge deposits energy that
// electron impact turns into radicals and metastables. At the reduced fields
// of air DBDs (100-200 Td) the yields per 100 eV are roughly (Penetrante et
// al. 1997, Kossyi et al. 1992, Sakiyama et al. 2012; BOLSIG+ energy
// partition in air):
//
//   e + O2 -> O + O / O + O(1D)   ~1.5 dissociations (6 and 8.4 eV channels; the
//                                 8.4 eV one gives O(1D), ~half at these fields)
//   e + N2 -> N2(A) (+ B, C cascade)  ~1.5 (the triplets take ~20 % of the energy)
//   e + N2 -> N + N(2D)           ~0.4 dissociations (predissociation gives
//                                 mostly N(4S) + N(2D))
//   e + H2O -> OH + H             ~0.3 at 2 % H2O (direct; O(1D) + H2O and
//                                 N2(A) + H2O are now explicit)
//
// N2(A) quenched by O2 adds ~2 O per 100 eV, so the total odd oxygen stays
// ~5 per 100 eV (ozonizer yields). Each channel scales linearly with its
// parent's mole fraction relative to the reference gas, and consumes the
// parent, so N, O and H atoms are conserved exactly. These are the tunables
// a Boltzmann (LoKI-B) solution of the reduced field will replace.

#include "air_source_table.h"   // LoKI-B++ G-values vs E/N (tools/ref/loki_gvalues.py)

struct AirSourceModel {
    f64 g_O2_diss;        // O2 dissociations per 100 eV at x_O2 = 0.21
    f64 f_O1D;            // share of them in the O + O(1D) channel
    f64 g_N2A;            // N2(A) per 100 eV at x_N2 = 0.78 (incl. cascade)
    f64 g_N2_diss;        // N2 dissociations per 100 eV at x_N2 = 0.78 (-> N + N(2D))
    f64 g_OH_per_100eV;   // H2O dissociations at x_H2O = 0.02
    // Boltzmann source: when on, the first four come from the LoKI-B++ table at
    // the micro-discharges' reduced field instead of the hand-set values above
    b32 boltzmann;
    f64 reduced_field_Td;
    // Micro-discharge (filament) energy density: each gas parcel is hit by a
    // filament depositing filament_J_cm3, at rate power / filament_J_cm3.
    // 0 = the averaged picture: one pulse per half-cycle with power / (2 f).
    f64 filament_J_cm3;
};

// Linear interpolation in the LoKI-B++ table (clamped at its ends).
internal void air_source_table_lookup(f64 en_Td, f64 *g_o2, f64 *f_o1d, f64 *g_n2a, f64 *g_n2d, f64 *eta_vib) {
    u32 n = AIR_SOURCE_TABLE_ROWS;
    f64 x = Clamp(en_Td, air_source_table[0][0], air_source_table[n - 1][0]);
    u32 i = 0;
    while (i + 2 < n && air_source_table[i + 1][0] < x) ++i;
    const f64 *a = air_source_table[i], *b = air_source_table[i + 1];
    f64 t = (x - a[0]) / (b[0] - a[0]);
    *g_o2 = a[1] + t * (b[1] - a[1]);
    *f_o1d = a[2] + t * (b[2] - a[2]);
    *g_n2a = a[3] + t * (b[3] - a[3]);
    *g_n2d = a[4] + t * (b[4] - a[4]);
    if (eta_vib) *eta_vib = a[5] + t * (b[5] - a[5]);
}

internal AirSourceModel air_source_defaults() {
    AirSourceModel s;
    s.g_O2_diss = 1.5;
    s.f_O1D = 0.5;
    s.g_N2A = 1.5;
    s.g_N2_diss = 0.4;
    s.g_OH_per_100eV = 0.3;
    s.boltzmann = 0;
    s.reduced_field_Td = 180.0;   // air DBD micro-discharges: 150-250 Td
    s.filament_J_cm3 = 0.0;
    return s;
}

#define EV_PER_JOULE 6.241509074e18

// Quasi-steady distribution of `amount` of a metastable over its channels:
// share_c = k_c [partner_c] / sum. A channel may not consume more than half
// of its partner (only possible for a trace partner at a huge pulse); the
// excess goes to the dominant channel. Returns the shares in `share`.
struct AirChannel { f64 k; u32 partner; };
internal void air_branch(const AirChannel *ch, u32 count, const f64 *y, f64 amount, f64 *share) {
    f64 sum = 0.0;
    u32 big = 0;
    for (u32 c = 0; c < count; ++c) {
        share[c] = ch[c].k * y[ch[c].partner];
        if (share[c] > share[big]) big = c;
        sum += share[c];
    }
    if (sum <= 0.0) { for (u32 c = 0; c < count; ++c) share[c] = c == big ? 1.0 : 0.0; return; }
    f64 moved = 0.0;
    for (u32 c = 0; c < count; ++c) {
        share[c] /= sum;
        f64 cap = amount > 0.0 ? 0.5 * y[ch[c].partner] / amount : 1.0;
        if (c != big && share[c] > cap) { moved += share[c] - cap; share[c] = cap; }
    }
    share[big] += moved;
}

// Species change per cm^3 for `energy_J_cm3` deposited into gas of density y
// at temperature T: electron-impact channels, then the metastables' quasi-
// steady fates (N2(A) first: it makes N(2D); then N(2D); then O(1D)).
internal void air_source_delta(AirSpecies *sp, AirSourceModel *src, const f64 *y, u32 n, f64 energy_J_cm3, f64 *delta,
                               f64 T_K = 300.0) {
    for (u32 s = 0; s < n; ++s) delta[s] = 0.0;
    f64 total = 0.0;
    for (u32 s = 0; s < n; ++s) total += y[s];
    if (total <= 0.0 || energy_J_cm3 <= 0.0) return;
    f64 hundreds_eV = energy_J_cm3 * EV_PER_JOULE * 0.01;
    f64 g_o2 = src->g_O2_diss, f_o1d = src->f_O1D, g_n2a = src->g_N2A, g_n2d = src->g_N2_diss;
    if (src->boltzmann) air_source_table_lookup(src->reduced_field_Td, &g_o2, &f_o1d, &g_n2a, &g_n2d, 0);
    // (the table is for dry air 79/21: scale by the parents' mole fractions as below)
    f64 dO2 = g_o2  * (y[sp->O2]  / total / 0.21) * hundreds_eV;
    f64 dA  = g_n2a * (y[sp->N2]  / total / 0.78) * hundreds_eV;
    f64 dN2 = g_n2d * (y[sp->N2]  / total / 0.78) * hundreds_eV;
    f64 dOH = src->g_OH_per_100eV * (y[sp->H2O] / total / 0.02) * hundreds_eV;
    f64 d1D = f_o1d * dO2;
    // ground-state products and the parents consumed
    delta[sp->O2] -= dO2;   delta[sp->O] += 2.0 * dO2 - d1D;
    delta[sp->N2] -= dA + dN2;
    delta[sp->N]  += dN2;
    delta[sp->OH] += dOH;   delta[sp->H] += dOH;   delta[sp->H2O] -= dOH;
    f64 t3 = T_K / 300.0;
    f64 k_o2 = pow(t3, 0.55);
    f64 f[10];
    // N2(A 3Sigma_u+) (Kossyi 1992): back to N2 in every channel
    const AirChannel a_ch[] = {
        { 2.5e-12 * k_o2, sp->O2 },   // -> N2 + O + O
        { 1.3e-12 * k_o2, sp->O2 },   // -> N2 + O2 (quench)
        { 7.8e-14 * k_o2, sp->O2 },   // -> N2O + O
        { 7.0e-12, sp->O },           // -> NO + N(2D)
        { 7.0e-11, sp->NO },          // -> N2 + NO (quench)
        { 1.0e-12, sp->NO2 },         // -> N2 + NO + O
        { 8.0e-11, sp->N2O },         // -> N2 + N + NO (Kossyi 1992)
        { 5.0e-14, sp->H2O },         // -> N2 + H + OH
        { 3.0e-16, sp->N2 },          // -> N2 + N2 (quench)
    };
    air_branch(a_ch, ArrayCount(a_ch), y, dA, f);
    delta[sp->N2] += dA * (f[0] + f[1] + f[4] + f[5] + f[6] + f[7] + f[8]);
    delta[sp->O2] -= dA * (f[0] + f[2]);   delta[sp->O] += dA * (2.0 * f[0] + f[2] + f[5]);
    delta[sp->N2O] += dA * (f[2] - f[6]);
    delta[sp->O] -= dA * f[3];   delta[sp->NO] += dA * (f[3] + f[5] + f[6]);   f64 n2d = dN2 + dA * f[3];
    delta[sp->NO2] -= dA * f[5];
    delta[sp->N] += dA * f[6];    // N2(A) + N2O -> N2 + N + NO
    delta[sp->H2O] -= dA * f[7];   delta[sp->H] += dA * f[7];   delta[sp->OH] += dA * f[7];
    // N(2D) (Herron 1999)
    const AirChannel n_ch[] = {
        { 5.2e-12, sp->O2 },          // -> NO + O
        { 1.7e-14, sp->N2 },          // -> N + N2 (quench)
        { 6.9e-13, sp->O },           // -> N + O (quench)
        { 6.0e-11, sp->NO },          // -> N2 + O
        { 2.2e-12, sp->N2O },         // -> NO + N2
    };
    air_branch(n_ch, ArrayCount(n_ch), y, n2d, f);
    delta[sp->O2] -= n2d * f[0];   delta[sp->NO] += n2d * (f[0] - f[3] + f[4]);   delta[sp->O] += n2d * (f[0] + f[3]);
    delta[sp->N] += n2d * (f[1] + f[2]);
    delta[sp->N2] += n2d * (f[3] + f[4]);   delta[sp->N2O] -= n2d * f[4];
    // O(1D) (JPL 19-5)
    const AirChannel o_ch[] = {
        { 2.15e-11 * exp(110.0 / T_K), sp->N2 },    // -> O + N2
        { 3.3e-11 * exp(55.0 / T_K), sp->O2 },      // -> O + O2
        { 1.63e-10 * exp(60.0 / T_K), sp->H2O },    // -> OH + OH
        { 7.26e-11 * exp(20.0 / T_K), sp->N2O },    // -> NO + NO
        { 4.63e-11 * exp(20.0 / T_K), sp->N2O },    // -> N2 + O2
        { 2.4e-10, sp->O3 },                        // -> O2 + O2 (half) / O2 + O + O (half)
    };
    air_branch(o_ch, ArrayCount(o_ch), y, d1D, f);
    delta[sp->O] += d1D * (f[0] + f[1]);          // quenched to O(3P)
    // O(1D) + O3 -> O2 + O2 or O2 + O + O, equal rates: per O(1D), -O3, +1.5 O2, +1 O
    delta[sp->O3] -= d1D * f[5];   delta[sp->O2] += d1D * f[5] * 1.5;   delta[sp->O] += d1D * f[5];
    delta[sp->H2O] -= d1D * f[2];  delta[sp->OH] += d1D * f[2] * 2.0;
    delta[sp->N2O] -= d1D * (f[3] + f[4]);   delta[sp->NO] += d1D * f[3] * 2.0;   delta[sp->N2] += d1D * f[4];
    delta[sp->O2] += d1D * f[4];
}

// Humid air at T, p: returns total density and fills y (cm^-3).
#define BOLTZMANN_J_K 1.380649e-23
internal f64 air_feed(AirSpecies *sp, u32 n, f64 T_K, f64 p_Pa, f64 x_H2O, f64 *y) {
    f64 total = p_Pa / (BOLTZMANN_J_K * T_K) * 1e-6;   // m^-3 -> cm^-3
    for (u32 s = 0; s < n; ++s) y[s] = 0.0;
    f64 dry = 1.0 - x_H2O;
    y[sp->N2]  = total * dry * 0.79;   // argon lumped into N2
    y[sp->O2]  = total * dry * 0.21;
    y[sp->H2O] = total * x_H2O;
    return total;
}
