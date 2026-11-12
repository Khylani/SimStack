// Humid-air neutral chemistry for a DBD reactor and its afterglow, plus the
// surrogate electron-impact source (G-values) that stands in for the
// Boltzmann/plasma part until the M2 reference solver (ChemPlasKin) and the
// generated reduction tables exist.
//
// 17 species, 40 reactions. Rate data: JPL 19-5 / IUPAC evaluations at the
// low-pressure-limit / fall-off forms given there, rounded to two digits.
// It is a *neutral* afterglow mechanism: ions, excited states (O(1D),
// N2(A), N2(v)) and surface chemistry are not modelled; their net effect on
// radical yields is folded into the G-values below.

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
    kin_add_reaction(model, "N + NO2 -> N2O + O",          kin_arr(5.8e-12, 0.0, -220.0));
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

    kin_mech_finalize(model);
}

// ----------------------------------------------------------------------------
// Discharge source. Each filamentary micro-discharge deposits energy that
// electron impact turns into radicals; at the reduced fields of air DBDs
// (100-200 Td) the yields per 100 eV are roughly (Penetrante et al. 1997,
// Kossyi et al. 1992, Sakiyama et al. 2012 for humid air):
//
//   O2  -> 2 O        G(O)  ~ 4-6 per 100 eV at 21 % O2 (incl. N2* quenching by O2)
//   N2  -> 2 N        G(N)  ~ 0.5-1
//   H2O -> OH + H     G(OH) ~ 0.5-1 at 1-3 % H2O (incl. O(1D) + H2O)
//
// scaled linearly with the parent's mole fraction relative to the reference
// gas. Parents are consumed so N, O and H atoms are conserved exactly.
// These are the tunables the reduction tables will replace.

struct AirSourceModel {
    f64 g_O_per_100eV;    // at x_O2 = 0.21
    f64 g_N_per_100eV;    // at x_N2 = 0.78
    f64 g_OH_per_100eV;   // at x_H2O = 0.02
};

internal AirSourceModel air_source_defaults() {
    AirSourceModel s;
    s.g_O_per_100eV = 5.0;
    s.g_N_per_100eV = 0.8;
    s.g_OH_per_100eV = 0.6;
    return s;
}

#define EV_PER_JOULE 6.241509074e18

// Species change per cm^3 for `energy_J_cm3` deposited into gas of density y.
internal void air_source_delta(AirSpecies *sp, AirSourceModel *src, const f64 *y, u32 n, f64 energy_J_cm3, f64 *delta) {
    for (u32 s = 0; s < n; ++s) delta[s] = 0.0;
    f64 total = 0.0;
    for (u32 s = 0; s < n; ++s) total += y[s];
    if (total <= 0.0) return;
    f64 hundreds_eV = energy_J_cm3 * EV_PER_JOULE * 0.01;
    f64 dO  = src->g_O_per_100eV  * (y[sp->O2]  / total / 0.21) * hundreds_eV;
    f64 dN  = src->g_N_per_100eV  * (y[sp->N2]  / total / 0.78) * hundreds_eV;
    f64 dOH = src->g_OH_per_100eV * (y[sp->H2O] / total / 0.02) * hundreds_eV;
    delta[sp->O]   += dO;   delta[sp->O2]  -= 0.5 * dO;
    delta[sp->N]   += dN;   delta[sp->N2]  -= 0.5 * dN;
    delta[sp->OH]  += dOH;  delta[sp->H]   += dOH;  delta[sp->H2O] -= dOH;
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
