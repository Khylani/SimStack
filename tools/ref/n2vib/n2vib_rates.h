// Rate expressions for the N2(X,v) reference ladder, with sources. The
// literature table behind them (values, uncertainty, validation data) is in
// tools/ref/n2vib/README.md.
#pragma once

#define N2VIB_KB_EV 8.617333262e-5   // eV/K
#define N2VIB_CHI   0.006073         // we xe / we (LoKI), for the SSH scalings

// --- level energies: Dunham expansion, N2 X (Huber-Herzberg, from Lofthus & Krupenie,
// JPCRD 6, 113 (1977)): we 2358.57, wexe 14.324, weye -0.00226 cm^-1. Reliable to v ~ 40.
internal f64 n2_level_eV(u32 v) {
    f64 x = v + 0.5, x0 = 0.5;
    f64 g = 2358.57 * x - 14.324 * x * x - 0.00226 * x * x * x;
    f64 g0 = 2358.57 * x0 - 14.324 * x0 * x0 - 0.00226 * x0 * x0 * x0;
    return (g - g0) * 1.239841984e-4;
}

internal f64 n_atm(f64 T) { return 101325.0 / (1.380649e-23 * T) * 1e-6; }   // cm^-3 at 1 atm

// --- V-V: N2(v) + N2(w-1) -> N2(v-1) + N2(w)  (LoKI nitrogenMolecularVV.m; Alves et al.,
// PSST 21, 045008 (2012); Guerra et al., PSST 28, 073001 (2019): SSH fitted to Billing):
//   k = [v/(1-chi v)] [w/(1-chi w)] (1-chi)^2 F(Y) 6.354e-23 T^1.5 / kor     m^3/s
//   Y = 9.97 |v - w| / sqrt(T);  F = (3 - e^(-Y/1.5)) e^(-Y/1.5) / 2  (Y <= 20)
//   kor = 39.0625 - 1.5625 v (v < 10), 25.2 + 24.1 ((v-10)/30)^3 otherwise
// k10^01 = 1.0e-14 cm^3/s at 330 K. The formula is applied in the exothermic
// direction and the reverse follows by detailed balance.
internal f64 vv_loki(u32 v, u32 w, f64 T) {   // donor v -> v-1, acceptor w-1 -> w (w >= 1)
    f64 chi = N2VIB_CHI;
    f64 Y = 9.97 * fabs((f64)v - (f64)w) / sqrt(T);
    f64 F = Y <= 20.0 ? 0.5 * (3.0 - exp(-Y / 1.5)) * exp(-Y / 1.5)
                      : 8.0 * sqrt(3.14159265358979 / 3.0) * pow(Y, 7.0 / 3.0) * exp(-3.0 * pow(Y, 2.0 / 3.0));
    f64 kor = v < 10 ? 39.0625 - 1.5625 * v : 25.2 + 24.1 * pow((v - 10.0) / 30.0, 3.0);
    f64 k = (v / (1.0 - chi * v)) * (w / (1.0 - chi * w)) * (1.0 - chi) * (1.0 - chi) * F * 6.354e-23 * pow(T, 1.5) / kor;
    return k * 1e6;   // cm^3/s
}
// Our enumeration: N2(v) + N2(w) -> N2(v-1) + N2(w+1).
internal f64 k_vv(u32 v, u32 w, f64 T, const f64 *E) {
    f64 delta = (E[v] - E[v - 1]) - (E[w + 1] - E[w]);           // energy released
    if (delta >= 0.0) return vv_loki(v, w + 1, T);
    // endothermic: reverse of N2(w+1) + N2(v-1) -> N2(w) + N2(v)
    return vv_loki(w + 1, v, T) * exp(delta / (N2VIB_KB_EV * T));
}

// --- V-T and V-V' losses, k(v -> v-1) = v k10 (harmonic scaling, Guerra & Loureiro 1995)
enum { VT_N2, VT_O2, VT_O, VT_H2O, VT_AIR };
internal f64 k_vt(u32 kind, u32 v, f64 T, const f64 *E) {
    f64 t13 = pow(T, -1.0 / 3.0), k10 = 0;
    f64 lt = 1.0 - exp(-3353.0 / T);   // Landau-Teller: 1/tau = N k10 (1 - e^(-theta/T))
    switch (kind) {
        // Millikan & White, JCP 39, 3209 (1963) (the LoKI fit diverges below ~330 K)
        case VT_N2:  k10 = 1.0 / (exp(220.0 * (t13 - 0.03) - 18.42) * n_atm(T) * lt); break;
        // Colgan & Levitt, Trans. Faraday Soc. 63, 2898 (1967)
        case VT_O2:  k10 = 1.0 / (exp(162.0 * (t13 - 0.03) - 18.42) * n_atm(T) * lt); break;
        // McNeal, Whitson & Cook, JGR 79, 1527 (1974): 3.32e-15 (300 K) .. 4.43e-14 (740 K),
        // Arrhenius interpolation
        case VT_O:   k10 = 3.3e-15 * exp(-1235.0 * (1.0 / T - 1.0 / 300.0)); break;
        // Center & Newton, JCP 68, 3327 (1978): p tau = exp(21 T^-1/3 - 1.35) us atm
        // (Whitson & McNeal 1977: 1.2e-14 at 293 K; ISO 9613-1 / Griffin 1981: 5-7e-15)
        case VT_H2O: k10 = 1.0 / (exp(21.0 * t13 - 1.35) * 1e-6 * n_atm(T)); break;
        // dry atmospheric air as a whole (N2-O2 V-V', CO2 near-resonant V-V, ...):
        // ISO 9613-1 relaxation frequency of N2 in dry air, 9 (T/293.15)^-1/2 Hz atm^-1,
        // 1/tau = 2 pi f_r; applied as a third body (the dryer keeps the CO2)
        case VT_AIR: k10 = 2.0 * 3.14159265358979 * 9.0 * pow(T / 293.15, -0.5) / (n_atm(T) * lt); break;
    }
    return (f64)v * k10;
}

// --- Zeldovich, state-specific: N2(v) + O -> NO + N. The literature disagrees ~100x:
//   ZEL_FM      Fridman-Macheret alpha = 1, A fixed so a Boltzmann ladder reproduces the
//               thermal rate 3.0e-10 exp(-38250/T) (Baulch et al., JPCRD 23, 847 (1994)) at 3000 K
//   ZEL_STEP13  1e-13 for v >= 13, 0 below (Guerra & Loureiro 1995, LoKI-C)
//   ZEL_STEP11  1e-11 for v >= 13 (Capitelli / Gordiets)
enum { ZEL_FM, ZEL_STEP13, ZEL_STEP11 };
global u32 g_zeldovich_model = ZEL_FM;
internal f64 k_zeldovich(u32 v, f64 T, const f64 *E) {
    if (g_zeldovich_model == ZEL_STEP13) return v >= 13 ? 1e-13 : 0.0;
    if (g_zeldovich_model == ZEL_STEP11) return v >= 13 ? 1e-11 : 0.0;
    static f64 A = 0;
    const f64 Ea = 38250.0 * N2VIB_KB_EV;
    if (A == 0) {
        f64 Tn = 3000.0, z = 0, s = 0;
        for (u32 i = 0; i <= VMAX; ++i) {
            f64 Ei = n2_level_eV(i), f = exp(-Ei / (N2VIB_KB_EV * Tn));
            z += f;
            s += f * exp(-Maximum(Ea - Ei, 0.0) / (N2VIB_KB_EV * Tn));
        }
        A = 3.0e-10 * exp(-38250.0 / Tn) / (s / z);
    }
    return A * exp(-Maximum(Ea - E[v], 0.0) / (N2VIB_KB_EV * T));
}

// e-V from v > 0: k(v -> v+n) = k(0 -> n) / (1 + 0.15 v)  (Bourdon & Vervisch, JTHT 14, 489
// (2000), as LoKI nitrogenEVscaling.m)
internal f64 ev_scaling(u32 v) { return 1.0 / (1.0 + 0.15 * v); }
