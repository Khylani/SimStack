// Water physics for the NFT model, f64: vapour pressure and evaporation,
// root-mat film hydraulics (Brinkman flow through a fibrous layer), and the
// pH of a nutrient solution from its alkalinity (carbonate system open to air
// CO2, phosphate, ammonium, water; Davies activities). Included by nft.cpp.

// --- vapour, evaporation ---------------------------------------------------------

#define NFT_P_ATM      101325.0
#define NFT_CP_AIR     1006.0      // J / (kg K)
#define NFT_LEWIS_2_3  0.897       // Le^(2/3), Le = 0.85 for water vapour in air

// Saturation vapour pressure over water, Pa (Buck 1996: within 0.05 % of
// IAPWS over 0-50 C), and its temperature derivative.
internal f64 nft_es_Pa(f64 T_C) { return 611.21 * exp((18.678 - T_C / 234.5) * (T_C / (257.14 + T_C))); }
internal f64 nft_es_slope(f64 T_C) {   // d e_s / dT, Pa/K
    f64 d = 257.14 + T_C;
    f64 df = -T_C / (234.5 * d) + (18.678 - T_C / 234.5) * 257.14 / (d * d);
    return nft_es_Pa(T_C) * df;
}
// Latent heat of vaporization, J/kg (linear fit, 0-40 C).
internal f64 nft_latent(f64 T_C) { return 2.501e6 - 2361.0 * T_C; }

// Evaporation flux per Pa of vapour-pressure difference, kg / (m^2 s Pa), from
// the convective coefficient by the Chilton-Colburn (Lewis) analogy:
//   E = h_c / (cp Le^(2/3)) (w_s - w_a),  w = 0.622 e / p.
internal f64 nft_evap_coef(f64 h_c) { return h_c / (NFT_CP_AIR * NFT_LEWIS_2_3) * 0.622 / NFT_P_ATM; }

// --- root mats --------------------------------------------------------------------

// Permeability (m^2) of a bed of root fibres of diameter d at solid fraction
// phi: the smaller of Jackson & James (1986, dilute random fibres) and
// Carman-Kozeny for fibres (Kozeny constant 5, S0 = 4/d, dense beds). The two
// cross near phi ~ 0.2 and each is the better fit on its own side.
internal f64 nft_root_perm(f64 phi, f64 d) {
    f64 eps = 1.0 - phi;
    f64 kc = eps * eps * eps * d * d / (80.0 * phi * phi);
    f64 ln_term = -log(phi) - 0.931;
    f64 jj = ln_term > 0.0 ? 0.25 * d * d * 3.0 / (20.0 * phi) * ln_term : kc;
    return Minimum(kc, jj);
}

// Flow per unit width (m^2/s) of a film of depth h on a grade, flowing through
// a porous layer of permeability K over its whole depth: Brinkman's equation
//   nu u'' - nu u / K + g S = 0,  u(0) = 0 (floor),  u'(h) = 0 (free surface)
// gives u = (g S K / nu)(1 - cosh((h - z)/sqrt K) / cosh(h / sqrt K)) and
//   q = (g S K / nu) (h - sqrt K tanh(h / sqrt K)),
// which is the Nusselt film g S h^3 / (3 nu) as K -> inf and Darcy g S K h / nu
// in a dense mat.
internal f64 nft_brinkman_q(f64 h, f64 K, f64 gS_nu) {
    f64 sk = sqrt(K), z = h / sk;
    f64 core;
    if (z < 0.05) {
        f64 z2 = z * z;   // h - sqrt K tanh z = sqrt K (z^3/3 - 2 z^5/15 + 17 z^7/315 - 62 z^9/2835 + ...)
        core = sk * z * z2 * (1.0 / 3.0 - z2 * (2.0 / 15.0 - z2 * (17.0 / 315.0 - z2 * (62.0 / 2835.0))));
    } else {
        core = h - sk * tanh(z);
    }
    return gS_nu * K * core;
}

// Residual F(log h) = log(Brinkman flow at depth h with the cell's roots spread
// through it / q) and its analytic derivative dF/dlog h (for Newton).
//   dq/dh|K = G K tanh^2 z,   dq/dK|h = G (3h - 3 sqrt K tanh z - h tanh^2 z) / 2,
//   dphi/dh = -tr / h^2 below phi_max (0 above), dK/dphi from the branch in use.
internal f64 nft_root_residual(f64 lh, f64 qu, f64 tr, f64 d, f64 phi_max, f64 gS_nu, f64 *dF) {
    f64 h = exp(lh);
    b32 capped = tr / h >= phi_max;
    f64 phi = capped ? phi_max : tr / h;
    f64 eps = 1.0 - phi;
    f64 kc = eps * eps * eps * d * d / (80.0 * phi * phi);
    f64 L = -log(phi) - 0.931;
    f64 jj = L > 0.0 ? 0.25 * d * d * 3.0 / (20.0 * phi) * L : kc;
    f64 K, dK_dphi;
    if (jj < kc) { K = jj; dK_dphi = -0.25 * d * d * 3.0 / 20.0 * (L + 1.0) / (phi * phi); }
    else         { K = kc; dK_dphi = kc * (-3.0 / eps - 2.0 / phi); }
    f64 sk = sqrt(K), z = h / sk, t = tanh(z);
    f64 core;   // h - sqrt K tanh z, by series where it cancels (as nft_brinkman_q)
    if (z < 0.05) { f64 z2 = z * z; core = sk * z * z2 * (1.0 / 3.0 - z2 * (2.0 / 15.0 - z2 * (17.0 / 315.0 - z2 * (62.0 / 2835.0)))); }
    else core = h - sk * t;
    f64 q = gS_nu * K * core;
    f64 dq_dh = gS_nu * K * t * t;
    f64 dq_dK = 0.5 * gS_nu * (3.0 * h - 3.0 * sk * t - h * t * t);
    f64 dphi_dh = capped ? 0.0 : -tr / (h * h);
    *dF = h * (dq_dh + dq_dK * dK_dphi * dphi_dh) / q;
    return log(q / qu);
}

// Film through one cell's root mat. qu = flow per unit width, h0 = the
// root-free film depth (laminar/turbulent, nft_film_depth), tr = root volume
// per unit floor area (m), d = root diameter. Two limits, take the deeper:
//   sparse roots: they displace water and the film rises by their volume, h0 + tr;
//   dense mat:   the flow must pass through the mat, Brinkman depth with
//                phi = min(tr / h, phi_max) (above phi_max the rest of the mat
//                stands above the water).
// Either way the water depth h (1 - phi) >= h0, so roots only lower the
// Courant number and the substep stays set by the root-free channel.
// Newton in log h with the analytic derivative, bracketed below by the sparse
// limit (the residual is monotone in h); `guess` is the previous solution, so
// a re-solve takes 1-3 iterations.
internal void nft_root_film(f64 qu, f64 S, f64 nu, f64 h0, f64 tr, f64 d, f64 phi_max, f64 guess,
                            f64 *h_out, f64 *phi_out) {
    f64 ha = h0 + tr;
    *h_out = ha;
    *phi_out = tr > 0.0 ? tr / ha : 0.0;
    if (tr <= 0.0 || qu <= 0.0) return;
    f64 gS_nu = NFT_G * Maximum(S, 1e-5) / nu;
    f64 lo = log(ha), hi = 0.0;                        // 1 m: no channel film is deeper
    f64 d_lo;
    if (nft_root_residual(lo, qu, tr, d, phi_max, gS_nu, &d_lo) >= 0.0) return;   // free film carries it: sparse limit
    f64 x = guess > ha && guess < 1.0 ? log(guess) : lo + 0.5;
    for (u32 it = 0; it < 60; ++it) {
        f64 dfdx;
        f64 f = nft_root_residual(x, qu, tr, d, phi_max, gS_nu, &dfdx);
        if (fabs(f) < 1e-12) break;                    // flow matched to 1e-12 (relative)
        if (f < 0.0) lo = x; else hi = x;
        f64 xn = dfdx > 0.0 ? x - f / dfdx : 0.5 * (lo + hi);
        if (!(xn > lo && xn < hi)) xn = 0.5 * (lo + hi);
        b32 done = fabs(xn - x) < 1e-12 || hi - lo < 1e-12;
        x = xn;
        if (done) break;
    }
    *h_out = exp(x);
    *phi_out = Minimum(tr / *h_out, phi_max);
}

// --- pH ---------------------------------------------------------------------------
//
// Alkalinity (Dickson's proton reference: H2CO3*, H2PO4-, NH4+, H2O):
//   TA = [HCO3-] + 2[CO3--] + [OH-] + [HPO4--] + 2[PO4---] + [NH3] - [H+] - [H3PO4]
// Charge balance makes TA a conservative sum of the strong ions:
//   TA = sum(cations, NH4 as total ammonium) - sum(NO3-, P_total, 2 SO4--) + R,
// where R collects ions the model does not track (Na+, Cl-, the bicarbonate of
// the tap water...). So plants change pH only through the ions they take up
// (nitrate-dominated uptake raises it), acid / base dosing through the NO3 / K
// they add, and water through R. Given TA, pH solves the speciation above with
// CO2* in equilibrium with the air.

struct NftAcidBase {
    f64 K0;            // CO2 solubility, mol / (L atm)                 Weiss 1974
    f64 K1, K2;        // carbonic acid                                  Plummer & Busenberg 1982
    f64 Kw;            // water                                          Harned & Owen 1958
    f64 Kn;            // NH4+ = NH3 + H+                                Emerson et al. 1975
    f64 P1, P2, P3;    // phosphoric acid, van't Hoff from 25 C          (pK 2.148, 7.198, 12.35)
    f64 g1, g2, g3;    // Davies activity coefficients, |z| = 1, 2, 3
    // ion pairs (formation constants from the free ions; phreeqc.dat, Nordstrom et al. 1990)
    f64 CaHP, CaH2P, CaP, MgHP, MgH2P, MgP, KHP;   // with HPO4--, H2PO4-, PO4---
    f64 CaS, MgS, KS;                              // with SO4--
    f64 CaHC, CaC, MgHC, MgC;                      // Ca/Mg + CO3-- + H+ = MHCO3+, M + CO3-- = MCO3
};

// Totals (mol/L) of the cations and sulfate that form the pairs above.
struct NftPairs { f64 Ca, Mg, K, S; };

internal f64 nft_vant_hoff_log(f64 logK25, f64 dH_J, f64 TK) {
    return logK25 - dH_J / (8.314462618 * 2.302585093) * (1.0 / TK - 1.0 / 298.15);
}

internal NftAcidBase nft_acid_base(f64 T_C, f64 I) {
    NftAcidBase a;
    f64 TK = T_C + 273.15, lt = log10(TK);
    a.K0 = exp(-58.0931 + 90.5069 * (100.0 / TK) + 22.2940 * log(TK / 100.0));
    a.K1 = pow(10.0, -356.3094 - 0.06091964 * TK + 21834.37 / TK + 126.8339 * lt - 1684915.0 / (TK * TK));
    a.K2 = pow(10.0, -107.8871 - 0.03252849 * TK + 5151.79 / TK + 38.92561 * lt - 563713.9 / (TK * TK));
    a.Kw = pow(10.0, -(4470.99 / TK - 6.0875 + 0.01706 * TK));
    a.Kn = pow(10.0, -(0.09018 + 2729.92 / TK));
    a.P1 = pow(10.0, nft_vant_hoff_log(-2.148, -8.0e3, TK));
    a.P2 = pow(10.0, nft_vant_hoff_log(-7.198, 3.6e3, TK));
    a.P3 = pow(10.0, nft_vant_hoff_log(-12.35, 16.0e3, TK));
    const f64 kcal = 4184.0;
    a.CaHP  = pow(10.0, nft_vant_hoff_log(2.739, 3.3 * kcal, TK));
    a.CaH2P = pow(10.0, nft_vant_hoff_log(1.408, 3.4 * kcal, TK));
    a.CaP   = pow(10.0, nft_vant_hoff_log(6.459, 3.1 * kcal, TK));
    a.MgHP  = pow(10.0, nft_vant_hoff_log(2.87, 3.3 * kcal, TK));
    a.MgH2P = pow(10.0, nft_vant_hoff_log(1.513, 3.4 * kcal, TK));
    a.MgP   = pow(10.0, nft_vant_hoff_log(6.589, 3.1 * kcal, TK));
    a.KHP   = pow(10.0, 0.29);
    a.CaS   = pow(10.0, nft_vant_hoff_log(2.25, 1.325 * kcal, TK));
    a.MgS   = pow(10.0, nft_vant_hoff_log(2.42, 19.0e3, TK));
    a.KS    = pow(10.0, nft_vant_hoff_log(0.6, -10.4e3, TK));
    a.CaHC  = pow(10.0, nft_vant_hoff_log(10.91, 4.38 * kcal, TK));
    a.CaC   = pow(10.0, nft_vant_hoff_log(3.224, 3.545 * kcal, TK));
    a.MgHC  = pow(10.0, nft_vant_hoff_log(11.399, -2.771 * kcal, TK));
    a.MgC   = pow(10.0, nft_vant_hoff_log(2.98, 2.713 * kcal, TK));
    // Davies: log g = -A z^2 (sqrt I / (1 + sqrt I) - 0.3 I); A from the
    // dielectric constant of water (Malmberg & Maryott 1956).
    f64 eps = 87.74 - 0.40008 * T_C + 9.398e-4 * T_C * T_C - 1.41e-6 * T_C * T_C * T_C;
    f64 A = 1.82483e6 / pow(eps * TK, 1.5);
    f64 sI = sqrt(Maximum(I, 0.0));
    f64 lg1 = -A * (sI / (1.0 + sI) - 0.3 * I);
    a.g1 = pow(10.0, lg1); a.g2 = pow(10.0, 4.0 * lg1); a.g3 = pow(10.0, 9.0 * lg1);
    return a;
}

// Alkalinity (mol/L) of the speciated solution at pH (activity scale).
//
// With `pr`, Ca, Mg and K pair with phosphate, sulfate and carbonate. In a
// working nutrient solution (Ca ~4 mM, P ~1.3 mM) about 10 % of the calcium
// is paired, and CaHPO4 / MgHPO4 pull the phosphate toward HPO4--, which
// carries alkalinity: ignoring them put the pH 0.1-0.17 high and the acid
// demand 4-5 % high against PHREEQC (tools/ref/phreeqc). The free ions follow
// from a fixed-point iteration on the Ca / Mg / K / sulfate mass balances; a
// fixed iteration count keeps the function smooth for the Newton solve.
internal f64 nft_alk_species(const NftAcidBase *a, f64 pH, f64 pCO2_atm, f64 PT, f64 NT);   // free ions only (below)
internal f64 nft_alk_species(const NftAcidBase *a, f64 pH, f64 pCO2_atm, f64 PT, f64 NT, const NftPairs *pr) {
    if (pr && (pr->Ca > 0.0 || pr->Mg > 0.0 || pr->K > 0.0)) {
        f64 aH = pow(10.0, -pH);
        f64 g1 = a->g1, g2 = a->g2, g3 = a->g3;
        f64 co2 = a->K0 * pCO2_atm;
        f64 hco3 = a->K1 * co2 / (aH * g1);
        f64 co3 = a->K1 * a->K2 * co2 / (aH * aH * g2);
        f64 oh = a->Kw / (aH * g1), h = aH / g1;
        f64 r1 = a->P1 / (aH * g1), r2 = a->P2 * g1 / (aH * g2), r3 = a->P3 * g2 / (aH * g3);
        f64 rn = a->Kn * g1 / aH;
        f64 nh3 = NT * rn / (1.0 + rn);
        f64 g22 = g2 * g2, g23 = g2 * g3 / g1;
        f64 ca = pr->Ca, mg = pr->Mg, k = pr->K, x = 0.0, hpo4 = 0.0, po4 = 0.0, so4 = pr->S;
        for (u32 it = 0; it < 10; ++it) {
            // phosphate relative to free H2PO4- (x)
            f64 D = 1.0 / r1 + 1.0 + r2 * (1.0 + (a->CaHP * ca + a->MgHP * mg) * g22 + a->KHP * g2 * k) +
                    r2 * r3 * (1.0 + (a->CaP * ca + a->MgP * mg) * g23) + (a->CaH2P * ca + a->MgH2P * mg) * g2;
            x = PT / D;
            hpo4 = x * r2; po4 = hpo4 * r3;
            so4 = pr->S / (1.0 + (a->CaS * ca + a->MgS * mg) * g22 + a->KS * g2 * k);
            f64 cpairs = a->CaHC * g22 * co3 * aH / g1 + a->CaC * g22 * co3;
            f64 mpairs = a->MgHC * g22 * co3 * aH / g1 + a->MgC * g22 * co3;
            ca = pr->Ca / (1.0 + a->CaHP * g22 * hpo4 + a->CaH2P * g2 * x + a->CaP * g23 * po4 + a->CaS * g22 * so4 + cpairs);
            mg = pr->Mg / (1.0 + a->MgHP * g22 * hpo4 + a->MgH2P * g2 * x + a->MgP * g23 * po4 + a->MgS * g22 * so4 + mpairs);
            k = pr->K / (1.0 + a->KS * g2 * so4 + a->KHP * g2 * hpo4);
        }
        f64 h3po4 = x / r1;
        f64 hp_pairs = (a->CaHP * ca + a->MgHP * mg) * g22 * hpo4 + a->KHP * g2 * k * hpo4;   // count as HPO4--
        f64 p_pairs = (a->CaP * ca + a->MgP * mg) * g23 * po4;                                // count as PO4---
        f64 hc_pairs = (a->CaHC * ca + a->MgHC * mg) * g22 * co3 * aH / g1;                    // as HCO3-
        f64 c_pairs = (a->CaC * ca + a->MgC * mg) * g22 * co3;                                 // as CO3--
        return hco3 + hc_pairs + 2.0 * (co3 + c_pairs) + oh + hpo4 + hp_pairs + 2.0 * (po4 + p_pairs) + nh3 - h - h3po4;
    }
    return nft_alk_species(a, pH, pCO2_atm, PT, NT);
}

internal f64 nft_alk_species(const NftAcidBase *a, f64 pH, f64 pCO2_atm, f64 PT, f64 NT) {
    f64 aH = pow(10.0, -pH);
    f64 co2 = a->K0 * pCO2_atm;
    f64 hco3 = a->K1 * co2 / (aH * a->g1);
    f64 co3 = a->K1 * a->K2 * co2 / (aH * aH * a->g2);
    f64 oh = a->Kw / (aH * a->g1);
    f64 h = aH / a->g1;
    f64 r1 = a->P1 / (aH * a->g1);                  // [H2PO4-] / [H3PO4]
    f64 r2 = a->P2 * a->g1 / (aH * a->g2);          // [HPO4--] / [H2PO4-]
    f64 r3 = a->P3 * a->g2 / (aH * a->g3);          // [PO4---] / [HPO4--]
    f64 h3po4 = PT / (1.0 + r1 * (1.0 + r2 * (1.0 + r3)));
    f64 hpo4 = h3po4 * r1 * r2, po4 = hpo4 * r3;
    f64 rn = a->Kn * a->g1 / aH;                    // [NH3] / [NH4+]
    f64 nh3 = NT * rn / (1.0 + rn);
    return hco3 + 2.0 * co3 + oh + hpo4 + 2.0 * po4 + nh3 - h - h3po4;
}

struct NftWaterChem {
    f64 TA_meq;        // total alkalinity, meq/L
    f64 I;             // ionic strength, mol/L
    f64 PT, NT;        // total phosphate / ammonium, mol/L
    NftPairs pr;       // pair-forming totals
};

// From the ion concentrations (mg/L, NftField order) and the untracked charge R (meq/L).
internal NftWaterChem nft_water_chem(const f64 *c, f64 R_meq) {
    NftWaterChem w;
    f64 ta = R_meq, I = 0.0;
    for (u32 i = 0; i < NFT_IONS; ++i) {
        f64 meq = Maximum(c[NFT_ION0 + i], 0.0) / nft_meq_mg[i];
        ta += nft_charge_sign[i] * meq;
        // z^2 c / 2 with c in mol/L = meq / |z| / 1000: z = 2 for Ca, Mg, SO4
        b32 divalent = NFT_ION0 + i == NftF_Ca || NFT_ION0 + i == NftF_Mg || NFT_ION0 + i == NftF_S;
        I += 0.5e-3 * meq * (divalent ? 2.0 : 1.0);
    }
    I += 0.5e-3 * fabs(R_meq);                      // untracked ions taken as monovalent
    w.TA_meq = ta;
    w.I = I;
    w.PT = Maximum(c[NftF_P], 0.0) / 30974.0;
    w.NT = Maximum(c[NftF_NH4], 0.0) / 14007.0;
    w.pr.Ca = Maximum(c[NftF_Ca], 0.0) / 40078.0;
    w.pr.Mg = Maximum(c[NftF_Mg], 0.0) / 24305.0;
    w.pr.K = Maximum(c[NftF_K], 0.0) / 39098.0;
    w.pr.S = Maximum(c[NftF_S], 0.0) / 32065.0;
    return w;
}

// pH from TA: the species alkalinity rises monotonically with pH, so Newton on
// a [0, 14] bracket with bisection fallback converges to 1e-10 in ~6 steps.
internal f64 nft_solve_ph(const NftAcidBase *a, f64 TA_mol, f64 pCO2_atm, f64 PT, f64 NT, f64 guess, const NftPairs *pr = 0) {
    f64 lo = 0.0, hi = 14.0;
    f64 x = guess > lo && guess < hi ? guess : 6.0;
    for (u32 it = 0; it < 80; ++it) {
        f64 f = nft_alk_species(a, x, pCO2_atm, PT, NT, pr) - TA_mol;
        if (f > 0.0) hi = x; else lo = x;
        if (hi - lo < 1e-10) break;
        f64 d = (nft_alk_species(a, x + 1e-6, pCO2_atm, PT, NT, pr) - TA_mol - f) * 1e6;
        f64 xn = d > 0.0 ? x - f / d : 0.5 * (lo + hi);
        if (!(xn > lo && xn < hi)) xn = 0.5 * (lo + hi);
        if (fabs(xn - x) < 1e-11) { x = xn; break; }
        x = xn;
    }
    return x;
}

internal f64 nft_ph(const f64 *c, f64 T_C, f64 R_meq, f64 pCO2_atm, f64 guess) {
    NftWaterChem w = nft_water_chem(c, R_meq);
    NftAcidBase a = nft_acid_base(T_C, w.I);
    return nft_solve_ph(&a, w.TA_meq * 1e-3, pCO2_atm, w.PT, w.NT, guess, &w.pr);
}
