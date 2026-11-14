// The sim's water chemistry on the cases tools/ref/phreeqc/compare.py runs
// through PHREEQC. Prints `key value` lines. Built by build.sh in this folder.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dlfcn.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>
#include <sched.h>

#define DEBUG_RECORD_BASE 0
#include "../../../src/sim_platform.h"
#include "../../../src/sim/dbd.cpp"
#include "../../../src/sim/nft.cpp"

// Batch water: source fill, then N added as nitric / nitrous acid (closed system).
internal f64 batch_ph(PawParams *pp, f64 n_mmol_L, f64 nitrite_share) {
    PawWater w;
    paw_fill_source(&w, pp, 1.0);
    paw_solve_ph(&w);
    w.no3_mmol += n_mmol_L * (1.0 - nitrite_share);
    w.no2_mmol += n_mmol_L * nitrite_share;
    return paw_solve_ph(&w);
}

internal void source_case(const char *name, u32 water) {
    NftParams p = nft_preset(NftPreset_PawSkid);
    nft_set_water(&p, water);
    PawParams *pp = &p.paw;
    PawWater w;
    paw_fill_source(&w, pp, 1.0);
    NftAcidBase a = nft_acid_base(w.T_C, paw_ionic_strength(&w));
    f64 aH = pow(10.0, -(f64)pp->source_pH);
    f64 r1 = a.K1 / (aH * a.g1), r2 = a.K2 * a.g1 / (aH * a.g2);
    f64 dic = w.dic_mmol;                              // mmol in 1 L
    f64 co2 = dic / (1.0 + r1 * (1.0 + r2));
    printf("%s.T_C %.4f\n", name, w.T_C);
    printf("%s.source_pH %.6f\n", name, (f64)pp->source_pH);
    printf("%s.kh_meq %.6f\n", name, (f64)pp->source_kh_meq);
    printf("%s.dic_mmol %.6f\n", name, dic);
    printf("%s.co2_mmol %.6f\n", name, co2);
    printf("%s.I %.6e\n", name, paw_ionic_strength(&w));
    printf("%s.log_pco2 %.4f\n", name, log10(co2 * 1e-3 / a.K0));
    // titration: pH after n mmol/L of acid N (15 % as nitrite, the model's batch share)
    for (u32 k = 0; k <= 30; ++k) {
        f64 n = 0.1 * k;
        printf("%s.titr %.3f %.6f\n", name, n, batch_ph(pp, n, 0.15));
    }
    // N to reach the batch target pH (bisection on the closed-system curve)
    f64 lo = 0.0, hi = 10.0;
    for (u32 it = 0; it < 200; ++it) { f64 m = 0.5 * (lo + hi); if (batch_ph(pp, m, 0.15) > pp->batch_pH) lo = m; else hi = m; }
    printf("%s.batch_pH %.3f\n", name, (f64)pp->batch_pH);
    printf("%s.n_to_target_mmol %.6f\n", name, 0.5 * (lo + hi));
}

// NFT working solution: recipe made up in the source water, open to air CO2.
internal void nft_case(const char *name, u32 water, f64 T_C) {
    NftParams p = nft_preset(NftPreset_PawSkid);
    nft_set_water(&p, water);
    f64 c[NftF_Count] = {};
    for (u32 i = 0; i < NFT_IONS; ++i) c[NFT_ION0 + i] = p.recipe[i] + p.water.ion[i];
    f64 R = nft_fill_R_conc(&p);
    NftWaterChem wc = nft_water_chem(c, R);
    f64 ph = nft_ph(c, T_C, R, nft_pco2_atm(&p), 6.0);
    printf("%s.T_C %.3f\n", name, T_C);
    printf("%s.TA_meq %.6f\n", name, wc.TA_meq);
    printf("%s.I %.6e\n", name, wc.I);
    printf("%s.pH %.6f\n", name, ph);
    printf("%s.log_pco2 %.4f\n", name, log10(nft_pco2_atm(&p)));
    for (u32 i = 0; i < NFT_IONS; ++i) printf("%s.ion %s %.6f\n", name, nft_field_names[NFT_ION0 + i], c[NFT_ION0 + i]);
    printf("%s.na %.4f\n%s.cl %.4f\n%s.sio2 %.4f\n", name, (f64)p.water.na, name, (f64)p.water.cl, name, (f64)p.water.sio2);
    // nitric acid (mmol/L) to the pH set point, open to air CO2
    auto ph_after = [&](f64 acid) {
        f64 cc[NftF_Count];
        for (u32 k = 0; k < NftF_Count; ++k) cc[k] = c[k];
        cc[NftF_NO3] += acid * 14.007;
        return nft_ph(cc, T_C, R, nft_pco2_atm(&p), 6.0);
    };
    for (u32 k = 0; k <= 20; ++k) printf("%s.titr %.3f %.6f\n", name, 0.1 * k, ph_after(0.1 * k));
    f64 lo = 0.0, hi = 20.0;
    for (u32 it = 0; it < 200; ++it) { f64 m = 0.5 * (lo + hi); if (ph_after(m) > p.ph_set) lo = m; else hi = m; }
    printf("%s.ph_set %.3f\n%s.acid_to_set_mmol %.6f\n", name, (f64)p.ph_set, name, 0.5 * (lo + hi));
}

int main() {
    source_case("design", NftWater_Design);
    source_case("w78", NftWater_MoloaaW78);
    nft_case("nft_w78_20", NftWater_MoloaaW78, 20.0);
    nft_case("nft_w78_30", NftWater_MoloaaW78, 30.0);
    nft_case("nft_design_20", NftWater_Design, 20.0);
    return 0;
}
