// State-to-state N2(X,v) reference: SimStack's air mechanism and discharge
// source (the same code the real-time model runs) plus a vibrational ladder
// N2(v=1..VMAX) written as mass-action reactions, solved together by the same
// Rosenbrock solver with bigger table limits:
//
//   e-V        per filament: N2(v) -> N2(v+dv), LoKI-B++ G(0->dv) per 100 eV
//              (ev_gvalues.csv), Fridman scaling for v > 0
//   V-V        N2(v) + N2(w) -> N2(v-1) + N2(w+1)          (n2vib_rates.h)
//   V-T        N2(v) + M -> N2(v-1) + M, M = N2, O2, O, H2O, with detailed-balance reverses
//   Zeldovich  N2(v) + O -> NO + N, state-specific
//
// "N2" in the mechanism is N2(v=0). Rates are evaluated once at the run's
// fixed gas temperature (the real-time zones are isothermal too).
//
// usage: n2vib_ref T_K x_H2O E_f_mJ_cm3 P_W_cm3 residence_s E/N_Td [novib] [check]

#define KIN_MAX_SPECIES     64
#define KIN_MAX_REACTIONS   2600
#define KIN_MAX_NNZ         4096
#define KIN_MAX_LU_DIV      4096
#define KIN_MAX_LU_SUB      65000

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <semaphore.h>
#define DEBUG_RECORD_BASE 0
#include "sim_platform.h"
#include "sim/dbd.cpp"

#define VMAX 40
#include "n2vib_rates.h"

static u8 g_mem[1u << 30];
global b32 g_no_air_relax = 0;

struct Ladder {
    u32 level[VMAX + 1];        // species index of N2(v); level[0] = N2
    f64 E[VMAX + 1];            // eV above v = 0
    f64 g_ev[11];               // G(0 -> dv) per 100 eV at the run's E/N
};

internal void read_ev_gvalues(const char *path, f64 en, f64 *g) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    char line[1024];
    f64 prev_en = -1, prev[11] = {}, cur[11] = {};
    for (u32 i = 0; i < 11; ++i) g[i] = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == 'E') continue;
        f64 e = strtod(line, 0);
        char *p = strchr(line, ',');
        for (u32 v = 1; v <= 10 && p; ++v) { cur[v] = strtod(p + 1, 0); p = strchr(p + 1, ','); }
        if (e >= en && prev_en >= 0) {
            f64 t = (en - prev_en) / (e - prev_en);
            for (u32 v = 1; v <= 10; ++v) g[v] = prev[v] + t * (cur[v] - prev[v]);
            fclose(f);
            return;
        }
        prev_en = e;
        memcpy(prev, cur, sizeof(cur));
    }
    memcpy(g, prev, sizeof(prev));
    fclose(f);
}

// Mechanism: SimStack air chemistry + the ladder.
internal void build(KinModel *model, AirSpecies *sp, Ladder *L, f64 T, b32 vib) {
    air_mech_define(model, sp);
    L->level[0] = sp->N2;
    for (u32 v = 0; v <= VMAX; ++v) L->E[v] = n2_level_eV(v);
    if (vib) {
        char name[16], eq[64];
        for (u32 v = 1; v <= VMAX; ++v) { snprintf(name, sizeof(name), "N2(v%u)", v); L->level[v] = kin_add_species(model, name); }
        auto nm = [&](u32 v, char *out, u32 n) { if (v == 0) snprintf(out, n, "N2"); else snprintf(out, n, "N2(v%u)", v); };
        char a[16], b[16], c[16], d[16];
        // V-V: every ordered pair (v, w) with v >= 1, w + 1 <= VMAX, skipping the identity v - 1 == w
        for (u32 v = 1; v <= VMAX; ++v)
            for (u32 w = 0; w + 1 <= VMAX; ++w) {
                if (v - 1 == w) continue;
                f64 k = k_vv(v, w, T, L->E);
                if (k <= 0) continue;
                nm(v, a, 16); nm(w, b, 16); nm(v - 1, c, 16); nm(w + 1, d, 16);
                snprintf(eq, sizeof(eq), "%s + %s -> %s + %s", a, b, c, d);
                kin_add_reaction(model, eq, kin_arr(k));
            }
        // V-T, down and up (detailed balance)
        struct Partner { const char *name; u32 kind; } partners[] = {
            { "N2", VT_N2 }, { "O2", VT_O2 }, { "O", VT_O }, { "H2O", VT_H2O }, { "M", VT_AIR },
        };
        u32 n_partners = g_no_air_relax ? 4 : 5;
        for (u32 p = 0; p < n_partners; ++p)
            for (u32 v = 1; v <= VMAX; ++v) {
                f64 kd = k_vt(partners[p].kind, v, T, L->E);
                if (kd <= 0) continue;
                f64 ku = kd * exp(-(L->E[v] - L->E[v - 1]) / (8.617333262e-5 * T));
                nm(v, a, 16); nm(v - 1, c, 16);
                if (partners[p].kind == VT_N2 || partners[p].kind == VT_AIR) {
                    // collider = all N2 (third body x the N2 mole fraction 0.79) or the whole gas
                    f64 xf = partners[p].kind == VT_N2 ? 0.79 : 1.0;
                    snprintf(eq, sizeof(eq), "%s + M -> %s + M", a, c);
                    kin_add_reaction(model, eq, kin_m(xf * kd));
                    snprintf(eq, sizeof(eq), "%s + M -> %s + M", c, a);
                    kin_add_reaction(model, eq, kin_m(xf * ku));
                } else {
                    snprintf(eq, sizeof(eq), "%s + %s -> %s + %s", a, partners[p].name, c, partners[p].name);
                    kin_add_reaction(model, eq, kin_arr(kd));
                    snprintf(eq, sizeof(eq), "%s + %s -> %s + %s", c, partners[p].name, a, partners[p].name);
                    kin_add_reaction(model, eq, kin_arr(ku));
                }
            }
        // Zeldovich, state-specific (v = 0 is already "N2 + O -> NO + N" thermal: not in the mechanism
        // at these temperatures, so every level including 0 is added here)
        for (u32 v = 0; v <= VMAX; ++v) {
            f64 k = k_zeldovich(v, T, L->E);
            if (k <= 0) continue;
            nm(v, a, 16);
            snprintf(eq, sizeof(eq), "%s + O -> NO + N", a);
            kin_add_reaction(model, eq, kin_arr(k));
        }
    }
    kin_mech_finalize(model);
}

// e-V deltas for one pulse (per cm^3) from the current populations: from
// level v to v + dv with G(0 -> dv) x (n_v / n_N2total) (Fridman scaling).
internal void ev_pulse(Ladder *L, const f64 *y, f64 energy_J_cm3, f64 *delta) {
    f64 n_tot = 0;
    for (u32 v = 0; v <= VMAX; ++v) n_tot += y[L->level[v]];
    if (n_tot <= 0) return;
    f64 hundreds = energy_J_cm3 * EV_PER_JOULE * 0.01;
    for (u32 v = 0; v < VMAX; ++v) {
        f64 share = y[L->level[v]] / n_tot * ev_scaling(v);
        for (u32 dv = 1; dv <= 10 && v + dv <= VMAX; ++dv) {
            f64 d = L->g_ev[dv] * share * hundreds;
            delta[L->level[v]] -= d;
            delta[L->level[v + dv]] += d;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "usage: %s T_K x_H2O E_f_mJ_cm3 P_W_cm3 residence_s EN_Td [novib] [check]\n", argv[0]); return 1; }
    f64 T = atof(argv[1]), xh2o = atof(argv[2]), ef = atof(argv[3]) * 1e-3, P = atof(argv[4]), res = atof(argv[5]), en = atof(argv[6]);
    b32 vib = 1, check = 0;
    for (int i = 7; i < argc; ++i) {
        if (!strcmp(argv[i], "novib")) vib = 0;
        if (!strcmp(argv[i], "check")) check = 1;
        if (!strcmp(argv[i], "step13")) g_zeldovich_model = ZEL_STEP13;
        if (!strcmp(argv[i], "step11")) g_zeldovich_model = ZEL_STEP11;
        if (!strcmp(argv[i], "noair")) g_no_air_relax = 1;   // sensitivity: drop the ISO dry-air channel
    }
    const char *zname[] = { "Fridman-Macheret", "step 1e-13", "step 1e-11" };
    Arena arena = arena_make(g_mem, sizeof(g_mem));
    KinModel *model = push_struct(&arena, KinModel);
    AirSpecies sp;
    static Ladder L;
    build(model, &sp, &L, T, vib);
    if (model->error) { fprintf(stderr, "mechanism: %s\n", model->error_text); return 1; }
    char csv[512];
    snprintf(csv, sizeof(csv), "%s/ev_gvalues.csv", N2VIB_DIR);
    read_ev_gvalues(csv, en, L.g_ev);
    fprintf(stderr, "mechanism: %u species, %u reactions, nnz %u\n", model->mech.species_count, model->mech.reaction_count, model->mech.nnz);

    AirSourceModel src = air_source_defaults();
    src.boltzmann = 1; src.reduced_field_Td = en; src.filament_J_cm3 = ef;
    KinMethod method = kin_method(KinMethod_Rodas3);
    KinControl ctl = dbd_default_control();
    ctl.max_steps = 2000000;
    DbdZones z;
    dbd_zones_alloc(&z, &arena, model, 1);
    z.T_K[0] = T; z.p_Pa[0] = 101325.0; z.x_H2O[0] = xh2o; z.power_W_cm3[0] = check ? 0.0 : P; z.freq_Hz[0] = 6500.0;
    dbd_zones_fill_padding(&z);
    dbd_zones_reset(&z, &sp);
    // ladder at thermal equilibrium (Boltzmann at T) out of the feed's N2
    if (vib) {
        u32 pz = z.padded;
        f64 n2 = z.y[sp.N2 * pz], zsum = 0;
        for (u32 v = 0; v <= VMAX; ++v) zsum += exp(-L.E[v] / (8.617333262e-5 * T));
        for (u32 v = 0; v <= VMAX; ++v) z.y[L.level[v] * pz] = n2 * exp(-L.E[v] / (8.617333262e-5 * T)) / zsum;
        for (u32 s = 0; s < z.n_species; ++s) for (u32 i = 1; i < pz; ++i) z.y[s * pz + i] = z.y[s * pz];
    }
    dbd_zones_prepare_rates(&z, model);
    f64 y0[KIN_MAX_SPECIES];
    for (u32 s = 0; s < z.n_species; ++s) y0[s] = z.y[s * z.padded];

    f64 dt = 1e-3, vib_J = 0;
    u32 steps = (u32)ceil(res / dt);
    for (u32 i = 0; i < steps; ++i) {
        f64 a = i * dt, b = Minimum((i + 1) * dt, res);
        dbd_zones_prepare_source(&z, &sp, &src, DbdDischarge_Pulsed, a, b);
        if (vib && z.pulse_count[0] > 0) {
            f64 y[KIN_MAX_SPECIES], d[KIN_MAX_SPECIES] = {};
            for (u32 s = 0; s < z.n_species; ++s) y[s] = z.y[s * z.padded];
            ev_pulse(&L, y, z.power_W_cm3[0] / (z.period[0] > 0 ? 1.0 / z.period[0] : 1.0), d);
            for (u32 s = 0; s < z.n_species; ++s) z.pulse[s * z.padded] += d[s];
        }
        dbd_advance_cpu<f64x4>(&z, &model->mech, &method, &ctl, b - a);
        dbd_zones_commit_pulses(&z);
        if (z.failed[0] != 0.0) { fprintf(stderr, "solver failed at t = %.4f s\n", b); return 1; }
    }
    // results
    u32 pz = z.padded;
    f64 tot = 0;
    for (u32 s = 0; s < z.n_species; ++s) tot += z.y[s * pz];
    auto ppm = [&](u32 s) { return z.y[s * pz] / tot * 1e6; };
    f64 fixedN = ppm(sp.NO) + ppm(sp.NO2) + ppm(sp.NO3) + 2 * ppm(sp.N2O5) + ppm(sp.HNO2) + ppm(sp.HNO3);
    f64 E_in = P * res;
    f64 cost = fixedN > 0 ? E_in / (fixedN * 1e-6 * tot / 6.02214076e23) * 1e-6 : 0;
    printf("T %.0f K, H2O %.0f ppm, E_f %.0f mJ/cm3, %.2f J/cm3, %s: O3 %.0f ppm, NO %.1f, NO2 %.1f, N2O5 %.1f, HNO3 %.1f, N2O %.1f | fixed N %.1f ppm, %.1f MJ/mol | %.0f pulses\n",
           T, xh2o * 1e6, ef * 1e3, E_in, vib ? zname[g_zeldovich_model] : "no ladder", ppm(sp.O3), ppm(sp.NO), ppm(sp.NO2), ppm(sp.N2O5),
           ppm(sp.HNO3), ppm(sp.N2O), fixedN, cost, z.pulses[0]);
    if (vib) {
        f64 n2 = 0, ev = 0;
        for (u32 v = 0; v <= VMAX; ++v) { n2 += z.y[L.level[v] * pz]; ev += z.y[L.level[v] * pz] * L.E[v]; }
        f64 r10 = z.y[L.level[1] * pz] / z.y[L.level[0] * pz];
        f64 Tv = r10 > 0 ? L.E[1] / (8.617333262e-5 * log(1.0 / r10)) : 0;
        printf("  vibrational: T_v(0,1) %.0f K, mean %.4f eV per N2; populations v=0..%u:", Tv, ev / n2, VMAX);
        for (u32 v = 0; v <= VMAX; v += 2) printf(" %.2e", z.y[L.level[v] * pz] / n2);
        printf("\n");
        if (check) {
            f64 worst = 0;
            for (u32 s = 0; s < z.n_species; ++s) if (y0[s] > 1e3) worst = Maximum(worst, fabs(z.y[s * pz] - y0[s]) / y0[s]);
            printf("  check (no discharge, ladder at Boltzmann T): worst relative change of any species %.2e\n", worst);
        }
    }
    return 0;
}
