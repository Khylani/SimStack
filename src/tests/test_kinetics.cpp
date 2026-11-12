// Kinetics solver tests: one TU, no framework. Built by `./build.sh test`
// into build/simstack_tests; exit code is the number of failed checks.
//
//   ./build/simstack_tests            all CPU tests
//   ./build/simstack_tests --gpu      also compare the CUDA module (build/simstack_gpu.so)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dlfcn.h>
#include <time.h>

#define DEBUG_RECORD_BASE 0
#include "../base/base_types.h"
#include "../base/base_arena.h"
#include "../base/base_simd.h"
#include "../base/base_debug.h"
#include "../sim/dbd.cpp"
#include "../gpu/gpu_api.h"

global u32 g_checks, g_failures;

#define CHECK(cond, ...) do { ++g_checks; if (!(cond)) { ++g_failures; printf("  FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

internal f64 now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (f64)ts.tv_sec + (f64)ts.tv_nsec * 1e-9;
}

internal f64 rel_err(f64 a, f64 b) {
    f64 d = fabs(a - b), s = fmax(fabs(a), fabs(b));
    return s > 0.0 ? d / s : d;
}

global u8 g_arena_memory[Megabytes(64)] SIM_ALIGNED(64);

// ---------------------------------------------------------------------------

internal void test_mechanism(KinModel *model) {
    printf("[mechanism] %u species, %u reactions, nnz %u (of %u), lu %u div / %u sub ops\n",
           model->mech.species_count, model->mech.reaction_count, model->mech.nnz,
           model->mech.species_count * model->mech.species_count, model->mech.lu_div_count, model->mech.lu_sub_count);
    CHECK(!model->error, "mechanism error: %s", model->error_text);

    // The builder must reject an unbalanced reaction.
    KinModel *bad = (KinModel *)calloc(1, sizeof(KinModel));
    kin_add_species(bad, "O");
    kin_add_species(bad, "O2");
    kin_add_species(bad, "O3");
    kin_add_reaction(bad, "O + O2 -> O2 + O2", kin_arr(1.0));
    CHECK(bad->error, "unbalanced reaction accepted");
    free(bad);
}

// Analytic sparse Jacobian vs central finite differences of kin_fun.
internal void test_jacobian(KinModel *model, AirSpecies *sp) {
    KinMech *m = &model->mech;
    u32 n = m->species_count;
    f64 y[KIN_MAX_SPECIES], k[KIN_MAX_REACTIONS], src[KIN_MAX_SPECIES] = {};
    f64 total = air_feed(sp, n, 320.0, 101325.0, 0.02, y);
    for (u32 s = 0; s < n; ++s) if (y[s] == 0.0) y[s] = total * 1e-6 * (1.0 + 0.37 * (f64)s);   // all radicals present
    kin_eval_rate_constants(model, 320.0, total, k);

    f64 jac[KIN_MAX_NNZ];
    kin_jac(m, k, y, jac);
    // Row scale: sum of |production| + |loss| terms of species i. Central
    // differences of a polynomial are exact up to roundoff, and roundoff in
    // f_i is ~eps * row_scale_i, so compare J_ij * y_j against that scale.
    f64 row_scale[KIN_MAX_SPECIES] = {};
    for (u32 r = 0; r < m->reaction_count; ++r) {
        const KinReaction *rx = m->reactions + r;
        f64 rate = k[r];
        for (u32 p = 0; p < rx->reactant_count; ++p) rate *= y[rx->reactant[p]];
        for (u32 e = 0; e < rx->net_count; ++e) row_scale[rx->net_species[e]] += fabs(rx->net_coef[e] * rate);
    }
    f64 worst = 0.0;
    u32 wi = 0, wj = 0;
    for (u32 j = 0; j < n; ++j) {
        f64 yp[KIN_MAX_SPECIES], ym[KIN_MAX_SPECIES], fp[KIN_MAX_SPECIES], fm[KIN_MAX_SPECIES];
        for (u32 s = 0; s < n; ++s) { yp[s] = y[s]; ym[s] = y[s]; }
        f64 dy = y[j] * 1e-4;
        yp[j] += dy; ym[j] -= dy;
        kin_fun(m, k, src, yp, fp);
        kin_fun(m, k, src, ym, fm);
        for (u32 i = 0; i < n; ++i) {
            f64 fd = (fp[i] - fm[i]) / (2.0 * dy);
            f64 an = 0.0;
            if (i == j) an = jac[m->diag[i]];
            for (u32 e = m->lower_begin[i]; e < m->lower_begin[i + 1]; ++e) if (m->lower[e].col == j) an = jac[m->lower[e].index];
            for (u32 e = m->upper_begin[i]; e < m->upper_begin[i + 1]; ++e) if (m->upper[e].col == j) an = jac[m->upper[e].index];
            f64 e = fabs(an - fd) * y[j] / (row_scale[i] + 1e-300);
            if (e > worst) { worst = e; wi = i; wj = j; }
        }
    }
    printf("[jacobian] worst |J - FD| * y_j / row scale = %.2e at d(%s)/d(%s)\n", worst, model->species_name[wi], model->species_name[wj]);
    CHECK(worst < 1e-8, "jacobian mismatch %.3e", worst);
}

// Sparse no-pivot LU solve vs dense partial-pivot Gaussian elimination.
internal void test_lu(KinModel *model, AirSpecies *sp) {
    KinMech *m = &model->mech;
    u32 n = m->species_count;
    f64 y[KIN_MAX_SPECIES], k[KIN_MAX_REACTIONS];
    f64 total = air_feed(sp, n, 300.0, 101325.0, 0.01, y);
    for (u32 s = 0; s < n; ++s) if (y[s] == 0.0) y[s] = total * 1e-7;
    kin_eval_rate_constants(model, 300.0, total, k);
    f64 jac[KIN_MAX_NNZ], lu[KIN_MAX_NNZ];
    kin_jac(m, k, y, jac);
    f64 worst = 0.0;
    f64 hs[] = { 1e-12, 1e-9, 1e-6, 1e-3, 1.0 };
    for (u32 hi = 0; hi < ArrayCount(hs); ++hi) {
        f64 ghinv = 1.0 / (hs[hi] * 0.5);
        kin_decomp(m, jac, ghinv, lu);
        // dense copy of I*ghinv - J
        f64 A[KIN_MAX_SPECIES][KIN_MAX_SPECIES + 1] = {};
        for (u32 i = 0; i < n; ++i) {
            A[i][i] = ghinv - jac[m->diag[i]];
            for (u32 e = m->lower_begin[i]; e < m->lower_begin[i + 1]; ++e) A[i][m->lower[e].col] = -jac[m->lower[e].index];
            for (u32 e = m->upper_begin[i]; e < m->upper_begin[i + 1]; ++e) A[i][m->upper[e].col] = -jac[m->upper[e].index];
        }
        f64 b[KIN_MAX_SPECIES], x[KIN_MAX_SPECIES];
        for (u32 i = 0; i < n; ++i) { b[i] = 1.0 + (f64)i * 0.25; x[i] = b[i]; A[i][n] = b[i]; }
        kin_solve(m, lu, x);
        for (u32 c = 0; c < n; ++c) {   // dense GE with partial pivoting
            u32 piv = c;
            for (u32 r = c + 1; r < n; ++r) if (fabs(A[r][c]) > fabs(A[piv][c])) piv = r;
            for (u32 q = 0; q <= n; ++q) { f64 t = A[c][q]; A[c][q] = A[piv][q]; A[piv][q] = t; }
            for (u32 r = c + 1; r < n; ++r) {
                f64 f = A[r][c] / A[c][c];
                for (u32 q = c; q <= n; ++q) A[r][q] -= f * A[c][q];
            }
        }
        f64 xr[KIN_MAX_SPECIES];
        for (u32 i = n; i-- > 0;) {
            f64 acc = A[i][n];
            for (u32 q = i + 1; q < n; ++q) acc -= A[i][q] * xr[q];
            xr[i] = acc / A[i][i];
        }
        for (u32 i = 0; i < n; ++i) worst = fmax(worst, rel_err(x[i], xr[i]));
    }
    printf("[lu] sparse no-pivot vs dense pivoted solve: worst relative difference %.2e\n", worst);
    CHECK(worst < 1e-9, "lu mismatch %.3e", worst);
}

// Robertson's stiff problem (Hairer & Wanner II): rates span 9 orders.
internal void test_robertson() {
    KinModel *model = (KinModel *)calloc(1, sizeof(KinModel));
    kin_add_species(model, "A", 1);
    kin_add_species(model, "B", 1);
    kin_add_species(model, "C", 1);
    kin_add_reaction(model, "A -> B", kin_arr(0.04));
    kin_add_reaction(model, "B + B -> C + B", kin_arr(3.0e7));
    kin_add_reaction(model, "B + C -> A + C", kin_arr(1.0e4));
    kin_mech_finalize(model);
    CHECK(!model->error, "robertson model: %s", model->error_text);
    f64 k[3];
    kin_eval_rate_constants(model, 300.0, 1.0, k);

    const f64 ref[3] = { 0.7158270687193267, 9.185534764557338e-06, 0.2841637457458339 };   // t = 40
    for (u32 mi = 0; mi < KinMethod_Count; ++mi) {
        KinMethod method = kin_method((KinMethodId)mi);
        KinControl ctl = {};
        ctl.rtol = 1e-6; ctl.atol = 1e-12; ctl.h_min = 0.0; ctl.h_start = 1e-6; ctl.max_steps = 100000;
        f64 y[3] = { 1.0, 0.0, 0.0 }, src[3] = {}, t = 0.0, h = 0.0;
        KinLaneStats<f64> st = {};
        kin_integrate(&model->mech, &method, &ctl, k, src, y, t, 40.0, h, st);
        f64 e0 = rel_err(y[0], ref[0]), e1 = rel_err(y[1], ref[1]), e2 = rel_err(y[2], ref[2]);
        printf("[robertson %-6s] t=40: %.10f %.10e %.10f  rel err %.1e %.1e %.1e, %.0f steps %.0f rejects\n",
               method.name, y[0], y[1], y[2], e0, e1, e2, st.steps, st.rejects);
        CHECK(t == 40.0 && st.failed == 0.0, "%s did not finish", method.name);
        CHECK(e0 < 1e-4 && e1 < 1e-4 && e2 < 1e-4, "%s robertson inaccurate", method.name);
        CHECK(fabs(y[0] + y[1] + y[2] - 1.0) < 1e-12, "%s mass not conserved: %.3e", method.name, y[0] + y[1] + y[2] - 1.0);
    }
    free(model);
}

// Fixed-step convergence order on A + B -> C, which has a closed form.
internal void test_order() {
    KinModel *model = (KinModel *)calloc(1, sizeof(KinModel));
    kin_add_species(model, "A", 1);
    kin_add_species(model, "B", 1);
    kin_add_species(model, "C", 1);
    kin_add_reaction(model, "A + B -> C", kin_arr(1.0));
    kin_mech_finalize(model);
    f64 k[1] = { 1.0 };
    const f64 A0 = 2.0, B0 = 1.0, T = 2.0, d = A0 - B0;
    f64 exact = d / (1.0 - (B0 / A0) * exp(-d * T));
    for (u32 mi = 0; mi < KinMethod_Count; ++mi) {
        KinMethod method = kin_method((KinMethodId)mi);
        f64 errs[4];
        u32 steps[4] = { 10, 20, 40, 80 };
        for (u32 q = 0; q < 4; ++q) {
            KinControl ctl = {};
            ctl.rtol = 1.0; ctl.atol = 1.0; ctl.fixed_step = 1; ctl.max_steps = 1000;
            f64 y[3] = { A0, B0, 0.0 }, src[3] = {}, t = 0.0, h = T / (f64)steps[q];
            KinLaneStats<f64> st = {};
            kin_integrate(&model->mech, &method, &ctl, k, src, y, t, T, h, st);
            errs[q] = fabs(y[0] - exact);
        }
        f64 order = log2(errs[2] / errs[3]);
        printf("[order %-6s] errors %.2e %.2e %.2e %.2e  observed order %.2f (expected %u)\n", method.name,
               errs[0], errs[1], errs[2], errs[3], order, method.order);
        CHECK(fabs(order - (f64)method.order) < 0.3, "%s order %.2f", method.name, order);
    }
    free(model);
}

// A small set of zones with varied operating points.
internal void setup_zones(DbdZones *z, Arena *arena, KinModel *model, AirSpecies *sp, AirSourceModel *src,
                          u32 count, DbdDischargeMode mode) {
    dbd_zones_alloc(z, arena, model, count);
    for (u32 i = 0; i < count; ++i) {
        z->T_K[i] = 300.0 + 10.0 * (f64)(i % 5);
        z->p_Pa[i] = 101325.0;
        z->x_H2O[i] = 0.005 * (f64)(i % 7);
        z->power_W_cm3[i] = 0.05 + 0.1 * (f64)(i % 3);
        z->freq_Hz[i] = 5000.0 + 2500.0 * (f64)(i % 4);
    }
    dbd_zones_fill_padding(z);
    dbd_zones_reset(z, sp);
    dbd_zones_prepare(z, model, sp, src, mode);
}

// AVX2 lanes must reproduce the scalar reference bit for bit.
internal void test_simd_matches_scalar(KinModel *model, AirSpecies *sp, AirSourceModel *src) {
    Arena arena = arena_make(g_arena_memory, sizeof(g_arena_memory));
    KinMethod method = kin_method(KinMethod_Rodas3);
    KinControl ctl = dbd_default_control();
    for (u32 mode = 0; mode < 2; ++mode) {
        DbdZones a, b;
        setup_zones(&a, &arena, model, sp, src, 12, (DbdDischargeMode)mode);
        setup_zones(&b, &arena, model, sp, src, 12, (DbdDischargeMode)mode);
        for (u32 step = 0; step < 5; ++step) {
            f64 t0 = step * 1e-3, t1 = (step + 1) * 1e-3;
            dbd_zones_prepare(&a, model, sp, src, (DbdDischargeMode)mode);
            dbd_zones_prepare(&b, model, sp, src, (DbdDischargeMode)mode);
            dbd_advance_cpu<f64x4>(&a, &model->mech, &method, &ctl, t0, t1);
            dbd_advance_cpu<f64>(&b, &model->mech, &method, &ctl, t0, t1);
        }
        u64 bytes = sizeof(f64) * (u64)a.n_species * a.padded;
        b32 same = memcmp(a.y, b.y, bytes) == 0 && memcmp(a.h, b.h, sizeof(f64) * a.padded) == 0 &&
                   memcmp(a.steps, b.steps, sizeof(f64) * a.padded) == 0;
        printf("[simd] %s, 12 zones x 5 ms: f64x4 vs f64 %s (hash %016llx, zone 0: %.0f steps, %.0f pulses)\n",
               mode == 0 ? "pulsed  " : "averaged", same ? "bit-identical" : "DIFFERENT",
               (unsigned long long)hash_fnv1a_64(a.y, bytes), a.steps[0], a.pulses[0]);
        CHECK(same, "simd and scalar paths differ (mode %u)", mode);
    }
}

// Atoms of N, O, H are conserved by chemistry and by the discharge source.
internal void test_conservation(KinModel *model, AirSpecies *sp, AirSourceModel *src) {
    Arena arena = arena_make(g_arena_memory, sizeof(g_arena_memory));
    KinMethod method = kin_method(KinMethod_Rodas3);
    KinControl ctl = dbd_default_control();
    DbdZones z;
    setup_zones(&z, &arena, model, sp, src, 8, DbdDischarge_Pulsed);
    f64 before[8][KIN_ELEMENTS];
    for (u32 i = 0; i < 8; ++i) {
        f64 y[KIN_MAX_SPECIES];
        for (u32 s = 0; s < z.n_species; ++s) y[s] = z.y[s * z.padded + i];
        kin_count_elements(model, y, before[i]);
    }
    for (u32 step = 0; step < 50; ++step) {
        dbd_zones_prepare(&z, model, sp, src, DbdDischarge_Pulsed);
        dbd_advance_cpu<f64x4>(&z, &model->mech, &method, &ctl, step * 1e-3, (step + 1) * 1e-3);
    }
    f64 worst = 0.0;
    for (u32 i = 0; i < 8; ++i) {
        f64 y[KIN_MAX_SPECIES], after[KIN_ELEMENTS];
        for (u32 s = 0; s < z.n_species; ++s) y[s] = z.y[s * z.padded + i];
        kin_count_elements(model, y, after);
        for (u32 e = 0; e < KIN_ELEMENTS; ++e) if (before[i][e] > 0.0) worst = fmax(worst, rel_err(before[i][e], after[e]));
    }
    printf("[conservation] 8 zones x 50 ms pulsed: worst relative N/O/H atom drift %.2e; zone 0 O3 %.1f ppm, NO2 %.2f ppm, HNO3 %.3f ppm\n",
           worst, dbd_ppm(&z, sp->O3, 0), dbd_ppm(&z, sp->NO2, 0), dbd_ppm(&z, sp->HNO3, 0));
    CHECK(worst < 1e-10, "atom drift %.3e", worst);
    for (u32 i = 0; i < 8; ++i) CHECK(z.failed[i] == 0.0, "zone %u failed", i);
}

// Physics sanity: at low energy density in dry air nearly every O atom made by
// the discharge ends up as O3 (O + O2 + M beats every other O sink).
internal void test_ozone_yield(KinModel *model, AirSpecies *sp, AirSourceModel *src) {
    Arena arena = arena_make(g_arena_memory, sizeof(g_arena_memory));
    KinMethod method = kin_method(KinMethod_Rodas3);
    KinControl ctl = dbd_default_control();
    DbdZones z;
    dbd_zones_alloc(&z, &arena, model, 1);
    z.T_K[0] = 300.0; z.p_Pa[0] = 101325.0; z.x_H2O[0] = 0.0; z.power_W_cm3[0] = 0.01; z.freq_Hz[0] = 10000.0;
    dbd_zones_fill_padding(&z);
    dbd_zones_reset(&z, sp);
    for (u32 step = 0; step < 10; ++step) {
        dbd_zones_prepare(&z, model, sp, src, DbdDischarge_Pulsed);
        dbd_advance_cpu<f64x4>(&z, &model->mech, &method, &ctl, step * 1e-3, (step + 1) * 1e-3);
    }
    f64 energy = 0.01 * 10e-3;   // J/cm^3
    // Odd oxygen made: O from O2 dissociation, plus one O per N atom
    // (N + O2 -> NO + O, N + NO -> N2 + O). Dry air, so x_O2/x_N2 are the reference values.
    f64 o_made = (src->g_O_per_100eV + src->g_N_per_100eV) * energy * EV_PER_JOULE * 0.01;
    f64 ratio = z.y[sp->O3 * z.padded] / o_made;
    printf("[ozone] dry air, 0.1 mJ/cm^3: O3 / (O + N produced) = %.3f (%.0f pulses, %.0f steps)\n", ratio, z.pulses[0], z.steps[0]);
    CHECK(ratio > 0.8 && ratio < 1.0, "ozone yield %.3f", ratio);
    CHECK(z.pulses[0] == 200.0, "expected 200 pulses, got %.0f", z.pulses[0]);
}

internal void bench_cpu(KinModel *model, AirSpecies *sp, AirSourceModel *src) {
    Arena arena = arena_make(g_arena_memory, sizeof(g_arena_memory));
    KinMethod method = kin_method(KinMethod_Rodas3);
    KinControl ctl = dbd_default_control();
    for (u32 mode = 0; mode < 2; ++mode) {
        for (u32 simd = 0; simd < 2; ++simd) {
            arena.used = 0;
            DbdZones z;
            setup_zones(&z, &arena, model, sp, src, 64, (DbdDischargeMode)mode);
            f64 t_start = now_s();
            u32 ms = 20;
            for (u32 step = 0; step < ms; ++step) {
                dbd_zones_prepare(&z, model, sp, src, (DbdDischargeMode)mode);
                if (simd) dbd_advance_cpu<f64x4>(&z, &model->mech, &method, &ctl, step * 1e-3, (step + 1) * 1e-3);
                else dbd_advance_cpu<f64>(&z, &model->mech, &method, &ctl, step * 1e-3, (step + 1) * 1e-3);
            }
            f64 wall = now_s() - t_start;
            f64 steps = 0.0;
            for (u32 i = 0; i < z.count; ++i) steps += z.steps[i];
            printf("[bench] %s %-6s 64 zones x %u ms: %.1f ms wall, %.2f zone-ms per wall-ms, %.1f us per zone-step\n",
                   mode == 0 ? "pulsed  " : "averaged", simd ? "f64x4" : "f64", ms, wall * 1e3,
                   64.0 * ms / (wall * 1e3), wall * 1e6 / steps);
        }
    }
}

// ---------------------------------------------------------------------------

internal void test_gpu(KinModel *model, AirSpecies *sp, AirSourceModel *src);

int main(int argc, char **argv) {
    b32 want_gpu = 0, want_bench = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--gpu") == 0) want_gpu = 1;
        if (strcmp(argv[i], "--bench") == 0) want_bench = 1;
    }
    KinModel *model = (KinModel *)calloc(1, sizeof(KinModel));
    AirSpecies sp;
    air_mech_build(model, &sp);
    AirSourceModel src = air_source_defaults();

    test_mechanism(model);
    test_jacobian(model, &sp);
    test_lu(model, &sp);
    test_robertson();
    test_order();
    test_simd_matches_scalar(model, &sp, &src);
    test_conservation(model, &sp, &src);
    test_ozone_yield(model, &sp, &src);
    if (want_gpu) test_gpu(model, &sp, &src);
    if (want_bench) bench_cpu(model, &sp, &src);

    printf("%u checks, %u failed (simd=%d)\n", g_checks, g_failures, SIM_SIMD);
    free(model);
    return (int)g_failures;
}

internal void test_gpu(KinModel *model, AirSpecies *sp, AirSourceModel *src) {
    printf("[gpu] not wired yet\n");
}
