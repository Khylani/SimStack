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
#include <unistd.h>

#include <pthread.h>
#include <semaphore.h>
#include <sched.h>

#define DEBUG_RECORD_BASE 0
#include "../sim_platform.h"
#include "../sim/dbd.cpp"
#include "../sim/nft.cpp"
#include "../platform/linux_gpu.cpp"
#include "../platform/linux_work_queue.cpp"

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

global u8 g_arena_memory[Megabytes(256)] SIM_ALIGNED(64);

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
    dbd_zones_prepare_rates(z, model);
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
            dbd_step_cpu<f64x4>(&a, model, sp, src, (DbdDischargeMode)mode, &method, &ctl, t0, t1);
            dbd_step_cpu<f64>(&b, model, sp, src, (DbdDischargeMode)mode, &method, &ctl, t0, t1);
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

// Generated straight-line kernels must match the table walk bit for bit.
internal void test_generated_matches_table(KinModel *model, AirSpecies *sp, AirSourceModel *src) {
    CHECK(dbd_generated_matches(model), "generated kernels are stale (signature mismatch): rebuild");
    Arena arena = arena_make(g_arena_memory, sizeof(g_arena_memory));
    KinMethod method = kin_method(KinMethod_Rodas3);
    KinControl ctl = dbd_default_control();
    for (u32 mode = 0; mode < 2; ++mode) {
        DbdZones a, b;
        setup_zones(&a, &arena, model, sp, src, 12, (DbdDischargeMode)mode);
        setup_zones(&b, &arena, model, sp, src, 12, (DbdDischargeMode)mode);
        for (u32 step = 0; step < 5; ++step) {
            f64 t0 = step * 1e-3, t1 = (step + 1) * 1e-3;
            dbd_step_cpu<f64x4, AirGenOps>(&a, model, sp, src, (DbdDischargeMode)mode, &method, &ctl, t0, t1);
            dbd_step_cpu<f64x4>(&b, model, sp, src, (DbdDischargeMode)mode, &method, &ctl, t0, t1);
        }
        u64 bytes = sizeof(f64) * (u64)a.n_species * a.padded;
        b32 same = memcmp(a.y, b.y, bytes) == 0 && memcmp(a.steps, b.steps, sizeof(f64) * a.padded) == 0;
        printf("[codegen] %s: generated vs table kernels %s\n", mode == 0 ? "pulsed  " : "averaged",
               same ? "bit-identical" : "DIFFERENT");
        CHECK(same, "generated and table kernels differ (mode %u)", mode);
    }
}

// f32 lanes (delta formulation): f32x8 == f32 scalar bit for bit; accuracy of
// the products and atom conservation against the f64 path over a long run.
internal void test_f32(KinModel *model, AirSpecies *sp, AirSourceModel *src) {
    Arena arena = arena_make(g_arena_memory, sizeof(g_arena_memory));
    KinMethod method = kin_method(KinMethod_Rodas3);
    KinControl ctl = dbd_default_control();
    u32 watch[] = { sp->O3, sp->NO, sp->NO2, sp->N2O5, sp->HNO2, sp->HNO3, sp->H2O2, sp->N2O };
    for (u32 mode = 0; mode < 2; ++mode) {
        DbdZones a, b, c;
        setup_zones(&a, &arena, model, sp, src, 16, (DbdDischargeMode)mode);
        setup_zones(&b, &arena, model, sp, src, 16, (DbdDischargeMode)mode);
        setup_zones(&c, &arena, model, sp, src, 16, (DbdDischargeMode)mode);
        f64 before[16][KIN_ELEMENTS];
        for (u32 i = 0; i < 16; ++i) {
            f64 y[KIN_MAX_SPECIES];
            for (u32 s = 0; s < a.n_species; ++s) y[s] = a.y[s * a.padded + i];
            kin_count_elements(model, y, before[i]);
        }
        u32 ms = mode == 0 ? 100 : 1000;
        f64 t32 = 0.0, t64 = 0.0;
        for (u32 step = 0; step < ms; ++step) {
            f64 t0 = step * 1e-3, t1 = (step + 1) * 1e-3;
            f64 s0 = now_s();
            dbd_step_cpu<f32x8, AirGenOps, true>(&a, model, sp, src, (DbdDischargeMode)mode, &method, &ctl, t0, t1);
            f64 s1 = now_s();
            dbd_step_cpu<f64x4, AirGenOps>(&c, model, sp, src, (DbdDischargeMode)mode, &method, &ctl, t0, t1);
            f64 s2 = now_s();
            if (step < 5) dbd_step_cpu<f32, AirGenOps, true>(&b, model, sp, src, (DbdDischargeMode)mode, &method, &ctl, t0, t1);
            if (step == 4) {
                b32 same = memcmp(a.y, b.y, sizeof(f64) * (u64)a.n_species * a.padded) == 0;
                printf("[f32] %s: f32x8 vs f32 scalar after 5 ms %s\n", mode == 0 ? "pulsed  " : "averaged", same ? "bit-identical" : "DIFFERENT");
                CHECK(same, "f32x8 and f32 differ (mode %u)", mode);
            }
            t32 += s1 - s0; t64 += s2 - s1;
        }
        f64 worst = 0.0, drift = 0.0;
        u32 worst_s = 0;
        for (u32 i = 0; i < 16; ++i) {
            for (u32 w = 0; w < ArrayCount(watch); ++w) {
                if (dbd_ppm(&c, watch[w], i) < 1e-3) continue;
                f64 e = rel_err(a.y[watch[w] * a.padded + i], c.y[watch[w] * c.padded + i]);
                if (e > worst) { worst = e; worst_s = watch[w]; }
            }
            f64 y[KIN_MAX_SPECIES], after[KIN_ELEMENTS];
            for (u32 s = 0; s < a.n_species; ++s) y[s] = a.y[s * a.padded + i];
            kin_count_elements(model, y, after);
            for (u32 e = 0; e < KIN_ELEMENTS; ++e) if (before[i][e] > 0.0) drift = fmax(drift, rel_err(before[i][e], after[e]));
        }
        f64 steps32 = 0, steps64 = 0;
        for (u32 i = 0; i < 16; ++i) { steps32 += a.steps[i]; steps64 += c.steps[i]; }
        printf("[f32] %s 16 zones x %u ms: worst product error vs f64 %.1e (%s), atom drift %.1e (f64: ~1e-13), "
               "steps %.0f vs %.0f, wall %.1f ms vs %.1f ms (%.2fx)\n",
               mode == 0 ? "pulsed  " : "averaged", ms, worst, model->species_name[worst_s], drift, steps32, steps64,
               t32 * 1e3, t64 * 1e3, t64 / t32);
        CHECK(worst < 1e-2, "f32 product error %.3e", worst);
        // f32 floor: each master step rounds the increments to f32 (~6e-8 of the
        // increment), so drift grows ~linearly in time: bound it per simulated second.
        f64 drift_per_s = drift / (ms * 1e-3);
        CHECK(drift_per_s < 1e-7, "f32 atom drift %.3e per s", drift_per_s);
        for (u32 i = 0; i < 16; ++i) CHECK(a.failed[i] == 0.0, "f32 zone %u failed", i);
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
        dbd_step_cpu<f64x4, AirGenOps>(&z, model, sp, src, DbdDischarge_Pulsed, &method, &ctl, step * 1e-3, (step + 1) * 1e-3);
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
    dbd_zones_prepare_rates(&z, model);
    for (u32 step = 0; step < 10; ++step) {
        dbd_step_cpu<f64x4, AirGenOps>(&z, model, sp, src, DbdDischarge_Pulsed, &method, &ctl, step * 1e-3, (step + 1) * 1e-3);
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

// Cost of each solver kernel on one f64x4 lane group, in TSC cycles.
internal void bench_kernels(KinModel *model, AirSpecies *sp) {
    KinMech *m = &model->mech;
    u32 n = m->species_count;
    f64 y1[KIN_MAX_SPECIES], k1[KIN_MAX_REACTIONS];
    f64 total = air_feed(sp, n, 300.0, 101325.0, 0.01, y1);
    for (u32 s = 0; s < n; ++s) if (y1[s] == 0.0) y1[s] = total * 1e-7;
    kin_eval_rate_constants(model, 300.0, total, k1);
    f64x4 y[KIN_MAX_SPECIES], k[KIN_MAX_REACTIONS], src[KIN_MAX_SPECIES], f[KIN_MAX_SPECIES], x[KIN_MAX_SPECIES];
    for (u32 s = 0; s < n; ++s) { y[s] = lane_splat((f64x4 *)0, y1[s]); src[s] = lane_splat((f64x4 *)0, 0.0); x[s] = y[s]; }
    for (u32 r = 0; r < m->reaction_count; ++r) k[r] = lane_splat((f64x4 *)0, k1[r]);
    static f64x4 jac[KIN_MAX_NNZ], lu[KIN_MAX_NNZ];
    const u32 reps = 20000;
    u64 c0 = read_cpu_timer();
    for (u32 i = 0; i < reps; ++i) { kin_fun(m, k, src, y, f); y[0] = y[0] + f[1] * lane_splat((f64x4 *)0, 1e-300); }
    u64 c1 = read_cpu_timer();
    for (u32 i = 0; i < reps; ++i) { kin_jac(m, k, y, jac); y[0] = y[0] + jac[3] * lane_splat((f64x4 *)0, 1e-300); }
    u64 c2 = read_cpu_timer();
    for (u32 i = 0; i < reps; ++i) { kin_decomp(m, jac, lane_splat((f64x4 *)0, 1e6 + i), lu); }
    u64 c3 = read_cpu_timer();
    for (u32 i = 0; i < reps; ++i) { kin_solve(m, lu, x); }
    u64 c4 = read_cpu_timer();
    f64x4 acc = lane_splat((f64x4 *)0, 0.5);
    for (u32 i = 0; i < reps; ++i) { acc = acc + lane_cbrt_clamped(acc); }
    u64 c5 = read_cpu_timer();
    printf("[kernels] f64x4 table  TSC cycles per call: fun %.0f  jac %.0f  decomp %.0f  solve %.0f  cbrt %.0f  (sink %g)\n",
           (f64)(c1 - c0) / reps, (f64)(c2 - c1) / reps, (f64)(c3 - c2) / reps, (f64)(c4 - c3) / reps,
           (f64)(c5 - c4) / reps, lane_get(x[0], 0) + lane_get(acc, 1) + lane_get(lu[5], 2));
    u64 g0 = read_cpu_timer();
    for (u32 i = 0; i < reps; ++i) { AirGenOps::fun(m, k, src, y, f); y[0] = y[0] + f[1] * lane_splat((f64x4 *)0, 1e-300); }
    u64 g1 = read_cpu_timer();
    for (u32 i = 0; i < reps; ++i) { AirGenOps::neg_jac(m, k, y, lu); y[0] = y[0] + lu[3] * lane_splat((f64x4 *)0, 1e-300); }
    u64 g2 = read_cpu_timer();
    for (u32 i = 0; i < reps; ++i) { AirGenOps::factor(m, lane_splat((f64x4 *)0, 1e6 + i), lu); }
    u64 g3 = read_cpu_timer();
    for (u32 i = 0; i < reps; ++i) { AirGenOps::solve(m, lu, x); }
    u64 g4 = read_cpu_timer();
    printf("[kernels] f64x4 gen    TSC cycles per call: fun %.0f  jac %.0f  decomp %.0f  solve %.0f  (sink %g)\n",
           (f64)(g1 - g0) / reps, (f64)(g2 - g1) / reps, (f64)(g3 - g2) / reps, (f64)(g4 - g3) / reps,
           lane_get(x[0], 0) + lane_get(lu[5], 2));
}

// Accuracy vs cost: 8 synchronized humid-air zones, 20 ms pulsed, against a
// tight-tolerance reference. Error = worst relative error over the
// long-lived products that matter downstream (above 1e-6 ppm).
internal void bench_tolerance(KinModel *model, AirSpecies *sp, AirSourceModel *src) {
    Arena arena = arena_make(g_arena_memory, sizeof(g_arena_memory));
    u32 watch[] = { sp->O3, sp->NO, sp->NO2, sp->NO3, sp->N2O5, sp->N2O, sp->HNO2, sp->HNO3, sp->H2O2 };
    struct Run { KinMethodId method; f64 rtol, atol; } runs[] = {
        { KinMethod_Rodas3, 1e-9, 1e-3 },   // reference
        { KinMethod_Rodas3, 1e-5, 1e2 }, { KinMethod_Rodas3, 1e-4, 1e2 }, { KinMethod_Rodas3, 1e-3, 1e2 },
        { KinMethod_Rodas3, 1e-3, 1e4 }, { KinMethod_Rodas3, 1e-3, 1e6 }, { KinMethod_Rodas3, 1e-3, 1e8 },
        { KinMethod_Rodas3, 1e-3, 1e9 }, { KinMethod_Rodas3, 1e-2, 1e4 },
        { KinMethod_Ros3, 1e-4, 1e2 }, { KinMethod_Ros3, 1e-3, 1e4 },
        { KinMethod_Ros2, 1e-3, 1e4 },
    };
    f64 ref[KIN_MAX_SPECIES][8];
    for (u32 mode = 0; mode < 2; ++mode) {
        for (u32 ri = 0; ri < ArrayCount(runs); ++ri) {
            arena.used = 0;
            DbdZones z;
            dbd_zones_alloc(&z, &arena, model, 8);
            for (u32 i = 0; i < 8; ++i) {
                z.T_K[i] = 310.0; z.p_Pa[i] = 101325.0; z.x_H2O[i] = 0.02;
                z.power_W_cm3[i] = 0.05 * (f64)(i + 1); z.freq_Hz[i] = 10000.0;
            }
            dbd_zones_fill_padding(&z);
            dbd_zones_reset(&z, sp);
    dbd_zones_prepare_rates(&z, model);
            KinMethod method = kin_method(runs[ri].method);
            KinControl ctl = dbd_default_control();
            ctl.rtol = runs[ri].rtol; ctl.atol = runs[ri].atol; ctl.max_steps = 1000000;
            f64 t_start = now_s();
            for (u32 step = 0; step < 20; ++step) {
                dbd_step_cpu<f64x4, AirGenOps>(&z, model, sp, src, (DbdDischargeMode)mode, &method, &ctl, step * 1e-3, (step + 1) * 1e-3);
            }
            f64 wall = now_s() - t_start;
            f64 worst = 0.0, steps = 0.0;
            u32 worst_s = 0;
            for (u32 i = 0; i < 8; ++i) {
                steps += z.steps[i];
                for (u32 w = 0; w < ArrayCount(watch); ++w) {
                    u32 sp_i = watch[w];
                    f64 v = z.y[sp_i * z.padded + i];
                    if (ri == 0) { ref[sp_i][i] = v; continue; }
                    if (dbd_ppm(&z, sp_i, i) < 1e-6) continue;
                    f64 e = rel_err(v, ref[sp_i][i]);
                    if (e > worst) { worst = e; worst_s = sp_i; }
                }
            }
            printf("[tolerance] %s %-6s rtol %.0e atol %.0e: %7.0f steps/zone, %6.1f ms wall, worst error %.1e (%s)\n",
                   mode == 0 ? "pulsed  " : "averaged", method.name, runs[ri].rtol, runs[ri].atol, steps / 8.0,
                   wall * 1e3, worst, ri ? model->species_name[worst_s] : "reference");
        }
    }
}

// ===========================================================================
// NFT tests

global u8 g_nft_memory[Megabytes(256)] SIM_ALIGNED(64);

// Runs `windows` windows on one thread with lane type V (f32x8 or f32).
template <class V>
internal void nft_run(NftSystem *s, u32 windows) {
    for (u32 w = 0; w < windows; ++w) {
        nft_environment(s, s->t);
        f32 shift[NftF_Count];
        b32 rebase = nft_inlet(s, shift);
        f32 gain = nft_light_gain(s);
        for (u32 g = 0; g < s->groups; ++g) nft_group_window<V>(s, g, s->ref, shift, rebase, gain);
        nft_reservoir_window(s);
        s->t += s->window_s;
    }
}

internal f64 nft_total_ion_mg(NftSystem *s, u32 i) {
    return s->res.mass_mg[NFT_ION0 + i] + nft_rail_mass_mg(s, NFT_ION0 + i);
}

internal void test_nft() {
    Arena arena = arena_make(g_nft_memory, sizeof(g_nft_memory));
    NftParams p = nft_default_params();
    p.rails = 12;   // 2 groups, the second half-padded

    // 1. f32x8 == f32 per lane, bit for bit
    NftSystem *a = push_struct(&arena, NftSystem), *b = push_struct(&arena, NftSystem);
    nft_init(a, &arena, p);
    nft_init(b, &arena, p);
    nft_run<f32x8>(a, 600);
    nft_run<f32>(b, 600);
    b32 same = memcmp(&a->res, &b->res, sizeof(a->res)) == 0;
    for (u32 g = 0; g < a->groups; ++g) {
        same &= memcmp(a->group[g].rows, b->group[g].rows, sizeof(NftRow) * a->cells) == 0;
        same &= memcmp(a->group[g].sites, b->group[g].sites, sizeof(NftSiteRow) * a->sites) == 0;
    }
    printf("[nft] %u channels x %u cells (%u plants), 10 h: f32x8 vs f32 lanes %s\n", a->rails, a->cells, a->sites,
           same ? "bit-identical" : "DIFFERENT");
    CHECK(same, "nft f32x8 and f32 lanes differ");

    // 2. mass balance per ion and water balance over 30 days with dosing, top-off and a solution change
    arena.used = 0;
    NftParams pm = p;
    pm.dump_days = 10.0f;
    NftSystem *m = push_struct(&arena, NftSystem);
    nft_init(m, &arena, pm);
    f64 initial[NFT_IONS], v0 = m->res.V_L;
    for (u32 i = 0; i < NFT_IONS; ++i) initial[i] = nft_total_ion_mg(m, i);
    nft_run<f32x8>(m, 30 * 1440);
    f64 worst = 0.0;
    u32 worst_i = 0;
    for (u32 i = 0; i < NFT_IONS; ++i) {
        f64 in = initial[i] + m->acc.dosed_mg[i];
        f64 out = nft_total_ion_mg(m, i) + m->acc.uptake_mg[i] + m->acc.dumped_mg[i];
        f64 e = fabs(in - out) / in;
        if (e > worst) { worst = e; worst_i = i; }
    }
    f64 water_err = fabs((v0 - m->acc.transpired_L + m->acc.topoff_L + m->acc.dosed_L - m->acc.dumped_L) - m->res.V_L) / v0;
    printf("[nft] 30 days, dosing + top-off + dump every 10 d: worst ion mass balance error %.1e (%s), water %.1e\n",
           worst, nft_field_names[NFT_ION0 + worst_i], water_err);
    CHECK(worst < 1e-6, "nft ion mass balance %.3e", worst);
    CHECK(water_err < 1e-9, "nft water balance %.3e", water_err);

    // 3. thermal: at steady state the energy balance closes exactly -- pump heat in
    //    = channel walls + tank walls out (no chiller, radiant or ambient swing)
    arena.used = 0;
    NftParams pt = p;
    pt.auto_size = 0; pt.chiller_on = 0; pt.radiant_W_m2 = 0; pt.ambient_amp_C = 0; pt.ambient_mean_C = 27.0f;
    pt.reservoir_L = 100.0f; pt.tank_ua_W_K = 3.0f; pt.topoff_on = 1;
    NftSystem *t = push_struct(&arena, NftSystem);
    nft_init(t, &arena, pt);
    nft_run<f32x8>(t, 20 * 1440);
    f64 q_walls = 0.0;
    for (u32 r = 0; r < t->rails; ++r) {
        const NftWrap *wrap = nft_wraps + pt.bank[nft_rail_bank(t, r) ? 1 : 0].wrap;
        f64 ua_cell = wrap->u_w_m2k * nft_profiles[pt.hyd.profile].perimeter_m * t->cell_m;
        for (u32 j = 0; j < t->cells; ++j) q_walls += ua_cell * ((f64)nft_abs(t, r / NFT_LANES, j, NftF_T, r % NFT_LANES) - 27.0);
    }
    f64 q_tank = pt.tank_ua_W_K * (t->res.T_C - 27.0);
    f64 q_pump = nft_pump_heat_W(&t->params, &t->hyd);
    f64 balance = (q_pump - q_walls - q_tank) / q_pump;
    printf("[nft] steady state at 27 C ambient: tank %.4f C; pump heat %.2f W = walls %.2f W + tank %.2f W (residual %.1e)\n",
           t->res.T_C, q_pump, q_walls, q_tank, balance);
    CHECK(fabs(balance) < 2e-3, "steady-state energy balance residual %.3e", balance);

    // 4. chiller keeps the tank inside its band; mylar wrap needs less chiller energy than bare PVC
    f64 kwh[2];
    for (u32 wrap = 0; wrap < 2; ++wrap) {
        arena.used = 0;
        NftParams pc = p;
        pc.auto_size = 0; pc.ambient_mean_C = 28.0f; pc.ambient_amp_C = 4.0f; pc.radiant_W_m2 = 300.0f;
        pc.chiller_W = 1500.0f;
        pc.bank[0].wrap = pc.bank[1].wrap = wrap == 0 ? NftWrap_BubbleMylar : NftWrap_BarePvc;
        NftSystem *c = push_struct(&arena, NftSystem);
        nft_init(c, &arena, pc);
        nft_run<f32x8>(c, 2 * 1440);   // settle
        f64 lo = 1e9, hi = -1e9, e0 = c->acc.chiller_elec_J;
        for (u32 w = 0; w < 5 * 1440; ++w) {
            nft_run<f32x8>(c, 1);
            lo = Minimum(lo, c->res.T_C); hi = Maximum(hi, c->res.T_C);
        }
        kwh[wrap] = (c->acc.chiller_elec_J - e0) / 3.6e6;
        printf("[nft] %-17s 28+-4 C, 300 W/m2: tank %.2f..%.2f C (set %.1f +- %.1f), chiller %.2f kWh/day\n",
               nft_wraps[wrap == 0 ? NftWrap_BubbleMylar : NftWrap_BarePvc].name, lo, hi, pc.chiller_set_C,
               pc.chiller_band_C, kwh[wrap] / 5.0);
        // mylar: load well under capacity -> the band (plus one window's drift) holds.
        // bare PVC at this load runs the 1500 W chiller near capacity and may overshoot (reported).
        if (wrap == 0)
            CHECK(lo > pc.chiller_set_C - pc.chiller_band_C - 0.15 && hi < pc.chiller_set_C + pc.chiller_band_C + 0.15,
                  "chiller band violated: %.2f..%.2f", lo, hi);
    }
    CHECK(kwh[0] < kwh[1], "mylar should need less chiller energy (%.2f vs %.2f)", kwh[0], kwh[1]);

    // 5. growth: unlimited plants follow the analytic logistic curve to harvest
    arena.used = 0;
    NftParams pg = p;
    pg.bank[0].stagger_days = pg.bank[1].stagger_days = 0.0f;   // all planted at t = 0
    pg.ambient_amp_C = 0.0f; pg.ambient_mean_C = 20.0f; pg.photoperiod_h = 24.0f;   // constant conditions
    NftSystem *gs = push_struct(&arena, NftSystem);
    nft_init(gs, &arena, pg);
    const NftCrop *cil = nft_crops + pg.bank[0].crop;
    f64 days = 20.0;
    nft_run<f32x8>(gs, (u32)(days * 1440));
    f64 w, ft, fn, fd;
    nft_bank_plants(gs, 0, &w, &ft, &fn, &fd);
    f64 e = exp(cil->rgr_per_day * days);
    f64 analytic = cil->wmax_g * cil->w0_g * e / (cil->wmax_g + cil->w0_g * (e - 1.0));
    printf("[nft] %s after %.0f days, all limits 1 (fT %.2f fN %.2f fDO %.2f): mean dry mass %.4f g vs logistic %.4f g (%.2f %%)\n",
           cil->name, days, ft, fn, fd, w, analytic, 100.0 * (w - analytic) / analytic);
    CHECK(fabs(w - analytic) / analytic < 0.01, "growth deviates from logistic: %.4f vs %.4f", w, analytic);
}

internal void test_hydraulics() {
    // water properties vs reference tables (IAPWS): mu(20 C) 1.0016e-3 Pa s, rho(20 C) 998.21, rho(4 C) 999.97
    f64 mu20 = nft_water_mu(20.0), rho20 = nft_water_rho(20.0), rho4 = nft_water_rho(4.0);
    printf("[hyd] water: mu(20) %.4e Pa s, rho(20) %.3f, rho(4) %.3f kg/m3\n", mu20, rho20, rho4);
    CHECK(fabs(mu20 / 1.0016e-3 - 1.0) < 3e-3, "viscosity %.4e", mu20);
    CHECK(fabs(rho20 - 998.207) < 0.02 && fabs(rho4 - 999.972) < 0.01, "density %.3f %.3f", rho20, rho4);

    // Colebrook: the solution satisfies the equation to machine precision; Moody value at Re 1e5, e/D 1e-4
    f64 f = nft_colebrook(1e5, 1e-4);
    f64 res = 1.0 / sqrt(f) + 2.0 * log10(1e-4 / 3.7 + 2.51 / (1e5 * sqrt(f)));
    printf("[hyd] Colebrook Re 1e5, e/D 1e-4: f = %.6f, equation residual %.1e\n", f, res);
    CHECK(fabs(res) < 1e-12, "colebrook residual %.3e", res);
    CHECK(fabs(f - 0.018524) < 2e-5, "colebrook value %.6f", f);
    CHECK(fabs(nft_friction(1000.0, 0.02) - 0.064) < 1e-15, "laminar f");

    // network: flow conservation, pump curve balance, manifold non-uniformity, balancing
    NftHydraulicParams hp = nft_hydraulics_defaults();
    static NftHydraulics h;
    nft_hydraulics_solve(&hp, 16, 20.0, &h);
    f64 sum = h.q_eductor_motive;
    for (u32 i = 0; i < 16; ++i) sum += h.q_channel[i];
    f64 cons = fabs(sum - h.q_pump) / h.q_pump;
    f64 curve = fabs(nft_pump_head(nft_pumps + hp.pump, h.q_pump) - h.head_pump);
    printf("[hyd] %s, 3/4\" Sch40, 16 taps: pump %.2f L/min at %.3f m (%.0f W, eta %.2f), channels %.3f..%.3f L/min, "
           "eductor %.2f -> %.2f L/min, return %.0f %% of capacity, %u regula-falsi steps\n",
           nft_pumps[hp.pump].name, h.q_pump * 60000, h.head_pump, h.pump_power_W, h.pump_eta, h.q_min * 60000,
           h.q_max * 60000, h.q_eductor_motive * 60000, h.q_eductor_total * 60000, 100 * h.return_use, h.iterations);
    CHECK(h.ok, "pump does not reach the manifold");
    CHECK(cons < 1e-12, "flow not conserved: %.3e", cons);
    CHECK(curve < 1e-9, "pump curve not balanced: %.3e m", curve);
    CHECK(h.q_channel[0] > h.q_channel[15], "first tap should get more than the dead end");
    hp.balance_taps = 1;
    nft_hydraulics_balance(&hp, 16, 20.0, &h);
    f64 spread = h.q_max / h.q_min - 1.0;
    printf("[hyd] balanced taps: channels %.4f..%.4f L/min (spread %.1e)\n", h.q_min * 60000, h.q_max * 60000, spread);
    CHECK(spread < 1e-3, "balancing spread %.3e", spread);

    // film: laminar depth is the Nusselt film
    f64 nu = 1.0e-6, q = 1.5 / 60000.0, w = 0.1, S = 0.02;
    f64 hn = cbrt(3.0 * nu * (q / w) / (NFT_G * S));
    CHECK(fabs(nft_film_depth(q, w, S, nu) - hn) < 1e-15, "laminar film depth");
}

internal void bench_nft() {
    Arena arena = arena_make(g_nft_memory, sizeof(g_nft_memory));
    u32 channels[] = { 8, 64, 512 };
    for (u32 ci = 0; ci < ArrayCount(channels); ++ci) {
        arena.used = 0;
        NftParams p = nft_default_params();
        p.rails = channels[ci];
        NftSystem *s = push_struct(&arena, NftSystem);
        nft_init(s, &arena, p);
        f64 t0 = now_s();
        nft_run<f32x8>(s, 1440);
        f64 wall = now_s() - t0;
        f64 updates = (f64)s->groups * s->cells * s->substeps * 1440.0 * (f64)NftF_Count * NFT_LANES;
        printf("[nft bench] %4u channels x %u cells, 1 day on 1 thread: %.1f ms (%.0fx real time, %.2f G lane-field updates/s)\n",
               s->rails, s->cells, wall * 1e3, 86400.0 / wall, updates / wall * 1e-9);
    }
}

internal void bench_cpu(KinModel *model, AirSpecies *sp, AirSourceModel *src) {
    Arena arena = arena_make(g_arena_memory, sizeof(g_arena_memory));
    KinMethod method = kin_method(KinMethod_Rodas3);
    KinControl ctl = dbd_default_control();
    for (u32 mode = 0; mode < 2; ++mode) {
        for (u32 variant = 0; variant < 4; ++variant) {
            arena.used = 0;
            DbdZones z;
            setup_zones(&z, &arena, model, sp, src, 64, (DbdDischargeMode)mode);
            f64 t_start = now_s();
            u32 ms = 20;
            for (u32 step = 0; step < ms; ++step) {
                f64 t0 = step * 1e-3, t1 = (step + 1) * 1e-3;
                if (variant == 0) dbd_step_cpu<f64>(&z, model, sp, src, (DbdDischargeMode)mode, &method, &ctl, t0, t1);
                else if (variant == 1) dbd_step_cpu<f64x4>(&z, model, sp, src, (DbdDischargeMode)mode, &method, &ctl, t0, t1);
                else if (variant == 2) dbd_step_cpu<f64x4, AirGenOps>(&z, model, sp, src, (DbdDischargeMode)mode, &method, &ctl, t0, t1);
                else dbd_step_cpu<f32x8, AirGenOps, true>(&z, model, sp, src, (DbdDischargeMode)mode, &method, &ctl, t0, t1);
            }
            f64 wall = now_s() - t_start;
            f64 steps = 0.0;
            for (u32 i = 0; i < z.count; ++i) steps += z.steps[i];
            printf("[bench] %s %-11s 64 zones x %u ms: %.1f ms wall, %.2f zone-ms per wall-ms, %.1f us per zone-step\n",
                   mode == 0 ? "pulsed  " : "averaged", variant == 0 ? "f64 table" : variant == 1 ? "f64x4 table" : variant == 2 ? "f64x4 gen" : "f32x8 gen", ms, wall * 1e3,
                   64.0 * ms / (wall * 1e3), wall * 1e6 / steps);
        }
    }
}

// ---------------------------------------------------------------------------

internal void test_gpu(KinModel *model, AirSpecies *sp, AirSourceModel *src, b32 bench);

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
    test_generated_matches_table(model, &sp, &src);
    test_f32(model, &sp, &src);
    test_conservation(model, &sp, &src);
    test_ozone_yield(model, &sp, &src);
    test_hydraulics();
    test_nft();
    if (want_gpu) test_gpu(model, &sp, &src, want_bench);
    if (want_bench) { bench_nft(); bench_kernels(model, &sp); bench_tolerance(model, &sp, &src); bench_cpu(model, &sp, &src); }

    printf("%u checks, %u failed (simd=%d)\n", g_checks, g_failures, SIM_SIMD);
    free(model);
    return (int)g_failures;
}

// CUDA path: bit-identical to the CPU path, then throughput vs zone count.
internal void test_gpu(KinModel *model, AirSpecies *sp, AirSourceModel *src, b32 bench) {
    char path[512];
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    path[len > 0 ? len : 0] = 0;
    char *slash = strrchr(path, '/');
    snprintf(slash ? slash + 1 : path, sizeof(path) - (slash ? (u64)(slash + 1 - path) : 0), "simstack_gpu.so");
    GpuAPI gpu;
    if (!linux_load_gpu_module(path, Megabytes(512), &gpu)) {
        CHECK(0, "gpu module not available");
        return;
    }
    printf("[gpu] %s, sm_%u%u, %u SMs, %.0f MB free of %.0f MB at init\n", gpu.info.name, gpu.info.sm_major,
           gpu.info.sm_minor, gpu.info.multiprocessors, gpu.info.free_bytes / 1048576.0, gpu.info.total_bytes / 1048576.0);

    KinMethod method = kin_method(KinMethod_Rodas3);
    KinControl ctl = dbd_default_control();
    s32 err = gpu.gpu_kin_set_mech(&model->mech, &method, &ctl);
    CHECK(err == 0, "gpu_kin_set_mech: %s", gpu.gpu_error_string(err));

    Arena arena = arena_make(g_arena_memory, sizeof(g_arena_memory));
    for (u32 mode = 0; mode < 2; ++mode) {
        DbdZones a, b;
        DbdGpu g = {};
        setup_zones(&a, &arena, model, sp, src, 12, (DbdDischargeMode)mode);
        setup_zones(&b, &arena, model, sp, src, 12, (DbdDischargeMode)mode);
        for (u32 step = 0; step < 5; ++step) {
            f64 t0 = step * 1e-3, t1 = (step + 1) * 1e-3;
            dbd_zones_prepare_source(&a, sp, src, (DbdDischargeMode)mode, t0, t1);
            err = dbd_advance_gpu(&a, &gpu, &g, t1 - t0);
            dbd_zones_commit_pulses(&a);
            CHECK(err == 0, "gpu advance: %s", gpu.gpu_error_string(err));
            dbd_step_cpu<f64x4>(&b, model, sp, src, (DbdDischargeMode)mode, &method, &ctl, t0, t1);
        }
        u64 bytes = sizeof(f64) * (u64)a.n_species * a.padded;
        b32 same = memcmp(a.y, b.y, bytes) == 0 && memcmp(a.h, b.h, sizeof(f64) * a.padded) == 0 &&
                   memcmp(a.steps, b.steps, sizeof(f64) * a.padded) == 0;
        f64 worst = 0.0;
        for (u64 i = 0; i < (u64)a.n_species * a.padded; ++i) worst = fmax(worst, rel_err(a.y[i], b.y[i]));
        printf("[gpu] %s, 12 zones x 5 ms: CUDA vs AVX2 %s (hash %016llx, worst relative difference %.1e)\n",
               mode == 0 ? "pulsed  " : "averaged", same ? "bit-identical" : "DIFFERENT",
               (unsigned long long)hash_fnv1a_64(a.y, bytes), worst);
        CHECK(same, "gpu and cpu paths differ (mode %u)", mode);
    }
    for (u32 mode = 0; mode < 2; ++mode) {
        DbdZones a, b;
        DbdGpu g = {};
        setup_zones(&a, &arena, model, sp, src, 16, (DbdDischargeMode)mode);
        setup_zones(&b, &arena, model, sp, src, 16, (DbdDischargeMode)mode);
        for (u32 step = 0; step < 5; ++step) {
            f64 t0 = step * 1e-3, t1 = (step + 1) * 1e-3;
            dbd_zones_prepare_source(&a, sp, src, (DbdDischargeMode)mode, t0, t1);
            err = dbd_advance_gpu(&a, &gpu, &g, t1 - t0, DbdPrecision_F32);
            dbd_zones_commit_pulses(&a);
            CHECK(err == 0, "gpu f32 advance: %s", gpu.gpu_error_string(err));
            dbd_step_cpu<f32x8, AirGenOps, true>(&b, model, sp, src, (DbdDischargeMode)mode, &method, &ctl, t0, t1);
        }
        b32 same = memcmp(a.y, b.y, sizeof(f64) * (u64)a.n_species * a.padded) == 0;
        printf("[gpu] %s, 16 zones x 5 ms, f32 lanes: CUDA vs AVX2 %s\n", mode == 0 ? "pulsed  " : "averaged",
               same ? "bit-identical" : "DIFFERENT");
        CHECK(same, "gpu and cpu f32 paths differ (mode %u)", mode);
    }
    // Regression: long multi-step runs with uploads every step (a pageable cudaMemcpy
    // on the default stream once raced a kernel on the module's non-blocking stream).
    {
        KinMethod m4 = kin_method(KinMethod_Rodas3);
        gpu.gpu_kin_set_mech(&model->mech, &m4, &ctl);
        for (u32 precision = 0; precision < 2; ++precision) {
            DbdZones a, b;
            DbdGpu g = {};
            setup_zones(&a, &arena, model, sp, src, 64, DbdDischarge_Pulsed);
            setup_zones(&b, &arena, model, sp, src, 64, DbdDischarge_Pulsed);
            const f64 dt = 0.02;
            for (u32 step = 0; step < 10; ++step) {
                f64 t0 = step * dt, t1 = (step + 1) * dt;
                dbd_zones_prepare_source(&a, sp, src, DbdDischarge_Pulsed, t0, t1);
                err = dbd_advance_gpu(&a, &gpu, &g, dt, precision);
                dbd_zones_commit_pulses(&a);
                dbd_zones_prepare_source(&b, sp, src, DbdDischarge_Pulsed, t0, t1);
                if (precision == DbdPrecision_F32) dbd_advance_cpu<f32x8, AirGenOps, true>(&b, &model->mech, &m4, &ctl, dt);
                else dbd_advance_cpu<f64x4, AirGenOps>(&b, &model->mech, &m4, &ctl, dt);
                dbd_zones_commit_pulses(&b);
            }
            b32 same = memcmp(a.y, b.y, sizeof(f64) * (u64)a.n_species * a.padded) == 0;
            printf("[gpu] %s lanes, 64 zones x 10 x 20 ms (upload every step): CUDA vs AVX2 %s\n",
                   precision == DbdPrecision_F32 ? "f32" : "f64", same ? "bit-identical" : "DIFFERENT");
            CHECK(same, "multi-step GPU run differs (precision %u)", precision);
        }
    }

    // Warp per zone: same bits as the CPU, both precisions and discharge modes.
    {
        KinMethod method3 = kin_method(KinMethod_Rodas3);
        gpu.gpu_kin_set_mech(&model->mech, &method3, &ctl);
        CHECK(gpu.gpu_kin_set_mode(1) == 0, "warp mode rejected");
        for (u32 precision = 0; precision < 2; ++precision)
            for (u32 mode = 0; mode < 2; ++mode) {
                DbdZones a, b;
                DbdGpu g = {};
                setup_zones(&a, &arena, model, sp, src, 24, (DbdDischargeMode)mode);
                setup_zones(&b, &arena, model, sp, src, 24, (DbdDischargeMode)mode);
                for (u32 step = 0; step < 5; ++step) {
                    f64 t0 = step * 1e-3, t1 = (step + 1) * 1e-3;
                    dbd_zones_prepare_source(&a, sp, src, (DbdDischargeMode)mode, t0, t1);
                    err = dbd_advance_gpu(&a, &gpu, &g, t1 - t0, precision);
                    dbd_zones_commit_pulses(&a);
                    CHECK(err == 0, "warp advance: %s", gpu.gpu_error_string(err));
                    if (precision == DbdPrecision_F32) dbd_step_cpu<f32x8, AirGenOps, true>(&b, model, sp, src, (DbdDischargeMode)mode, &method3, &ctl, t0, t1);
                    else dbd_step_cpu<f64x4, AirGenOps>(&b, model, sp, src, (DbdDischargeMode)mode, &method3, &ctl, t0, t1);
                }
                b32 same = memcmp(a.y, b.y, sizeof(f64) * (u64)a.n_species * a.padded) == 0 &&
                           memcmp(a.steps, b.steps, sizeof(f64) * a.padded) == 0;
                printf("[gpu] warp per zone, %s lanes, %s, 24 zones x 5 ms: CUDA vs AVX2 %s\n",
                       precision == DbdPrecision_F32 ? "f32" : "f64", mode == 0 ? "pulsed  " : "averaged", same ? "bit-identical" : "DIFFERENT");
                CHECK(same, "warp kernel differs (precision %u mode %u)", precision, mode);
            }
        gpu.gpu_kin_set_mode(0);
    }

    // Hybrid: any split of lanes between CUDA and the CPU gives the same bits.
    {
        PlatformWorkQueue *queue = (PlatformWorkQueue *)calloc(1, sizeof(PlatformWorkQueue));
        linux_make_work_queue(queue, 3, -1, 0);
        PlatformAPI api = {};
        api.add_work_entry = linux_add_work_entry;
        api.complete_all_work = linux_complete_all_work;
        api.now_ns = [](void) -> u64 { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (u64)ts.tv_sec * 1000000000ull + ts.tv_nsec; };
        static u8 frame_mem[Megabytes(8)];
        Arena frame = arena_make(frame_mem, sizeof(frame_mem));
        for (u32 precision = 0; precision < 2; ++precision) {
            KinMethod method2 = kin_method(KinMethod_Rodas3);
            gpu.gpu_kin_set_mech(&model->mech, &method2, &ctl);
            DbdZones ref;
            setup_zones(&ref, &arena, model, sp, src, 64, DbdDischarge_Pulsed);
            for (u32 step = 0; step < 3; ++step) {
                if (precision == DbdPrecision_F32) dbd_step_cpu<f32x8, AirGenOps, true>(&ref, model, sp, src, DbdDischarge_Pulsed, &method2, &ctl, step * 1e-3, (step + 1) * 1e-3);
                else dbd_step_cpu<f64x4, AirGenOps>(&ref, model, sp, src, DbdDischarge_Pulsed, &method2, &ctl, step * 1e-3, (step + 1) * 1e-3);
            }
            u32 splits[] = { 0, 8, 24, 40, 64 };
            b32 all_same = 1;
            for (u32 si = 0; si < ArrayCount(splits); ++si) {
                DbdZones z;
                DbdGpu g = {};
                DbdHybrid hy = {};
                hy.force_split = (s32)splits[si];
                setup_zones(&z, &arena, model, sp, src, 64, DbdDischarge_Pulsed);
                for (u32 step = 0; step < 3; ++step) {
                    dbd_zones_prepare_source(&z, sp, src, DbdDischarge_Pulsed, step * 1e-3, (step + 1) * 1e-3);
                    s32 e2 = dbd_advance_hybrid(&z, &gpu, &g, &hy, 1e-3, precision, &model->mech, &method2, &ctl, 1, &api, queue, 4, &frame);
                    CHECK(e2 == 0, "hybrid advance: %s", gpu.gpu_error_string(e2));
                    dbd_zones_commit_pulses(&z);
                }
                all_same &= memcmp(z.y, ref.y, sizeof(f64) * (u64)z.n_species * z.padded) == 0;
            }
            printf("[gpu] hybrid CUDA+CPU, %s lanes, splits 0/8/24/40/64 of 64 zones: %s\n",
                   precision == DbdPrecision_F32 ? "f32" : "f64", all_same ? "all bit-identical to the CPU" : "DIFFERENT");
            CHECK(all_same, "hybrid split changes results (precision %u)", precision);
        }
    }
    if (!bench) return;

    // Throughput: averaged mode, 10 ms per run; one launch per 1 ms master step.
    u32 counts[] = { 64, 256, 1024, 4096, 16384 };
    for (u32 ci = 0; ci < ArrayCount(counts); ++ci) {
        for (u32 mode = 0; mode < 2; ++mode) {
            u32 n = counts[ci];
            u32 ms = mode == 0 ? 2 : 10;
            DbdZones a, b;
            DbdGpu g = {};
            arena.used = 0;
            setup_zones(&a, &arena, model, sp, src, n, (DbdDischargeMode)mode);
            setup_zones(&b, &arena, model, sp, src, n, (DbdDischargeMode)mode);
            f64 gpu_s = 0.0, cpu_s = 0.0, gpu32_s = 0.0, cpu32_s = 0.0;
            DbdZones c, d;
            DbdGpu g32 = {};
            setup_zones(&c, &arena, model, sp, src, n, (DbdDischargeMode)mode);
            setup_zones(&d, &arena, model, sp, src, n, (DbdDischargeMode)mode);
            b32 run_cpu = n <= 4096;
            for (u32 step = 0; step < ms; ++step) {
                f64 t0 = step * 1e-3, t1 = (step + 1) * 1e-3;
                f64 s0 = now_s();
                dbd_zones_prepare_source(&a, sp, src, (DbdDischargeMode)mode, t0, t1);
                dbd_advance_gpu(&a, &gpu, &g, t1 - t0);
                dbd_zones_commit_pulses(&a);
                f64 s1 = now_s();
                if (run_cpu) dbd_step_cpu<f64x4, AirGenOps>(&b, model, sp, src, (DbdDischargeMode)mode, &method, &ctl, t0, t1);
                f64 s2 = now_s();
                dbd_zones_prepare_source(&c, sp, src, (DbdDischargeMode)mode, t0, t1);
                dbd_advance_gpu(&c, &gpu, &g32, t1 - t0, DbdPrecision_F32);
                dbd_zones_commit_pulses(&c);
                f64 s3 = now_s();
                if (run_cpu) dbd_step_cpu<f32x8, AirGenOps, true>(&d, model, sp, src, (DbdDischargeMode)mode, &method, &ctl, t0, t1);
                f64 s4 = now_s();
                gpu_s += s1 - s0; cpu_s += s2 - s1; gpu32_s += s3 - s2; cpu32_s += s4 - s3;
            }
            // the same f32 case with one warp per zone
            DbdZones w;
            DbdGpu gw = {};
            setup_zones(&w, &arena, model, sp, src, n, (DbdDischargeMode)mode);
            gpu.gpu_kin_set_mode(1);
            f64 warp_s = 0.0;
            for (u32 step = 0; step < ms; ++step) {
                f64 t0 = step * 1e-3, t1 = (step + 1) * 1e-3;
                f64 s0 = now_s();
                dbd_zones_prepare_source(&w, sp, src, (DbdDischargeMode)mode, t0, t1);
                dbd_advance_gpu(&w, &gpu, &gw, t1 - t0, DbdPrecision_F32);
                dbd_zones_commit_pulses(&w);
                warp_s += now_s() - s0;
            }
            gpu.gpu_kin_set_mode(0);
            printf("[gpu bench] %s %5u zones x %2u ms:  f64: CUDA %7.1f  AVX2x1 %7.1f  |  f32: CUDA thread %7.1f  warp %7.1f  AVX2x1 %7.1f ms\n",
                   mode == 0 ? "pulsed  " : "averaged", n, ms, gpu_s * 1e3, run_cpu ? cpu_s * 1e3 : 0.0, gpu32_s * 1e3,
                   warp_s * 1e3, run_cpu ? cpu32_s * 1e3 : 0.0);
        }
    }
}
