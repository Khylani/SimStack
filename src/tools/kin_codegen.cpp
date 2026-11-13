// Metaprogramming step (Handmade Hero's "simple preprocessor" idea): builds a
// mechanism with the same code the sim uses, then writes straight-line
// C++ for its four solver kernels into a header.
//
//   build/kin_codegen build/gen/air_mech_gen.h
//
// The emitted kernels perform exactly the operations of the table walk in
// kin_solver.h (kin_fun, kin_neg_jac, kin_factor, kin_solve), in the same
// order, so results are bit-identical; only the index loads and loop control
// disappear and every array index becomes a constant. The header also
// carries a signature of the KinMech tables it was generated from; the sim
// compares it with the mechanism it builds at runtime and falls back to the
// table kernels if they differ.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../base/base_types.h"
#include "../sim/kinetics/kin_build.cpp"
#include "../sim/kinetics/air_mech.cpp"

struct Out {
    FILE *f;
};

internal void emit_coef_name(char *buf, u32 size, s32 coef) {
    if (coef < 0) snprintf(buf, size, "cm%d", -coef);
    else snprintf(buf, size, "cp%d", coef);
}

internal void emit_kernels(FILE *f, KinModel *model, const char *prefix, const char *ops_name) {
    KinMech *m = &model->mech;
    u32 n = m->species_count;
    u64 signature = hash_fnv1a_64(m, sizeof(KinMech));

    fprintf(f, "#define %s_SPECIES %u\n#define %s_REACTIONS %u\n#define %s_NNZ %u\n", prefix, n, prefix,
            m->reaction_count, prefix, m->nnz);
    fprintf(f, "#define %s_SIGNATURE 0x%016llxull   // FNV-1a of the KinMech it was generated from\n\n", prefix,
            (unsigned long long)signature);

    // Coefficient constants used by the mechanism (net stoichiometry -2..+2 typically).
    s32 coef_used[32] = {};
    for (u32 r = 0; r < m->reaction_count; ++r)
        for (u32 e = 0; e < m->reactions[r].net_count; ++e) {
            s32 c = m->reactions[r].net_coef[e];
            coef_used[c + 16] = 1;
            coef_used[-c + 16] = 1;
        }
    char cname[16];
#define EMIT_COEFS() \
    for (s32 c = -16; c < 16; ++c) if (coef_used[c + 16] && c != 0) { \
        emit_coef_name(cname, sizeof(cname), c); \
        fprintf(f, "    const V %s = lane_splat((V *)0, %d.0);\n", cname, c); \
    }

    // fun --------------------------------------------------------------------
    fprintf(f, "template <class V>\nSIM_HD void %s_fun(const V *k, const V *src, const V *y, V *f) {\n", prefix);
    EMIT_COEFS();
    for (u32 s = 0; s < n; ++s) fprintf(f, "    f[%u] = src[%u];\n", s, s);
    for (u32 r = 0; r < m->reaction_count; ++r) {
        KinReaction *rx = m->reactions + r;
        fprintf(f, "    { V rate = k[%u]", r);
        for (u32 p = 0; p < rx->reactant_count; ++p) fprintf(f, " * y[%u]", rx->reactant[p]);
        fprintf(f, ";   // %s\n", model->equation[r]);
        for (u32 e = 0; e < rx->net_count; ++e) {
            emit_coef_name(cname, sizeof(cname), rx->net_coef[e]);
            fprintf(f, "      f[%u] = lane_fmadd(%s, rate, f[%u]);\n", rx->net_species[e], cname, rx->net_species[e]);
        }
        fprintf(f, "    }\n");
    }
    fprintf(f, "}\n\n");

    // -J ---------------------------------------------------------------------
    fprintf(f, "template <class V>\nSIM_HD void %s_neg_jac(const V *k, const V *y, V *a) {\n", prefix);
    EMIT_COEFS();
    fprintf(f, "    const V zero = lane_splat((V *)0, 0.0);\n");
    fprintf(f, "    for (u32 i = 0; i < %u; ++i) a[i] = zero;\n", m->nnz);
    for (u32 r = 0; r < m->reaction_count; ++r) {
        KinReaction *rx = m->reactions + r;
        for (u32 p = 0; p < rx->reactant_count; ++p) {
            fprintf(f, "    { V d = k[%u]", r);
            for (u32 q = 0; q < rx->reactant_count; ++q) if (q != p) fprintf(f, " * y[%u]", rx->reactant[q]);
            fprintf(f, ";\n");
            for (u32 e = 0; e < rx->net_count; ++e) {
                emit_coef_name(cname, sizeof(cname), -rx->net_coef[e]);
                u32 idx = rx->jac_index[p][e];
                fprintf(f, "      a[%u] = lane_fmadd(%s, d, a[%u]);\n", idx, cname, idx);
            }
            fprintf(f, "    }\n");
        }
    }
    fprintf(f, "}\n\n");

    // LU -------------------------------------------------------------------
    fprintf(f, "template <class V>\nSIM_HD void %s_factor(V ghinv, V *a) {\n", prefix);
    fprintf(f, "    const V one = lane_splat((V *)0, 1.0);\n");
    for (u32 i = 0; i < n; ++i) {
        for (u32 d = m->div_begin[i]; d < m->div_begin[i + 1]; ++d) {
            KinLuDiv div = m->lu_div[d];
            fprintf(f, "    { V l = a[%u] * a[%u]; a[%u] = l;\n", div.ik, div.kk, div.ik);
            for (u32 su = div.sub_begin; su < div.sub_end; ++su) {
                KinLuSub sub = m->lu_sub[su];
                fprintf(f, "      a[%u] = lane_fnmadd(l, a[%u], a[%u]);\n", sub.ij, sub.kj, sub.ij);
            }
            fprintf(f, "    }\n");
        }
        fprintf(f, "    a[%u] = one / (a[%u] + ghinv);\n", m->diag[i], m->diag[i]);
    }
    fprintf(f, "}\n\n");

    // solve ------------------------------------------------------------------
    fprintf(f, "template <class V>\nSIM_HD void %s_solve(const V *a, V *x) {\n", prefix);
    for (u32 i = 0; i < n; ++i)
        for (u32 e = m->lower_begin[i]; e < m->lower_begin[i + 1]; ++e)
            fprintf(f, "    x[%u] = lane_fnmadd(a[%u], x[%u], x[%u]);\n", i, m->lower[e].index, m->lower[e].col, i);
    for (u32 ii = n; ii-- > 0;) {
        for (u32 e = m->upper_begin[ii]; e < m->upper_begin[ii + 1]; ++e)
            fprintf(f, "    x[%u] = lane_fnmadd(a[%u], x[%u], x[%u]);\n", ii, m->upper[e].index, m->upper[e].col, ii);
        fprintf(f, "    x[%u] = x[%u] * a[%u];\n", ii, ii, m->diag[ii]);
    }
    fprintf(f, "}\n\n");

    // Ops policy -------------------------------------------------------------
    fprintf(f,
            "struct %s {\n"
            "    enum { species = %u, reactions = %u, nnz = %u };\n"
            "    static const u64 signature = %s_SIGNATURE;\n"
            "    static SIM_HD u32 n(const KinMech *) { return %u; }\n"
            "    static SIM_HD bool writer() { return true; }\n"
            "    template <class V> static SIM_HD void fun(const KinMech *, const V *k, const V *src, const V *y, V *f) { %s_fun(k, src, y, f); }\n"
            "    template <class V> static SIM_HD void neg_jac(const KinMech *, const V *k, const V *y, V *a) { %s_neg_jac(k, y, a); }\n"
            "    template <class V> static SIM_HD void factor(const KinMech *, V ghinv, V *a) { %s_factor(ghinv, a); }\n"
            "    template <class V> static SIM_HD void solve(const KinMech *, const V *a, V *x) { %s_solve(a, x); }\n"
            "};\n",
            ops_name, n, m->reaction_count, m->nnz, prefix, n, prefix, prefix, prefix, prefix);
#undef EMIT_COEFS
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: kin_codegen out.h\n");
        return 1;
    }
    static KinModel model;
    AirSpecies sp;
    air_mech_build(&model, &sp);
    if (model.error) {
        fprintf(stderr, "kin_codegen: mechanism error: %s\n", model.error_text);
        return 1;
    }
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s.tmp", argv[1]);
    FILE *f = fopen(tmp, "wb");
    if (!f) {
        fprintf(stderr, "kin_codegen: cannot write %s\n", tmp);
        return 1;
    }
    fprintf(f, "// GENERATED by src/tools/kin_codegen.cpp from src/sim/kinetics/air_mech.cpp. Do not edit.\n");
    fprintf(f, "// Straight-line solver kernels for the humid-air mechanism (see kin_solver.h, \"Ops\").\n");
    fprintf(f, "#ifndef AIR_MECH_GEN_H\n#define AIR_MECH_GEN_H\n\n#include \"sim/kinetics/kin_solver.h\"\n\n");
    emit_kernels(f, &model, "airgen", "AirGenOps");
    fprintf(f, "\n#endif // AIR_MECH_GEN_H\n");
    fclose(f);
    // Only replace the header when it changed, so unchanged builds stay unchanged.
    FILE *old = fopen(argv[1], "rb");
    b32 same = 0;
    if (old) {
        FILE *nw = fopen(tmp, "rb");
        same = 1;
        for (;;) {
            int a = fgetc(old), b = fgetc(nw);
            if (a != b) { same = 0; break; }
            if (a == EOF) break;
        }
        fclose(nw);
        fclose(old);
    }
    if (same) remove(tmp);
    else rename(tmp, argv[1]);
    return 0;
}
