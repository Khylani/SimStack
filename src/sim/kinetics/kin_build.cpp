// Host side of the kinetics solver: mechanism construction from readable
// equations, element-balance checking, symbolic sparse LU, and rate-constant
// evaluation. Runs at init and when T / pressure change; never per substep.
// Included by the sim unity TU and the test TU (not by nvcc).
//
// Units: number densities in cm^-3, time in s; bimolecular rate constants in
// cm^3 s^-1, termolecular in cm^6 s^-1 (the third body M is folded into k).

#include <math.h>
#include <stdio.h>
#include "kin_solver.h"

#define KIN_ELEMENTS 3   // N, O, H: enough for humid-air chemistry
global const char *const kin_element_names[KIN_ELEMENTS] = {"N", "O", "H"};

enum KinRateKind {
    KinRate_Arrhenius,   // k = A (T/300)^n exp(-E/T)
    KinRate_ThirdBody,   // k = A (T/300)^n exp(-E/T) * [M]          (low-pressure limit)
    KinRate_Troe,        // JPL fall-off: k0 = A (T/300)^n, kinf = A_inf (T/300)^n_inf, Fc = fc
    KinRate_EqReverse,   // k = k[ref] / (Keq_A exp(Keq_B / T))     (thermal decomposition)
};

struct KinRateLaw {
    u32 kind;
    f64 A, n, E;
    f64 A_inf, n_inf, fc;
    u32 ref;             // EqReverse: index of the forward reaction (must come earlier)
    f64 keq_A, keq_B;
};

struct KinModel {
    KinMech mech;                                  // flat tables shared with the solver / GPU
    KinRateLaw laws[KIN_MAX_REACTIONS];
    char species_name[KIN_MAX_SPECIES][12];
    u8 elements[KIN_MAX_SPECIES][KIN_ELEMENTS];
    char equation[KIN_MAX_REACTIONS][64];
    b32 error;                                     // first error is in error_text
    char error_text[160];
};

// Rate-law constructors (read like the JPL / IUPAC tables they come from).
internal KinRateLaw kin_arr(f64 A, f64 n = 0.0, f64 E = 0.0) {
    KinRateLaw l = {}; l.kind = KinRate_Arrhenius; l.A = A; l.n = n; l.E = E; return l;
}
internal KinRateLaw kin_m(f64 A, f64 n = 0.0, f64 E = 0.0) {
    KinRateLaw l = {}; l.kind = KinRate_ThirdBody; l.A = A; l.n = n; l.E = E; return l;
}
internal KinRateLaw kin_troe(f64 A0, f64 n0, f64 A_inf, f64 n_inf, f64 fc = 0.6) {
    KinRateLaw l = {}; l.kind = KinRate_Troe; l.A = A0; l.n = n0; l.A_inf = A_inf; l.n_inf = n_inf; l.fc = fc; return l;
}
internal KinRateLaw kin_eq_reverse(u32 forward_reaction, f64 keq_A, f64 keq_B) {
    KinRateLaw l = {}; l.kind = KinRate_EqReverse; l.ref = forward_reaction; l.keq_A = keq_A; l.keq_B = keq_B; return l;
}

internal void kin_fail(KinModel *model, const char *text, const char *detail) {
    if (model->error) return;
    model->error = 1;
    snprintf(model->error_text, sizeof(model->error_text), "%s: %s", text, detail);
}

// Element counts from a formula made of N, O, H and digits ("HNO3", "N2O5").
internal b32 kin_parse_formula(const char *name, u8 *counts) {
    for (u32 e = 0; e < KIN_ELEMENTS; ++e) counts[e] = 0;
    const char *p = name;
    while (*p && *p != '(') {                 // "N2(A)", "O(1D)": the label in parentheses is the state
        s32 element = -1;
        for (u32 e = 0; e < KIN_ELEMENTS; ++e) if (*p == kin_element_names[e][0]) element = (s32)e;
        if (element < 0) return 0;
        ++p;
        u32 count = 0;
        while (*p >= '0' && *p <= '9') count = count * 10 + (u32)(*p++ - '0');
        counts[element] += (u8)(count ? count : 1);
    }
    return 1;
}

// abstract = 1: a non-chemical variable (test problems); no element bookkeeping.
internal u32 kin_add_species(KinModel *model, const char *name, b32 abstract = 0) {
    KinMech *m = &model->mech;
    Assert(m->species_count < KIN_MAX_SPECIES);
    u32 s = m->species_count++;
    snprintf(model->species_name[s], sizeof(model->species_name[s]), "%s", name);
    if (abstract) {
        for (u32 e = 0; e < KIN_ELEMENTS; ++e) model->elements[s][e] = 0;
    } else if (!kin_parse_formula(name, model->elements[s])) {
        kin_fail(model, "species formula must use N, O, H", name);
    }
    return s;
}

internal s32 kin_find_species(KinModel *model, const char *name) {
    for (u32 s = 0; s < model->mech.species_count; ++s) {
        if (strcmp(model->species_name[s], name) == 0) return (s32)s;
    }
    return -1;
}

// "O + O2 + M -> O3 + M". M is the third body (handled by the rate law).
// Repeated species are written out ("NO2 + NO2"). Returns the reaction index.
internal u32 kin_add_reaction(KinModel *model, const char *equation, KinRateLaw law) {
    KinMech *m = &model->mech;
    Assert(m->reaction_count < KIN_MAX_REACTIONS);
    u32 r = m->reaction_count++;
    KinReaction *rx = m->reactions + r;
    memset(rx, 0, sizeof(*rx));
    for (u32 p = 0; p < KIN_MAX_REACTANTS; ++p) rx->reactant[p] = KIN_NONE;
    model->laws[r] = law;
    snprintf(model->equation[r], sizeof(model->equation[r]), "%s", equation);

    s32 net[KIN_MAX_SPECIES] = {};
    s32 balance[KIN_ELEMENTS] = {};
    b32 products = 0;
    const char *p = equation;
    char token[16];
    while (*p) {
        while (*p == ' ' || *p == '+') ++p;
        if (!*p) break;
        if (p[0] == '-' && p[1] == '>') { products = 1; p += 2; continue; }
        u32 len = 0;
        while (*p && *p != ' ' && *p != '+' && !(p[0] == '-' && p[1] == '>') && len < sizeof(token) - 1) token[len++] = *p++;
        token[len] = 0;
        if (strcmp(token, "M") == 0) continue;
        s32 s = kin_find_species(model, token);
        if (s < 0) { kin_fail(model, "unknown species in reaction", equation); return r; }
        if (!products) {
            if (rx->reactant_count >= KIN_MAX_REACTANTS) { kin_fail(model, "too many reactants", equation); return r; }
            rx->reactant[rx->reactant_count++] = (u8)s;
            net[s] -= 1;
            for (u32 e = 0; e < KIN_ELEMENTS; ++e) balance[e] -= model->elements[s][e];
        } else {
            net[s] += 1;
            for (u32 e = 0; e < KIN_ELEMENTS; ++e) balance[e] += model->elements[s][e];
        }
    }
    if (!products || rx->reactant_count == 0) kin_fail(model, "reaction needs reactants and '->'", equation);
    for (u32 e = 0; e < KIN_ELEMENTS; ++e) {
        if (balance[e] != 0) kin_fail(model, "element balance violated", equation);
    }
    if (law.kind == KinRate_EqReverse && law.ref >= r) kin_fail(model, "reverse reaction must follow its forward reaction", equation);
    for (u32 s = 0; s < m->species_count; ++s) {
        if (net[s] == 0) continue;
        if (rx->net_count >= KIN_MAX_NET) { kin_fail(model, "too many net species", equation); return r; }
        rx->net_species[rx->net_count] = (u8)s;
        rx->net_coef[rx->net_count] = (s8)net[s];
        ++rx->net_count;
    }
    return r;
}

// Symbolic phase: sparsity of J + diagonal, fill-in of no-pivot IKJ LU, and
// the flat op lists the solver walks. Must be called after the last reaction.
internal void kin_mech_finalize(KinModel *model) {
    KinMech *m = &model->mech;
    const u32 n = m->species_count;
    b32 pattern[KIN_MAX_SPECIES][KIN_MAX_SPECIES] = {};
    for (u32 i = 0; i < n; ++i) pattern[i][i] = 1;
    for (u32 r = 0; r < m->reaction_count; ++r) {
        KinReaction *rx = m->reactions + r;
        for (u32 p = 0; p < rx->reactant_count; ++p)
            for (u32 e = 0; e < rx->net_count; ++e) pattern[rx->net_species[e]][rx->reactant[p]] = 1;
    }
    for (u32 i = 1; i < n; ++i)
        for (u32 k = 0; k < i; ++k)
            if (pattern[i][k])
                for (u32 j = k + 1; j < n; ++j)
                    if (pattern[k][j]) pattern[i][j] = 1;

    u16 index[KIN_MAX_SPECIES][KIN_MAX_SPECIES];
    m->nnz = 0;
    for (u32 i = 0; i < n; ++i)
        for (u32 j = 0; j < n; ++j) {
            index[i][j] = 0xFFFF;
            if (pattern[i][j]) {
                if (m->nnz >= KIN_MAX_NNZ) { kin_fail(model, "mechanism too large", "KIN_MAX_NNZ"); return; }
                index[i][j] = (u16)m->nnz++;
            }
        }
    for (u32 i = 0; i < n; ++i) m->diag[i] = index[i][i];

    m->lu_div_count = m->lu_sub_count = 0;
    m->div_begin[0] = 0;
    for (u32 i = 0; i < n; ++i) {
        m->div_begin[i] = (u16)m->lu_div_count;
        for (u32 k = 0; k < i; ++k) {
            if (!pattern[i][k]) continue;
            if (m->lu_div_count >= KIN_MAX_LU_DIV) { kin_fail(model, "mechanism too large", "KIN_MAX_LU_DIV"); return; }
            KinLuDiv *d = m->lu_div + m->lu_div_count++;
            d->ik = index[i][k];
            d->kk = index[k][k];
            d->sub_begin = (u16)m->lu_sub_count;
            for (u32 j = k + 1; j < n; ++j) {
                if (!pattern[k][j]) continue;
                if (m->lu_sub_count >= KIN_MAX_LU_SUB) { kin_fail(model, "mechanism too large", "KIN_MAX_LU_SUB"); return; }
                KinLuSub *s = m->lu_sub + m->lu_sub_count++;
                s->ij = index[i][j];
                s->kj = index[k][j];
            }
            d->sub_end = (u16)m->lu_sub_count;
        }
    }
    m->div_begin[n] = (u16)m->lu_div_count;

    u32 lo = 0, up = 0;
    for (u32 i = 0; i < n; ++i) {
        m->lower_begin[i] = (u16)lo;
        m->upper_begin[i] = (u16)up;
        for (u32 j = 0; j < i; ++j)
            if (pattern[i][j]) { m->lower[lo].index = index[i][j]; m->lower[lo].col = (u8)j; ++lo; }
        // upper entries in DESCENDING column order: back substitution then applies
        // each row's updates in the same order as a column-oriented sweep (which is
        // what the warp-cooperative GPU solve does), so every path stays bit-identical
        for (u32 j = n; j-- > i + 1;)
            if (pattern[i][j]) { m->upper[up].index = index[i][j]; m->upper[up].col = (u8)j; ++up; }
    }
    m->lower_begin[n] = (u16)lo;
    m->upper_begin[n] = (u16)up;

    for (u32 r = 0; r < m->reaction_count; ++r) {
        KinReaction *rx = m->reactions + r;
        for (u32 p = 0; p < rx->reactant_count; ++p)
            for (u32 e = 0; e < rx->net_count; ++e) rx->jac_index[p][e] = index[rx->net_species[e]][rx->reactant[p]];
    }
}

// Rate constants for one lane at temperature T [K] and total density M [cm^-3].
internal void kin_eval_rate_constants(KinModel *model, f64 T, f64 M, f64 *k) {
    f64 t300 = T / 300.0;
    for (u32 r = 0; r < model->mech.reaction_count; ++r) {
        KinRateLaw *l = model->laws + r;
        switch (l->kind) {
            case KinRate_Arrhenius: k[r] = l->A * pow(t300, l->n) * exp(-l->E / T); break;
            case KinRate_ThirdBody: k[r] = l->A * pow(t300, l->n) * exp(-l->E / T) * M; break;
            case KinRate_Troe: {
                f64 k0m = l->A * pow(t300, l->n) * M;
                f64 kinf = l->A_inf * pow(t300, l->n_inf);
                f64 ratio = k0m / kinf;
                f64 lg = log10(ratio);
                k[r] = (k0m / (1.0 + ratio)) * pow(l->fc, 1.0 / (1.0 + lg * lg));
            } break;
            case KinRate_EqReverse: k[r] = k[l->ref] / (l->keq_A * exp(l->keq_B / T)); break;
            default: k[r] = 0.0; break;
        }
    }
}

// Total atoms of each element (for conservation checks).
internal void kin_count_elements(KinModel *model, const f64 *y, f64 *atoms) {
    for (u32 e = 0; e < KIN_ELEMENTS; ++e) atoms[e] = 0.0;
    for (u32 s = 0; s < model->mech.species_count; ++s)
        for (u32 e = 0; e < KIN_ELEMENTS; ++e) atoms[e] += (f64)model->elements[s][e] * y[s];
}
