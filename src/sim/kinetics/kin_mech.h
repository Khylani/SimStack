#ifndef KIN_MECH_H
#define KIN_MECH_H

// A mass-action reaction mechanism compiled into flat index tables, plus the
// symbolic sparse LU of its Rosenbrock iteration matrix. Plain data: the same
// bytes are used by the CPU solver and copied into CUDA constant memory.
//
// Built on the host by kin_build.cpp (add reactions -> kin_mech_finalize).
// Rate *constants* are not in here: they depend on temperature and gas
// density per lane and are evaluated on the host into a k[reaction][lane]
// SoA array (kin_build.cpp: kin_eval_rate_constants).
//
// Sparse LU with no pivoting, as in KPP: the iteration matrix
// I/(h*gamma) - J of a chemistry Jacobian has a dominant positive diagonal,
// and a fixed elimination order lets every lane (AVX2 lanes, CUDA threads)
// run the same instruction stream with no data-dependent branches.

#include "../../base/base_types.h"

#define KIN_MAX_SPECIES     32
#define KIN_MAX_REACTIONS   96
#define KIN_MAX_REACTANTS   3
#define KIN_MAX_NET         6      // species whose count changes in one reaction
#define KIN_MAX_NNZ         512    // Jacobian + fill-in
#define KIN_MAX_LU_DIV      512
#define KIN_MAX_LU_SUB      4096
#define KIN_MAX_JAC_TERMS   (KIN_MAX_REACTIONS * KIN_MAX_REACTANTS * KIN_MAX_NET)
#define KIN_NONE            0xFF

struct KinReaction {
    u8 reactant[KIN_MAX_REACTANTS];   // species index or KIN_NONE; M (third body) is folded into k
    u8 reactant_count;
    u8 net_count;
    u8 net_species[KIN_MAX_NET];
    s8 net_coef[KIN_MAX_NET];         // products minus reactants (non-zero only)
    // Jacobian scatter: for reactant position p and net entry e,
    // J[jac_index[p][e]] += net_coef[e] * d(rate)/d(y[reactant[p]]).
    u16 jac_index[KIN_MAX_REACTANTS][KIN_MAX_NET];
};

// One step of no-pivot LU in IKJ order: L(i,k) = A(i,k) * (1 / A(k,k)); then for
// each listed (i,j),(k,j): A(i,j) -= L(i,k) * A(k,j). Indices are into the
// compact non-zero array. After row k is finished its diagonal entry holds
// 1/U(k,k), so elimination and the triangular solves never divide.
struct KinLuDiv {
    u16 ik, kk;
    u16 sub_begin, sub_end;
};
struct KinLuSub {
    u16 ij, kj;
};

// Triangular solve: row i of L (unit diagonal) and U, as (compact index, column).
struct KinRowEntry {
    u16 index;
    u8 col;
    u8 pad;
};

struct KinMech {
    u32 species_count;
    u32 reaction_count;
    KinReaction reactions[KIN_MAX_REACTIONS];

    u32 nnz;                               // non-zeros of J plus diagonal plus LU fill-in
    u16 diag[KIN_MAX_SPECIES];             // compact index of (i,i)
    u32 lu_div_count, lu_sub_count;
    u16 div_begin[KIN_MAX_SPECIES + 1];    // lu_div ops of row i: [div_begin[i], div_begin[i+1])
    KinLuDiv lu_div[KIN_MAX_LU_DIV];
    KinLuSub lu_sub[KIN_MAX_LU_SUB];
    u16 lower_begin[KIN_MAX_SPECIES + 1];  // into lower[]: strictly-lower entries of row i, ascending col
    u16 upper_begin[KIN_MAX_SPECIES + 1];  // into upper[]: strictly-upper entries of row i, ascending col
    KinRowEntry lower[KIN_MAX_NNZ];
    KinRowEntry upper[KIN_MAX_NNZ];
};

// Rosenbrock method coefficients in KPP's form (Sandu et al. 1997):
//   stage i: (I/(h*gamma_0) - J) K_i = F(Y + sum_j a_ij K_j) + sum_j (c_ij / h) K_j
//   Y_new = Y + sum_i m_i K_i,   Err = sum_i e_i K_i
// a and c are packed strictly-lower-triangular: (i,j), j<i at i*(i-1)/2 + j.
#define KIN_MAX_STAGES 4
struct KinMethod {
    u32 stages;
    u32 order;            // ELO: order used for step-size control
    f64 a[6], c[6];
    f64 m[KIN_MAX_STAGES], e[KIN_MAX_STAGES];
    f64 gamma0;           // diagonal gamma
    b32 new_f[KIN_MAX_STAGES];
    const char *name;
};

enum KinMethodId {
    KinMethod_Ros2,       // 2 stages, order 2(1), L-stable
    KinMethod_Ros3,       // 3 stages, order 3(2), L-stable
    KinMethod_Rodas3,     // 4 stages, order 3(2), stiffly accurate
    KinMethod_Count,
};

// Tolerances and step limits for one integration call (same for every lane).
struct KinControl {
    f64 rtol, atol;       // weighted RMS error norm per lane
    f64 h_min, h_max;     // seconds; h_max <= 0 means "segment length"
    f64 h_start;          // step after a reset (start, discharge pulse)
    u32 max_steps;        // per segment per lane; exceeded lanes are flagged
    b32 fixed_step;       // tests: accept every step and never change h
    b32 clamp_negative;   // clamp y to >= 0 after accepted steps
};

#endif // KIN_MECH_H
