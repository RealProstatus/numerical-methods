#pragma once

#include "MatrixOperations.h"

// =============================================================
// ===================== Magnus via formula 3.14 ===============
// =============================================================
CMatrix magnus_3_14(
    double t0, double t1, double dt,
    int N, const CMatrix& H0, const CMatrix& H_mod,
    double eps0, double W);

namespace matrix_ops {
    CMatrix compute_Omega_3_14(
        const CMatrix& H0,
        const CMatrix& Hmod,
        double h,
        double f_t1,
        double f_half,
        double fp_half,
        double f2_half,
        double f2_t1,
        int N);
}


