#pragma once

#include "MatrixOperations.h"

#include <vector>

using namespace std;

// ================================================================
// =================== FULL ACCURACY Recursive Magnus =============
// ================================================================
CMatrix magnus_expansion(double t0, double t1, double integration_dt, int N,
    const CMatrix& H0, const CMatrix& H_mod,
    double eps0, double W, int max_n = 10);


