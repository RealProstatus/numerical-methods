#pragma once

#include "MatrixOperations.h"

using namespace std;

// ------ Trapezoidal integration for matrix-valued functions ------
vector<CMatrix> generate_samples(double t0, double t1, double dt, int N,
    const CMatrix& H0, const CMatrix& H_mod,
    double eps0, double W);

CMatrix trapezoidal_integral(const vector<CMatrix>& samples, double dt, int N);

// ------ Simpson's rule integration (4th order accuracy) ------
// Requires an odd number of points (an even number of intervals).
CMatrix simpson_integral(const vector<CMatrix>& samples, double dt, int N);

// =============================================================
// =================== Classic Magnus up to 3 ==================
// =============================================================
CMatrix magnus_classic(
    double t0, double t1, double dt,
    int N, const CMatrix& H0, const CMatrix& H_mod,
    double eps0, double W,
    int max_order);


