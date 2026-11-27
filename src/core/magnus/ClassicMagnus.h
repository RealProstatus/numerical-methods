#pragma once

#include "MatrixOperations.h"

using namespace std;

// ------ Генерация сэмплов A(t) = -i * (H0 + eps0 * sin(W * t) * H_mod) ------
vector<CMatrix> generate_samples(double t0, double t1, double dt, int N,
    const CMatrix& H0, const CMatrix& H_mod,
    double eps0, double W);

CMatrix trapezoidal_integral(const vector<CMatrix>& samples, double dt, int N);

// ------ Simpson's rule integration (4th order accuracy, 1/3 + 3/8 for even M >= 4) ------
CMatrix simpson_integral(const vector<CMatrix>& samples, double dt, int N);

// ------ Cumulative Simpson integration (4th order accuracy at every grid node) ------
vector<CMatrix> cumulative_simpson_integral(const vector<CMatrix>& samples, double dt, int N);

// =============================================================
// =================== Classic Magnus up to 3 ==================
// =============================================================
CMatrix magnus_classic(
    double t0, double t1, double dt,
    int N, const CMatrix& H0, const CMatrix& H_mod,
    double eps0, double W,
    int max_order);
    