#pragma once

#include "MatrixOperations.h"

using namespace std;

CMatrix piecewise_magnus_solver(double t0, double T, double macro_dt, int N,
    const CMatrix& H0, const CMatrix& H_mod,
    double eps0, double W, int order = 4); //Кусочный магнус