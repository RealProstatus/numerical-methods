#pragma once

#include "Utils.h"
#include "MatrixOperations.h"

using namespace std;
using namespace utils;
using namespace matrix_ops;

namespace runge_kutta_simple {
    // Simple 4th-order Runge–Kutta step for the matrix ODE U'(t) = -i * H * U(t)
    CMatrix runge_kutta_step(const CMatrix& H, const CMatrix& U, double dt, int N) {
        complexd minus_i(0.0, -1.0);

        // k1 = -i * H * U
        CMatrix k1(N * N, complexd(0.0, 0.0));
        matmul(H, U, k1, N);
        for (auto& elem : k1) elem *= minus_i;

        // U + dt/2 * k1
        CMatrix temp1(N * N, complexd(0.0, 0.0));
        for (int i = 0; i < N * N; ++i) {
            temp1[i] = U[i] + k1[i] * complexd(dt / 2.0, 0.0);
        }

        // k2 = -i * H * (U + dt/2 * k1)
        CMatrix k2(N * N, complexd(0.0, 0.0));
        matmul(H, temp1, k2, N);
        for (auto& elem : k2) elem *= minus_i;

        // U + dt/2 * k2
        CMatrix temp2(N * N, complexd(0.0, 0.0));
        for (int i = 0; i < N * N; ++i) {
            temp2[i] = U[i] + k2[i] * complexd(dt / 2.0, 0.0);
        }

        // k3 = -i * H * (U + dt/2 * k2)
        CMatrix k3(N * N, complexd(0.0, 0.0));
        matmul(H, temp2, k3, N);
        for (auto& elem : k3) elem *= minus_i;

        // U + dt * k3
        CMatrix temp3(N * N, complexd(0.0, 0.0));
        for (int i = 0; i < N * N; ++i) {
            temp3[i] = U[i] + k3[i] * complexd(dt, 0.0);
        }

        // k4 = -i * H * (U + dt * k3)
        CMatrix k4(N * N, complexd(0.0, 0.0));
        matmul(H, temp3, k4, N);
        for (auto& elem : k4) elem *= minus_i;

        // k1 + 2*k2 + 2*k3 + k4
        CMatrix k_sum(N * N, complexd(0.0, 0.0));
        for (int i = 0; i < N * N; ++i) {
            k_sum[i] = k1[i] +
                k2[i] * complexd(2.0, 0.0) +
                k3[i] * complexd(2.0, 0.0) +
                k4[i];
        }

        // U_new = U + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        CMatrix U_new(N * N, complexd(0.0, 0.0));
        for (int i = 0; i < N * N; ++i) {
            U_new[i] = U[i] + k_sum[i] * complexd(dt / 6.0, 0.0);
        }

        return U_new;
    }
}