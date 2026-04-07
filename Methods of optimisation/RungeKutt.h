#pragma once
#include <iostream>
#include <complex>
#include <vector>
#include "Utils.h"
#include "MatrixOperations.h"

using namespace std;
using namespace utils;
using namespace matrix_ops;

namespace runge_kutta_simple {

    // Вспомогательная функция для сборки H(t) = A + B * cos(W * t)
    inline CMatrix get_H_t(const CMatrix& A, const CMatrix& B, double W, double t, int N) {
        CMatrix H(N * N, complexd(0.0, 0.0));
        double cos_wt = cos(W * t);
        for (int i = 0; i < N * N; ++i) {
            H[i] = A[i] + B[i] * complexd(cos_wt, 0.0);
        }
        return H;
    }

    // Метод Рунге-Кутты 4-го порядка для зависящего от времени H(t)
    CMatrix runge_kutta_step(double t, const CMatrix& A, const CMatrix& B, double W, const CMatrix& U, double dt, int N) {
        complexd minus_i(0.0, -1.0);

        // --- Шаг 1 --- (время t)
        CMatrix H1 = get_H_t(A, B, W, t, N);
        CMatrix k1(N * N, complexd(0.0, 0.0));
        matmul(H1, U, k1, N);
        for (auto& elem : k1) elem *= minus_i;

        CMatrix temp1(N * N, complexd(0.0, 0.0));
        for (int i = 0; i < N * N; ++i) {
            temp1[i] = U[i] + k1[i] * complexd(dt / 2.0, 0.0);
        }

        // --- Шаг 2 --- (время t + dt/2)
        CMatrix H2 = get_H_t(A, B, W, t + dt / 2.0, N);
        CMatrix k2(N * N, complexd(0.0, 0.0));
        matmul(H2, temp1, k2, N);
        for (auto& elem : k2) elem *= minus_i;

        CMatrix temp2(N * N, complexd(0.0, 0.0));
        for (int i = 0; i < N * N; ++i) {
            temp2[i] = U[i] + k2[i] * complexd(dt / 2.0, 0.0);
        }

        // --- Шаг 3 --- (время t + dt/2)
        // Для РК4 H3 берется в той же точке времени, что и H2
        CMatrix k3(N * N, complexd(0.0, 0.0));
        matmul(H2, temp2, k3, N);
        for (auto& elem : k3) elem *= minus_i;

        CMatrix temp3(N * N, complexd(0.0, 0.0));
        for (int i = 0; i < N * N; ++i) {
            temp3[i] = U[i] + k3[i] * complexd(dt, 0.0);
        }

        // --- Шаг 4 --- (время t + dt)
        CMatrix H4 = get_H_t(A, B, W, t + dt, N);
        CMatrix k4(N * N, complexd(0.0, 0.0));
        matmul(H4, temp3, k4, N);
        for (auto& elem : k4) elem *= minus_i;

        // --- Сборка ---
        CMatrix k_sum(N * N, complexd(0.0, 0.0));
        for (int i = 0; i < N * N; ++i) {
            k_sum[i] = k1[i] +
                k2[i] * complexd(2.0, 0.0) +
                k3[i] * complexd(2.0, 0.0) +
                k4[i];
        }

        CMatrix U_new(N * N, complexd(0.0, 0.0));
        for (int i = 0; i < N * N; ++i) {
            U_new[i] = U[i] + k_sum[i] * complexd(dt / 6.0, 0.0);
        }

        return U_new;
    }
}