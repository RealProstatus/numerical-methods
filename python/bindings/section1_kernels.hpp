#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <mkl.h>

#include "core/expm/ChebyshevExpm.h"
#include "core/magnus/AccMagnus.h"
#include "core/magnus/ClassicMagnus.h"
#include "core/magnus/Formula314Magnus.h"
#include "core/magnus/RecursiveMagnus.h"
#include "core/matrix/MatrixOperations.h"

namespace magnus_sec1 {

    using Complex = std::complex<double>;
    using CMatrix = std::vector<Complex>;

    // Вычисление генератора Omega на одном шаге [t0, t0 + dt] выбранным методом Магнуса.
    // Возвращает пару (Omega, время_вычисления_мс).
    inline std::pair<CMatrix, double> compute_omega_step(
        const std::string& method,
        int order,
        int N,
        const CMatrix& H0,
        const CMatrix& H1,
        double t0,
        double dt,
        double eps,
        double omega,
        int sub_steps = 10
    ) {
        CMatrix Omega(static_cast<std::size_t>(N) * N, Complex(0.0, 0.0));
        const double t1 = t0 + dt;

        auto t_start = std::chrono::high_resolution_clock::now();

        if (method == "Classic") {
            const double inner_dt = dt / static_cast<double>(std::max(2, sub_steps));
            Omega = magnus_classic(t0, t1, inner_dt, N, H0, H1, eps, omega, order);
        }
        else if (method == "ACC") {
            // Для 6-го порядка используем 7 точек (6 подинтервалов), для <=4 — 5 точек (4 подинтервала)
            const int m_sub = (order >= 5) ? 6 : 4;
            const double inner_dt = dt / static_cast<double>(m_sub);
            std::vector<CMatrix> samples = generate_samples(t0, t1, inner_dt, N, H0, H1, eps, omega);
            Omega = magnus_ACC(samples, inner_dt, N, order);
        }
        else if (method == "Formula314") {
            // Formula314 ожидает антиэрмитовы генераторы A0 = -i*H0, A1 = -i*H1
            CMatrix A0 = matrix_ops::mat_copy(H0);
            CMatrix A1 = matrix_ops::mat_copy(H1);
            matrix_ops::mat_scale_inplace(A0, N, Complex(0.0, -1.0));
            matrix_ops::mat_scale_inplace(A1, N, Complex(0.0, -1.0));
            Omega = magnus_3_14(t0, t1, dt, N, A0, A1, eps, omega);
        }
        else if (method == "Recursive") {
            const double inner_dt = dt / static_cast<double>(std::max(2, sub_steps));
            Omega = magnus_expansion(t0, t1, inner_dt, N, H0, H1, eps, omega, order);
        }
        else {
            throw std::invalid_argument("Unknown Magnus Omega method: " + method);
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        return { std::move(Omega), elapsed_ms };
    }

    // Быстрый плотный RK4 на MKL cblas_zgemm без переаллокаций внутри цикла
    inline std::pair<CMatrix, double> propagate_rk4_mkl(
        int N,
        const CMatrix& H0,
        const CMatrix& H1,
        double t0,
        double tf,
        int steps,
        double eps,
        double omega
    ) {
        const std::size_t elems = static_cast<std::size_t>(N) * N;
        const double h = (tf - t0) / static_cast<double>(steps);
        const double h2 = 0.5 * h;
        const double h6 = h / 6.0;

        CMatrix A0 = matrix_ops::mat_copy(H0);
        CMatrix A1 = matrix_ops::mat_copy(H1);
        matrix_ops::mat_scale_inplace(A0, N, Complex(0.0, -1.0));
        matrix_ops::mat_scale_inplace(A1, N, Complex(0.0, -1.0));

        CMatrix U(elems, Complex(0.0, 0.0));
        for (int i = 0; i < N; ++i) {
            U[static_cast<std::size_t>(i) * N + i] = Complex(1.0, 0.0);
        }

        CMatrix A_t(elems), K1(elems), K2(elems), K3(elems), K4(elems), U_tmp(elems);
        const Complex one(1.0, 0.0);
        const Complex zero(0.0, 0.0);

        auto eval_rhs = [&](double t, const CMatrix& U_in, CMatrix& K_out) {
            const double f_t = eps * std::cos(omega * t);
            std::memcpy(A_t.data(), A0.data(), elems * sizeof(Complex));
            matrix_ops::mat_axpy(A1, A_t, N, Complex(f_t, 0.0));
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, &one, A_t.data(), N, U_in.data(), N, &zero, K_out.data(), N);
            };

        auto t_start = std::chrono::high_resolution_clock::now();

        for (int s = 0; s < steps; ++s) {
            const double t = t0 + s * h;

            eval_rhs(t, U, K1);

            std::memcpy(U_tmp.data(), U.data(), elems * sizeof(Complex));
            matrix_ops::mat_axpy(K1, U_tmp, N, Complex(h2, 0.0));
            eval_rhs(t + h2, U_tmp, K2);

            std::memcpy(U_tmp.data(), U.data(), elems * sizeof(Complex));
            matrix_ops::mat_axpy(K2, U_tmp, N, Complex(h2, 0.0));
            eval_rhs(t + h2, U_tmp, K3);

            std::memcpy(U_tmp.data(), U.data(), elems * sizeof(Complex));
            matrix_ops::mat_axpy(K3, U_tmp, N, Complex(h, 0.0));
            eval_rhs(t + h, U_tmp, K4);

            matrix_ops::mat_axpy(K1, U, N, Complex(h6, 0.0));
            matrix_ops::mat_axpy(K2, U, N, Complex(2.0 * h6, 0.0));
            matrix_ops::mat_axpy(K3, U, N, Complex(2.0 * h6, 0.0));
            matrix_ops::mat_axpy(K4, U, N, Complex(h6, 0.0));
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        return { std::move(U), elapsed_ms };
    }

    // Кусочный метод Магнуса ACC с заданным числом макрошагов и внутренних подинтервалов (для EXP-08 и EXP-09)
    inline std::pair<CMatrix, double> propagate_piecewise_acc_inner(
        int N,
        const CMatrix& H0,
        const CMatrix& H1,
        double t0,
        double tf,
        int macro_steps,
        int inner_steps,
        double eps,
        double omega,
        int order = 4,
        int cheb_degree = 16
    ) {
        const std::size_t elems = static_cast<std::size_t>(N) * N;
        const double macro_dt = (tf - t0) / static_cast<double>(macro_steps);
        const int sub = std::max(2, inner_steps);
        const double inner_dt = macro_dt / static_cast<double>(sub);

        CMatrix U_total(elems, Complex(0.0, 0.0));
        for (int i = 0; i < N; ++i) {
            U_total[static_cast<std::size_t>(i) * N + i] = Complex(1.0, 0.0);
        }
        CMatrix U_next(elems, Complex(0.0, 0.0));

        auto t_start = std::chrono::high_resolution_clock::now();

        for (int m = 0; m < macro_steps; ++m) {
            const double ts = t0 + m * macro_dt;
            const double te = ts + macro_dt;
            std::vector<CMatrix> samples = generate_samples(ts, te, inner_dt, N, H0, H1, eps, omega);
            CMatrix Omega = magnus_ACC(samples, inner_dt, N, order);
            CMatrix U_local = matrix_ops::expm_cheb(Omega, N, cheb_degree);
            matrix_ops::matmul(U_local, U_total, U_next, N);
            U_total.swap(U_next);
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        return { std::move(U_total), elapsed_ms };
    }

} // namespace magnus_sec1