#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <mkl.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "core/expm/ChebyshevExpm.h"
#include "core/expm/EigenPadeExpm.h"
#include "core/expm/TaylorExpm.h"
#include "core/magnus/AccMagnus.h"
#include "core/magnus/ClassicMagnus.h"
#include "core/magnus/Formula314Magnus.h"
#include "core/magnus/RecursiveMagnus.h"
#include "core/matrix/MatrixOperations.h"
#include "solvers/single_cpu/Formula314ChebyshevSolver.h"
#include "solvers/single_cpu/PiecewiseMagnus.h"

namespace magnus {
    std::tuple<double, double, double> benchmark_gpu_events_raw(
        int N,
        const std::complex<double>* H0_ptr,
        const std::complex<double>* H1_ptr,
        double t0,
        double tf,
        int steps,
        double eps,
        double omega,
        int cheb_degree,
        std::complex<double>* U_out_ptr
    );

    std::pair<std::size_t, std::size_t> get_cuda_mem_info_raw();
}

namespace magnus_sec2 {

    using Complex = std::complex<double>;
    using CMatrix = std::vector<Complex>;

    // Обратная рекуррентность Миллера для J_0(alpha) .. J_M(alpha) без переполнения при больших alpha
    inline std::vector<double> bessel_j_array_miller(int M, double alpha) {
        std::vector<double> res(static_cast<std::size_t>(M) + 1, 0.0);
        if (alpha < 1e-16) {
            res[0] = 1.0;
            return res;
        }
        const int pad = std::max(50, static_cast<int>(std::ceil(5.0 * std::sqrt(static_cast<double>(M) + alpha))) + 30);
        const int k_start = M + pad;
        std::vector<double> J(static_cast<std::size_t>(k_start) + 2, 0.0);
        J[k_start] = 1e-30;
        const double two_over_alpha = 2.0 / alpha;

        for (int k = k_start; k >= 1; --k) {
            J[k - 1] = (static_cast<double>(k) * two_over_alpha) * J[k] - J[k + 1];
            if (std::abs(J[k - 1]) > 1e200) {
                for (int m = k - 1; m <= k_start; ++m) {
                    J[m] *= 1e-200;
                }
            }
        }

        double norm = J[0];
        for (int m = 2; m <= k_start; m += 2) {
            norm += 2.0 * J[m];
        }
        for (int k = 0; k <= M; ++k) {
            res[k] = J[k] / norm;
        }
        return res;
    }

    // EXP-1.3: exp(Omega) методом Чебышева СО скейлингом (возвращает ms, zgemm_count, squarings_s)
    inline std::tuple<double, int, int> expm_cheb_scaled_stats(
        const CMatrix& Omega, int N, int M, CMatrix& out
    ) {
        auto t0 = std::chrono::high_resolution_clock::now();
        const double norm = matrix_ops::mat_one_norm(Omega, N);
        if (norm < 1e-14) {
            out.assign(static_cast<std::size_t>(N) * N, Complex(0.0, 0.0));
            for (int i = 0; i < N; ++i) out[static_cast<std::size_t>(i) * N + i] = Complex(1.0, 0.0);
            auto t1 = std::chrono::high_resolution_clock::now();
            return { std::chrono::duration<double, std::milli>(t1 - t0).count(), 0, 0 };
        }

        int s = 0;
        double alpha = norm;
        while (alpha > 1.0) {
            alpha *= 0.5;
            ++s;
        }

        const Complex scale(0.0, 1.0 / (std::ldexp(1.0, s) * alpha));
        CMatrix X = Omega;
        matrix_ops::mat_scale_inplace(X, N, scale);

        const std::size_t elems = static_cast<std::size_t>(N) * N;
        CMatrix Tk_prev(elems, Complex(0.0, 0.0));
        for (int i = 0; i < N; ++i) Tk_prev[static_cast<std::size_t>(i) * N + i] = Complex(1.0, 0.0);
        CMatrix Tk_curr = X;
        CMatrix Tk_next(elems, Complex(0.0, 0.0));
        CMatrix res(elems, Complex(0.0, 0.0));

        const auto J = bessel_j_array_miller(M, alpha);
        matrix_ops::mat_axpy(Tk_prev, res, N, Complex(J[0], 0.0));
        if (M >= 1) {
            matrix_ops::mat_axpy(Tk_curr, res, N, Complex(0.0, -2.0 * J[1]));
        }

        int zgemm_count = 0;
        for (int k = 2; k <= M; ++k) {
            if (std::abs(J[k]) < 1e-18) continue;
            matrix_ops::matmul(X, Tk_curr, Tk_next, N, Complex(2.0, 0.0));
            ++zgemm_count;
            matrix_ops::mat_sub(Tk_next, Tk_prev, Tk_next, N);

            Complex i_pow;
            switch (k % 4) {
            case 0: i_pow = Complex(1.0, 0.0); break;
            case 1: i_pow = Complex(0.0, -1.0); break;
            case 2: i_pow = Complex(-1.0, 0.0); break;
            default: i_pow = Complex(0.0, 1.0); break;
            }
            matrix_ops::mat_axpy(Tk_next, res, N, Complex(2.0 * J[k], 0.0) * i_pow);
            std::swap(Tk_prev, Tk_curr);
            std::swap(Tk_curr, Tk_next);
        }

        CMatrix tmp(elems, Complex(0.0, 0.0));
        for (int i = 0; i < s; ++i) {
            matrix_ops::matmul(res, res, tmp, N);
            ++zgemm_count;
            std::swap(res, tmp);
        }

        out = std::move(res);
        auto t1 = std::chrono::high_resolution_clock::now();
        return { std::chrono::duration<double, std::milli>(t1 - t0).count(), zgemm_count, s };
    }

    // EXP-1.3: exp(Omega) методом Чебышева БЕЗ скейлинга (s = 0, M = ceil(1.36 * alpha + 14))
    inline std::tuple<double, int, int> expm_cheb_no_scaling_stats(
        const CMatrix& Omega, int N, CMatrix& out
    ) {
        auto t0 = std::chrono::high_resolution_clock::now();
        const double alpha = matrix_ops::mat_one_norm(Omega, N);
        if (alpha < 1e-14) {
            out.assign(static_cast<std::size_t>(N) * N, Complex(0.0, 0.0));
            for (int i = 0; i < N; ++i) out[static_cast<std::size_t>(i) * N + i] = Complex(1.0, 0.0);
            auto t1 = std::chrono::high_resolution_clock::now();
            return { std::chrono::duration<double, std::milli>(t1 - t0).count(), 0, 0 };
        }

        const int M = std::max(16, static_cast<int>(std::ceil(1.36 * alpha + 14.0)));
        const Complex scale(0.0, 1.0 / alpha);
        CMatrix X = Omega;
        matrix_ops::mat_scale_inplace(X, N, scale);

        const std::size_t elems = static_cast<std::size_t>(N) * N;
        CMatrix Tk_prev(elems, Complex(0.0, 0.0));
        for (int i = 0; i < N; ++i) Tk_prev[static_cast<std::size_t>(i) * N + i] = Complex(1.0, 0.0);
        CMatrix Tk_curr = X;
        CMatrix Tk_next(elems, Complex(0.0, 0.0));
        CMatrix res(elems, Complex(0.0, 0.0));

        const auto J = bessel_j_array_miller(M, alpha);
        matrix_ops::mat_axpy(Tk_prev, res, N, Complex(J[0], 0.0));
        matrix_ops::mat_axpy(Tk_curr, res, N, Complex(0.0, -2.0 * J[1]));

        int zgemm_count = 0;
        for (int k = 2; k <= M; ++k) {
            if (std::abs(J[k]) < 1e-18) continue;
            matrix_ops::matmul(X, Tk_curr, Tk_next, N, Complex(2.0, 0.0));
            ++zgemm_count;
            matrix_ops::mat_sub(Tk_next, Tk_prev, Tk_next, N);

            Complex i_pow;
            switch (k % 4) {
            case 0: i_pow = Complex(1.0, 0.0); break;
            case 1: i_pow = Complex(0.0, -1.0); break;
            case 2: i_pow = Complex(-1.0, 0.0); break;
            default: i_pow = Complex(0.0, 1.0); break;
            }
            matrix_ops::mat_axpy(Tk_next, res, N, Complex(2.0 * J[k], 0.0) * i_pow);
            std::swap(Tk_prev, Tk_curr);
            std::swap(Tk_curr, Tk_next);
        }

        out = std::move(res);
        auto t1 = std::chrono::high_resolution_clock::now();
        return { std::chrono::duration<double, std::milli>(t1 - t0).count(), zgemm_count, M };
    }

    // Универсальный пропагатор по всем методам (EXP-1.1, EXP-3.1)
    inline double propagate_method_cpu(
        const std::string& method_name,
        const std::string& expm_name,
        int N,
        const CMatrix& H0,
        const CMatrix& H1,
        double t0,
        double tf,
        int steps,
        double eps,
        double omega,
        int cheb_degree,
        CMatrix& U_out
    ) {
        const std::size_t elems = static_cast<std::size_t>(N) * N;
        const double h = (tf - t0) / static_cast<double>(steps);

        CMatrix A0 = matrix_ops::mat_copy(H0);
        CMatrix A1 = matrix_ops::mat_copy(H1);
        matrix_ops::mat_scale_inplace(A0, N, Complex(0.0, -1.0));
        matrix_ops::mat_scale_inplace(A1, N, Complex(0.0, -1.0));

        auto apply_expm = [&](const CMatrix& Om) -> CMatrix {
            if (expm_name == "PadeEigen") {
                return matrix_ops::expm_pade_eigen(Om, N);
            }
            if (expm_name == "Taylor") {
                return matrix_ops::expm_taylor(Om, N, 30);
            }
            return matrix_ops::expm_cheb(Om, N, cheb_degree);
            };

        auto t_start = std::chrono::high_resolution_clock::now();

        if (method_name == "RK4") {
            U_out.assign(elems, Complex(0.0, 0.0));
            for (int i = 0; i < N; ++i) U_out[static_cast<std::size_t>(i) * N + i] = Complex(1.0, 0.0);

            CMatrix A_t(elems), K1(elems), K2(elems), K3(elems), K4(elems), U_tmp(elems);
            const Complex one(1.0, 0.0), zero(0.0, 0.0);
            const double h2 = 0.5 * h, h6 = h / 6.0;

            auto eval_rhs = [&](double t, const CMatrix& U_in, CMatrix& K_out) {
                const double f_t = eps * std::cos(omega * t);
                std::memcpy(A_t.data(), A0.data(), elems * sizeof(Complex));
                matrix_ops::mat_axpy(A1, A_t, N, Complex(f_t, 0.0));
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, &one, A_t.data(), N, U_in.data(), N, &zero, K_out.data(), N);
                };

            for (int s = 0; s < steps; ++s) {
                const double t = t0 + s * h;
                eval_rhs(t, U_out, K1);
                std::memcpy(U_tmp.data(), U_out.data(), elems * sizeof(Complex));
                matrix_ops::mat_axpy(K1, U_tmp, N, Complex(h2, 0.0));
                eval_rhs(t + h2, U_tmp, K2);
                std::memcpy(U_tmp.data(), U_out.data(), elems * sizeof(Complex));
                matrix_ops::mat_axpy(K2, U_tmp, N, Complex(h2, 0.0));
                eval_rhs(t + h2, U_tmp, K3);
                std::memcpy(U_tmp.data(), U_out.data(), elems * sizeof(Complex));
                matrix_ops::mat_axpy(K3, U_tmp, N, Complex(h, 0.0));
                eval_rhs(t + h, U_tmp, K4);

                matrix_ops::mat_axpy(K1, U_out, N, Complex(h6, 0.0));
                matrix_ops::mat_axpy(K2, U_out, N, Complex(2.0 * h6, 0.0));
                matrix_ops::mat_axpy(K3, U_out, N, Complex(2.0 * h6, 0.0));
                matrix_ops::mat_axpy(K4, U_out, N, Complex(h6, 0.0));
            }
        }
        else {
            U_out.assign(elems, Complex(0.0, 0.0));
            for (int i = 0; i < N; ++i) U_out[static_cast<std::size_t>(i) * N + i] = Complex(1.0, 0.0);
            CMatrix U_next(elems, Complex(0.0, 0.0));

            CMatrix C1(elems, Complex(0.0, 0.0));
            matrix_ops::commutator_inplace(A0, A1, C1, N);

            for (int s = 0; s < steps; ++s) {
                const double ts = t0 + s * h;
                const double te = ts + h;
                CMatrix Omega(elems, Complex(0.0, 0.0));

                if (method_name == "Classic2") {
                    const double th = ts + 0.5 * h;
                    const double f0 = eps * std::cos(omega * ts);
                    const double fh = eps * std::cos(omega * th);
                    const double f1 = eps * std::cos(omega * te);
                    const double int_f = (h / 6.0) * (f0 + 4.0 * fh + f1);
                    matrix_ops::mat_axpy(A0, Omega, N, Complex(h, 0.0));
                    matrix_ops::mat_axpy(A1, Omega, N, Complex(int_f, 0.0));
                }
                else if (method_name == "Formula314_4") {
                    const double th = ts + 0.5 * h;
                    const double f0 = eps * std::cos(omega * ts);
                    const double fh = eps * std::cos(omega * th);
                    const double f1 = eps * std::cos(omega * te);
                    const double fp = -eps * omega * std::sin(omega * th);
                    const double f2 = -eps * omega * omega * std::cos(omega * th);
                    const double c_A1 = h * (0.1 * f0 + 0.8 * fh + 0.1 * f1) + (h * h * h / 60.0) * f2;
                    const double c_C1 = -(h * h / 20.0) * (f1 - f0) - (h * h * h / 30.0) * fp;
                    matrix_ops::mat_axpy(A0, Omega, N, Complex(h, 0.0));
                    matrix_ops::mat_axpy(A1, Omega, N, Complex(c_A1, 0.0));
                    matrix_ops::mat_axpy(C1, Omega, N, Complex(c_C1, 0.0));
                }
                else if (method_name == "Formula314_6") {
                    Omega = magnus_3_14(ts, te, h, N, A0, A1, eps, omega);
                }
                else if (method_name == "Acc4") {
                    const double micro_dt = 0.5 * h;
                    auto samples = generate_samples(ts, te, micro_dt, N, H0, H1, eps, omega);
                    Omega = magnus_ACC(samples, micro_dt, N, 4);
                }
                else if (method_name == "Acc6") {
                    const double micro_dt = h / 6.0;
                    auto samples = generate_samples(ts, te, micro_dt, N, H0, H1, eps, omega);
                    Omega = magnus_ACC(samples, micro_dt, N, 6);
                }
                else if (method_name == "Recursive8") {
                    const double micro_dt = h / 6.0;
                    Omega = magnus_expansion(ts, te, micro_dt, N, H0, H1, eps, omega, 8);
                }
                else {
                    throw std::invalid_argument("Unknown method in propagate_method_cpu: " + method_name);
                }

                CMatrix U_loc = apply_expm(Omega);
                matrix_ops::matmul(U_loc, U_out, U_next, N);
                U_out.swap(U_next);
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t_end - t_start).count();
    }

    // EXP-1.2: Сверхдлинный прогон на N_steps с записью чекпоинтов физических инвариантов
    inline std::vector<double> run_long_time_invariants(
        const std::string& solver_name,
        int N,
        const CMatrix& H0,
        const CMatrix& H1,
        double dt,
        int total_steps,
        int checkpoint_every,
        double eps,
        double omega,
        int cheb_degree
    ) {
        using RowMat = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        const std::size_t elems = static_cast<std::size_t>(N) * N;

        CMatrix A0 = matrix_ops::mat_copy(H0);
        CMatrix A1 = matrix_ops::mat_copy(H1);
        matrix_ops::mat_scale_inplace(A0, N, Complex(0.0, -1.0));
        matrix_ops::mat_scale_inplace(A1, N, Complex(0.0, -1.0));

        RowMat rho_pure0 = RowMat::Zero(N, N);
        rho_pure0(0, 0) = Complex(1.0, 0.0);

        RowMat rho_mix0 = RowMat::Zero(N, N);
        std::vector<double> eigs0(N);
        double w_sum = 0.0;
        for (int j = 0; j < N; ++j) {
            eigs0[j] = std::exp(-0.15 * j);
            w_sum += eigs0[j];
        }
        for (int j = 0; j < N; ++j) {
            eigs0[j] /= w_sum;
            rho_mix0(j, j) = Complex(eigs0[j], 0.0);
        }
        std::sort(eigs0.begin(), eigs0.end());

        CMatrix U(elems, Complex(0.0, 0.0));
        for (int i = 0; i < N; ++i) U[static_cast<std::size_t>(i) * N + i] = Complex(1.0, 0.0);
        CMatrix U_next(elems, Complex(0.0, 0.0));

        CMatrix A_t(elems), K1(elems), K2(elems), K3(elems), K4(elems), U_tmp(elems);
        const Complex one(1.0, 0.0), zero(0.0, 0.0);
        const double h2 = 0.5 * dt, h6 = dt / 6.0;

        auto eval_rhs = [&](double t, const CMatrix& U_in, CMatrix& K_out) {
            const double f_t = eps * std::cos(omega * t);
            std::memcpy(A_t.data(), A0.data(), elems * sizeof(Complex));
            matrix_ops::mat_axpy(A1, A_t, N, Complex(f_t, 0.0));
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, &one, A_t.data(), N, U_in.data(), N, &zero, K_out.data(), N);
            };

        std::vector<double> log_rows;
        log_rows.reserve(static_cast<std::size_t>(total_steps / checkpoint_every + 2) * 6);

        for (int step = 1; step <= total_steps; ++step) {
            const double ts = (step - 1) * dt;
            const double te = step * dt;

            if (solver_name == "Formula314Chebyshev") {
                CMatrix Om = magnus_3_14(ts, te, dt, N, A0, A1, eps, omega);
                CMatrix U_loc = matrix_ops::expm_cheb(Om, N, cheb_degree);
                matrix_ops::matmul(U_loc, U, U_next, N);
                U.swap(U_next);
            }
            else if (solver_name == "PiecewiseMagnusPade") {
                const double micro_dt = dt / 6.0;
                auto samples = generate_samples(ts, te, micro_dt, N, H0, H1, eps, omega);
                CMatrix Om = magnus_ACC(samples, micro_dt, N, 6);
                CMatrix U_loc = matrix_ops::expm_pade_eigen(Om, N);
                matrix_ops::matmul(U_loc, U, U_next, N);
                U.swap(U_next);
            }
            else if (solver_name == "RK4") {
                eval_rhs(ts, U, K1);
                std::memcpy(U_tmp.data(), U.data(), elems * sizeof(Complex));
                matrix_ops::mat_axpy(K1, U_tmp, N, Complex(h2, 0.0));
                eval_rhs(ts + h2, U_tmp, K2);
                std::memcpy(U_tmp.data(), U.data(), elems * sizeof(Complex));
                matrix_ops::mat_axpy(K2, U_tmp, N, Complex(h2, 0.0));
                eval_rhs(ts + h2, U_tmp, K3);
                std::memcpy(U_tmp.data(), U.data(), elems * sizeof(Complex));
                matrix_ops::mat_axpy(K3, U_tmp, N, Complex(dt, 0.0));
                eval_rhs(ts + dt, U_tmp, K4);

                matrix_ops::mat_axpy(K1, U, N, Complex(h6, 0.0));
                matrix_ops::mat_axpy(K2, U, N, Complex(2.0 * h6, 0.0));
                matrix_ops::mat_axpy(K3, U, N, Complex(2.0 * h6, 0.0));
                matrix_ops::mat_axpy(K4, U, N, Complex(h6, 0.0));
            }

            if (step % checkpoint_every == 0 || step == total_steps) {
                Eigen::Map<const RowMat> U_mat(U.data(), N, N);
                RowMat I_mat = RowMat::Identity(N, N);
                double unit_def = (U_mat.adjoint() * U_mat - I_mat).norm() / std::sqrt(static_cast<double>(N));

                RowMat rho_p = U_mat * rho_pure0 * U_mat.adjoint();
                double tr_def = std::abs(rho_p.trace() - Complex(1.0, 0.0));
                double pur_def = std::abs((rho_p * rho_p).trace() - Complex(1.0, 0.0));

                RowMat rho_m = U_mat * rho_mix0 * U_mat.adjoint();
                rho_m = 0.5 * (rho_m + rho_m.adjoint());
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(rho_m, Eigen::EigenvaluesOnly);
                double spec_drift = 0.0;
                for (int j = 0; j < N; ++j) {
                    spec_drift = std::max(spec_drift, std::abs(es.eigenvalues()[j] - eigs0[j]));
                }

                log_rows.push_back(static_cast<double>(step));
                log_rows.push_back(te);
                log_rows.push_back(unit_def);
                log_rows.push_back(tr_def);
                log_rows.push_back(pur_def);
                log_rows.push_back(spec_drift);
            }
        }

        return log_rows;
    }

    // EXP-4.1: Детальное GPU-профилирование через cudaEvent_t (выполняется в magnus_bridge_cuda.cu)
    inline std::tuple<double, double, double> benchmark_gpu_events(
        int N,
        const CMatrix& H0,
        const CMatrix& H1,
        double t0,
        double tf,
        int steps,
        double eps,
        double omega,
        int cheb_degree,
        CMatrix& U_out
    ) {
        U_out.resize(static_cast<std::size_t>(N) * N);
        return magnus::benchmark_gpu_events_raw(
            N, H0.data(), H1.data(), t0, tf, steps, eps, omega, cheb_degree, U_out.data()
        );
    }

    // EXP-5.1b: Мониторинг свободной и полной VRAM через cudaMemGetInfo
    inline std::pair<std::size_t, std::size_t> get_cuda_mem_info() {
        return magnus::get_cuda_mem_info_raw();
    }

    // EXP-5.1b: Детерминированные эрмитовы матрицы на std::mt19937_64(seed)
    inline void generate_mt19937_hermitian_pair(int N, std::uint64_t seed, CMatrix& H0, CMatrix& H1) {
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> dist(0.0, 1.0);
        const std::size_t elems = static_cast<std::size_t>(N) * N;
        H0.resize(elems);
        H1.resize(elems);

        for (int i = 0; i < N; ++i) {
            H0[i * N + i] = Complex(dist(rng), 0.0);
            H1[i * N + i] = Complex(dist(rng), 0.0);
            for (int j = i + 1; j < N; ++j) {
                Complex z0(0.5 * dist(rng), 0.5 * dist(rng));
                H0[i * N + j] = z0;
                H0[j * N + i] = std::conj(z0);

                Complex z1(0.5 * dist(rng), 0.5 * dist(rng));
                H1[i * N + j] = z1;
                H1[j * N + i] = std::conj(z1);
            }
        }
    }

} // namespace magnus_sec2