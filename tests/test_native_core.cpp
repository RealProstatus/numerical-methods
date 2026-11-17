#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/expm/ChebyshevExpm.h"
#include "core/expm/EigenPadeExpm.h"
#include "core/expm/TaylorExpm.h"
#include "core/matrix/MatrixOperations.h"
#include "core/utils/ProblemSetup.h"
#include "core/utils/Utils.h"
#include "solvers/single_cpu/Formula314ChebyshevSolver.h"
#include "solvers/single_cpu/PiecewiseMagnus.h"

#if defined(MAGNUS_HAS_CUDA)
#include "solvers/single_gpu/CudaMagnusChebyshev.cuh"
#endif

namespace {

    using Complex = std::complex<double>;

    double frobenius_diff(const CMatrix& A, const CMatrix& B) {
        if (A.size() != B.size()) {
            throw std::runtime_error("Matrix size mismatch in frobenius_diff");
        }
        double sum_sq = 0.0;
        for (std::size_t i = 0; i < A.size(); ++i) {
            sum_sq += std::norm(A[i] - B[i]);
        }
        return std::sqrt(sum_sq);
    }

    double unitarity_defect(const CMatrix& U, int N) {
        // Вычисляем ||U^dagger * U - I||_F / sqrt(N)
        double sum_sq = 0.0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                Complex acc(0.0, 0.0);
                for (int k = 0; k < N; ++k) {
                    acc += std::conj(U[k * N + i]) * U[k * N + j];
                }
                Complex target = (i == j) ? Complex(1.0, 0.0) : Complex(0.0, 0.0);
                sum_sq += std::norm(acc - target);
            }
        }
        return std::sqrt(sum_sq / static_cast<double>(N));
    }

    void check(bool condition, const std::string& msg) {
        if (!condition) {
            std::cerr << "[FAIL] " << msg << "\n";
            std::exit(1);
        }
        std::cout << "[PASS] " << msg << "\n";
    }

    // Тест 1: Аналитический предел (epsilon = 0, H(t) = (w0/2) * sigma_z)
    void test_analytical_stationary_limit() {
        const int N = 2;
        const double w0 = 3.5;
        const double T = 2.0;

        CMatrix H0 = {
            Complex(0.5 * w0, 0.0), Complex(0.0, 0.0),
            Complex(0.0, 0.0),      Complex(-0.5 * w0, 0.0)
        };
        CMatrix H1 = {
            Complex(0.0, 0.0), Complex(1.0, 0.0),
            Complex(1.0, 0.0), Complex(0.0, 0.0)
        };

        utils::PeriodicHamiltonian ham;
        ham.dimension = N;
        ham.H0 = H0;
        ham.H1 = H1;
        ham.epsilon = 0.0; // Выключаем возмущение
        ham.omega = 5.0;
        ham.hbar = 1.0;

        single_cpu::Formula314ChebyshevOptions opts;
        opts.chebyshev_degree = 16;

        CMatrix U_num = single_cpu::solve_formula314_chebyshev(ham, 0.0, T, 50, opts);

        CMatrix U_exact = {
            std::exp(Complex(0.0, -0.5 * w0 * T)), Complex(0.0, 0.0),
            Complex(0.0, 0.0),                     std::exp(Complex(0.0, 0.5 * w0 * T))
        };

        double err = frobenius_diff(U_num, U_exact);
        std::cout << "  Analytical stationary error = " << std::scientific << err << "\n";
        check(err < 1e-13, "Formula314Chebyshev matches exact analytical exp(-i H0 T)");
    }

    // Тест 2: Сверка 3 методов матричной экспоненты (Chebyshev vs Pade vs Taylor)
    void test_expm_consistency() {
        const int N = 6;
        CMatrix Omega(N * N, Complex(0.0, 0.0));

        // Строим антиэрмитову матрицу Omega = -i * H
        for (int i = 0; i < N; ++i) {
            Omega[i * N + i] = Complex(0.0, -0.4 * (i + 1));
            if (i + 1 < N) {
                Omega[i * N + (i + 1)] = Complex(0.15, -0.25);
                Omega[(i + 1) * N + i] = Complex(-0.15, -0.25);
            }
        }

        CMatrix E_cheb = matrix_ops::expm_cheb(Omega, N, 18);
        CMatrix E_pade = matrix_ops::expm_pade_eigen(Omega, N);
        CMatrix E_tayl = matrix_ops::expm_taylor(Omega, N, 30);

        double diff_cp = frobenius_diff(E_cheb, E_pade);
        double diff_ct = frobenius_diff(E_cheb, E_tayl);
        double u_def = unitarity_defect(E_cheb, N);

        std::cout << "  ||Cheb - Pade||_F = " << diff_cp
            << ", ||Cheb - Taylor||_F = " << diff_ct
            << ", Unitarity defect = " << u_def << "\n";

        check(diff_cp < 1e-13 && diff_ct < 1e-13 && u_def < 1e-14,
            "Expm implementations (Chebyshev, PadeEigen, Taylor) agree to machine precision");
    }

    // Тест 3: Некоммутирующая задача + проверка 6-го порядка сходимости
    void test_driven_noncommuting_and_order() {
        const int N = 4;
        const double eps = 0.7;
        const double omega = 4.0;
        const double T = 1.5;

        CMatrix H0(N * N, Complex(0.0, 0.0));
        CMatrix H1(N * N, Complex(0.0, 0.0));
        for (int i = 0; i < N; ++i) {
            H0[i * N + i] = Complex(1.2 * (i - 1.5), 0.0);
            if (i + 1 < N) {
                H1[i * N + (i + 1)] = Complex(0.8, 0.3);
                H1[(i + 1) * N + i] = Complex(0.8, -0.3);
            }
        }

        utils::PeriodicHamiltonian ham;
        ham.dimension = N;
        ham.H0 = H0;
        ham.H1 = H1;
        ham.epsilon = eps;
        ham.omega = omega;
        ham.hbar = 1.0;

        single_cpu::Formula314ChebyshevOptions opts;
        opts.chebyshev_degree = 24;

        // Эталон на мелкой сетке
        CMatrix U_ref = single_cpu::solve_formula314_chebyshev(ham, 0.0, T, 2000, opts);

        // Сравнение с PiecewiseMagnus 6-го порядка на мелкой сетке
        CMatrix U_acc6 = piecewise_magnus_solver(0.0, T, T / 500.0, N, H0, H1, eps, omega, 6);
        double cross_solver_err = frobenius_diff(U_ref, U_acc6);
        std::cout << "  ||Formula314 - PiecewiseACC6||_F = " << cross_solver_err << "\n";
        check(cross_solver_err < 1e-12, "Formula314 and PiecewiseACC6 converge to identical propagator");

        // Эмпирический порядок сходимости p = log2(E(10) / E(20))
        CMatrix U_10 = single_cpu::solve_formula314_chebyshev(ham, 0.0, T, 10, opts);
        CMatrix U_20 = single_cpu::solve_formula314_chebyshev(ham, 0.0, T, 20, opts);
        double err_10 = frobenius_diff(U_10, U_ref);
        double err_20 = frobenius_diff(U_20, U_ref);
        double empirical_order = std::log2(err_10 / err_20);
        double u_def = unitarity_defect(U_20, N);

        std::cout << "  err(10) = " << err_10 << ", err(20) = " << err_20
            << ", Empirical order p = " << std::fixed << std::setprecision(2) << empirical_order
            << ", Unitarity defect = " << std::scientific << u_def << "\n";

        check(empirical_order > 5.5, "Formula314 achieves ~6th order convergence (p > 5.5)");
        check(u_def < 1e-13, "Propagator preserves unitarity (< 1e-13)");

#if defined(MAGNUS_HAS_CUDA)
        cuda_optimized::MagnusChebyshevGPU gpu_solver(N, H0, H1);
        CMatrix U_gpu = gpu_solver.solve(0.0, T, T / 200.0, eps, omega, 20);
        double gpu_err = frobenius_diff(U_gpu, U_ref);
        std::cout << "  ||GPU - CPU_ref||_F = " << gpu_err << "\n";
        check(gpu_err < 1e-12, "CUDA MagnusChebyshevGPU matches CPU reference");
#endif
    }

} // namespace

int main() {
    std::cout << "=== Native C++ Magnus Solver Verification ===\n";
    test_analytical_stationary_limit();
    test_expm_consistency();
    test_driven_noncommuting_and_order();
    std::cout << "=== ALL NATIVE C++ TESTS PASSED ===\n";
    return 0;
}