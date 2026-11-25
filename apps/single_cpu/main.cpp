#if 0

#include <iomanip>
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <mkl.h>
#include <omp.h>
#include "AllComputationalMethods.h"
#include "RungeKutt.h"

using namespace std;
using namespace utils;
using namespace matrix_ops;
using namespace runge_kutta_simple;

using complexd = complex<double>;
using CMatrix = vector<complexd>;

const double PI = 3.14159265358979323846;

// Вводим константы для экспериментов
const int N = 128;              // Размерность системы
const int NUM_ITERS = 4;        // Количество итераций для усреднения времени (warm-up + bench)

int main() {
    mkl_set_dynamic(0);
    omp_set_dynamic(0);

    mkl_set_num_threads(6);
    omp_set_num_threads(6);

    const double dt = 1e-5;
    const double T = 1.0;
    const double diag_min = 10.0;

    // Вводим параметры динамического H(t)
    const double W = 2.0 * PI; // 1 Hz
    const double eps0 = 1.0;

    std::cout << "Generating test matrices N=" << N << "...\n";

    // H(t) = H0_test + Hmod_test * cos(W*t)
    CMatrix H0_test = utils::generate_hermitian_matrix(N, diag_min);
    CMatrix Hmod_test = utils::generate_hermitian_matrix(N, diag_min / 2.0);

    std::vector<double> condition_numbers = { 10.0, 100.0, 1000.0, 10000.0 };
    std::vector<CMatrix> spectral_matrices;
    std::vector<CMatrix> spectral_V_matrices;
    std::vector<CMatrix> spectral_D_matrices;

    // Второе слагаемое для спектральных тестов (чтобы H(t) зависело от времени)
    CMatrix Hmod_spectral = utils::generate_hermitian_matrix(N, 1.0);

    for (size_t i = 0; i < condition_numbers.size(); ++i) {
        double kappa = condition_numbers[i];
        auto [H, V, D] = utils::generate_spectral_matrix(N, kappa, 3.0);
        spectral_matrices.push_back(H);
        spectral_V_matrices.push_back(V);
        spectral_D_matrices.push_back(D);
    }

    std::ofstream results_txt(utils::results_path("magnus_experiment_results.txt"));
    results_txt << std::scientific << std::setprecision(12);
    std::cout << std::scientific << std::setprecision(12);

    // =====================================================================
    // EXPERIMENT 1: Omega Accuracy Comparison (H(t) dependent)
    // =====================================================================
    const double T_int = dt;
    const double integration_dt = dt * 0.01; // НА 4 ПОРЯДКА МЕНЬШЕ

    // ТАК КАК НЕТ АНАЛИТИЧЕСКОГО РЕШЕНИЯ ДЛЯ [A,B]!=0, используем сверхточный Магнус 10 порядка как Эталон
    std::cout << "Computing Fine-Grid Reference Omega for Exp 1...\n";
    CMatrix Omega_ref_dt = magnus_expansion(0.0, T_int, integration_dt * 0.1, N, H0_test, Hmod_test, eps0, W, 10);

    vector<CMatrix> A_samples = generate_samples(0.0, T_int, integration_dt, N, H0_test, Hmod_test, eps0, W);
    CMatrix H0_anti_herm = matrix_ops::mat_scale(H0_test, complexd(0.0, -1.0));
    CMatrix Hmod_anti_herm = matrix_ops::mat_scale(Hmod_test, complexd(0.0, -1.0));

    results_txt << "=================================================================\n";
    results_txt << "EXPERIMENT 1: Comparison of Omega Matrix Calculation Accuracy (H(t) = A + B cos(Wt))\n";
    results_txt << "Reference: Recursive Magnus 10th order on fine grid\n";
    results_txt << "-----------------------------------------------------------------\n";
    results_txt << "Method | K | Max Element Diff | Max Eig Diff | Time (ms)\n";
    results_txt << "-----------------------------------------------------------------\n";

    // 1. Classic Magnus Expansion
    for (int k = 1; k <= 2; ++k) {
        CMatrix Omega_calc(N * N);
        auto start = chrono::high_resolution_clock::now();
        for (int _iter = 0; _iter < NUM_ITERS; ++_iter) {
            Omega_calc = magnus_classic(0.0, T_int, integration_dt, N, H0_test, Hmod_test, eps0, W, k);
        }
        auto end = chrono::high_resolution_clock::now();
        double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0 / NUM_ITERS;

        CMatrix diff(N * N, complexd(0.0, 0.0));
        matrix_ops::mat_sub(Omega_calc, Omega_ref_dt, diff, N);
        results_txt << "Classic | " << k << " | " << matrix_ops::max_element_diff(Omega_calc, Omega_ref_dt, N)
            << " | " << matrix_ops::max_eigenvalue_modulus_hermitian(diff, N) << " | " << time_ms << "\n";
    }

    // 2. ACC Expansion
    for (int k = 1; k <= 5; ++k) {
        CMatrix Omega_calc(N * N);
        auto start = chrono::high_resolution_clock::now();
        for (int _iter = 0; _iter < NUM_ITERS; ++_iter) {
            Omega_calc = magnus_ACC(A_samples, integration_dt, N, k);
        }
        auto end = chrono::high_resolution_clock::now();
        double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0 / NUM_ITERS;

        CMatrix diff(N * N, complexd(0.0, 0.0));
        matrix_ops::mat_sub(Omega_calc, Omega_ref_dt, diff, N);
        results_txt << "ACC | " << k << " | " << matrix_ops::max_element_diff(Omega_calc, Omega_ref_dt, N)
            << " | " << matrix_ops::max_eigenvalue_modulus_hermitian(diff, N) << " | " << time_ms << "\n";
    }

    // 3. Recursive Expansion
    for (int k = 1; k <= 8; ++k) {
        CMatrix Omega_calc(N * N);
        auto start = chrono::high_resolution_clock::now();
        for (int _iter = 0; _iter < NUM_ITERS; ++_iter) {
            Omega_calc = magnus_expansion(0.0, T_int, integration_dt, N, H0_test, Hmod_test, eps0, W, k);
        }
        auto end = chrono::high_resolution_clock::now();
        double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0 / NUM_ITERS;

        CMatrix diff(N * N, complexd(0.0, 0.0));
        matrix_ops::mat_sub(Omega_calc, Omega_ref_dt, diff, N);
        results_txt << "Recursive | " << k << " | " << matrix_ops::max_element_diff(Omega_calc, Omega_ref_dt, N)
            << " | " << matrix_ops::max_eigenvalue_modulus_hermitian(diff, N) << " | " << time_ms << "\n";
    }

    // 4. Formula 3.14
    {
        CMatrix Omega_314(N * N);
        auto start = chrono::high_resolution_clock::now();
        for (int _iter = 0; _iter < NUM_ITERS; ++_iter) {
            Omega_314 = magnus_3_14(0.0, T_int, integration_dt, N, H0_anti_herm, Hmod_anti_herm, eps0, W);
        }
        auto end = chrono::high_resolution_clock::now();
        double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0 / NUM_ITERS;

        CMatrix diff(N * N, complexd(0.0, 0.0));
        matrix_ops::mat_sub(Omega_314, Omega_ref_dt, diff, N);
        results_txt << "Formula 3.14 | 4 | " << matrix_ops::max_element_diff(Omega_314, Omega_ref_dt, N)
            << " | " << matrix_ops::max_eigenvalue_modulus_hermitian(diff, N) << " | " << time_ms << "\n";
    }

    // =====================================================================
    // EXPERIMENT 2: Matrix Exponentiation Methods for Dynamic Omega
    // =====================================================================
    {
        std::ofstream exp2_results(utils::results_path("experiment2_results.txt"));
        exp2_results << std::scientific << std::setprecision(12);
        std::cout << "\nRunning Experiment 2: Matrix Exponentiation Comparison...\n";

        exp2_results << "=================================================================\n";
        exp2_results << "EXPERIMENT 2: Comparison of exp(Omega) calculation accuracy\n";
        exp2_results << "Omega is taken from Fine-Grid Magnus (Exp 1)\n";
        exp2_results << "Reference: Eigen-based Pade approximation (expm_pade_eigen)\n";
        exp2_results << "-----------------------------------------------------------------\n";
        exp2_results << "Method | Parameters | Max Element Diff | Max Eig Diff | Time (ms)\n";
        exp2_results << "-----------------------------------------------------------------\n";

        CMatrix U_ref = matrix_ops::expm_pade_eigen(Omega_ref_dt, N);

        // 1. Taylor Expansion
        std::vector<int> taylor_terms = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30 };
        for (int terms : taylor_terms) {
            CMatrix U_calc(N * N);
            auto start = chrono::high_resolution_clock::now();
            for (int _iter = 0; _iter < NUM_ITERS; ++_iter) {
                U_calc = matrix_ops::expm_taylor(Omega_ref_dt, N, terms);
            }
            auto end = chrono::high_resolution_clock::now();
            double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0 / NUM_ITERS;

            CMatrix diff(N * N, complexd(0.0, 0.0));
            matrix_ops::mat_sub(U_calc, U_ref, diff, N);
            exp2_results << "Taylor | terms=" << terms << " | " << matrix_ops::max_element_diff(U_calc, U_ref, N)
                << " | " << matrix_ops::max_eigenvalue_modulus_hermitian(diff, N) << " | " << time_ms << "\n";
        }

        // 2. Chebyshev Expansion
        std::vector<int> cheb_terms = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 25 };
        for (int M : cheb_terms) {
            CMatrix U_calc(N * N);
            auto start = chrono::high_resolution_clock::now();
            for (int _iter = 0; _iter < NUM_ITERS; ++_iter) {
                U_calc = matrix_ops::expm_cheb(Omega_ref_dt, N, M);
            }
            auto end = chrono::high_resolution_clock::now();
            double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0 / NUM_ITERS;

            CMatrix diff(N * N, complexd(0.0, 0.0));
            matrix_ops::mat_sub(U_calc, U_ref, diff, N);
            exp2_results << "Chebyshev | M=" << M << " | " << matrix_ops::max_element_diff(U_calc, U_ref, N)
                << " | " << matrix_ops::max_eigenvalue_modulus_hermitian(diff, N) << " | " << time_ms << "\n";
        }

        // 3. Стандартный Pade
        {
            CMatrix U_calc(N * N);
            auto start = chrono::high_resolution_clock::now();
            for (int _iter = 0; _iter < NUM_ITERS; ++_iter) {
                U_calc = matrix_ops::expm_pade_eigen(Omega_ref_dt, N);
            }
            auto end = chrono::high_resolution_clock::now();
            double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0 / NUM_ITERS;
            exp2_results << "Eigen-Pade | -- | 0.000000e+00 | 0.000000e+00 | " << time_ms << "\n";
        }

        exp2_results.close();
        std::cout << "Experiment 2 results written to experiment2_results.txt\n";
    }

    // Эталонная матрица для Экспериментов 3, 3b и 7
    CMatrix U_exact = utils::eye(N);

    // =====================================================================
    // EXPERIMENT 3: ODE Solving U'(t) = -iH(t)*U(t)
    // =====================================================================
    {
        std::ofstream exp3_results(utils::results_path("experiment3_results.txt"));
        exp3_results << std::scientific << std::setprecision(12);
        std::cout << "\nRunning Experiment 3: ODE Solving Comparison (H(t) dependent)...\n";

        // ГЕНЕРИРУЕМ ЭТАЛОН: Сверхточный РК4 (шаг 1e-6)
        double fine_rk_dt = 1e-6;
        int exact_steps = (int)std::round(T / fine_rk_dt);

        std::cout << "Computing Exact U(T) via fine RK4 (dt=1e-6)...\n";
        auto start_exact = chrono::high_resolution_clock::now();
        // RK4 не усредняем в цикле
        for (int step = 0; step < exact_steps; ++step) {
            U_exact = runge_kutta_simple::runge_kutta_step(step * fine_rk_dt, H0_test, Hmod_test, W, U_exact, fine_rk_dt, N);
        }
        auto end_exact = chrono::high_resolution_clock::now();
        std::cout << "Done in " << chrono::duration_cast<chrono::microseconds>(end_exact - start_exact).count() / 1000.0 << " ms\n";

        // Method 1: Standard Runge-Kutta
        {
            double rk_dt = 1e-4; // Обычный шаг
            int num_steps = (int)std::round(T / rk_dt);

            CMatrix U_rk = utils::eye(N);
            auto start_rk = chrono::high_resolution_clock::now();
            for (int step = 0; step < num_steps; ++step) {
                U_rk = runge_kutta_simple::runge_kutta_step(step * rk_dt, H0_test, Hmod_test, W, U_rk, rk_dt, N);
            }
            auto end_rk = chrono::high_resolution_clock::now();
            double time_rk_ms = chrono::duration_cast<chrono::microseconds>(end_rk - start_rk).count() / 1000.0; // RK не усредняем

            CMatrix diff_rk(N * N, complexd(0.0, 0.0));
            matrix_ops::mat_sub(U_rk, U_exact, diff_rk, N);
            exp3_results << "Runge-Kutta dt=1e-4 | " << matrix_ops::max_element_diff(U_rk, U_exact, N) << " | " << time_rk_ms << "\n";
        }

        // Method 2: Piecewise Magnus ACC + Chebyshev
        {
            double macro_dt = 0.01;
            CMatrix U_magnus(N * N);
            auto start_magnus = chrono::high_resolution_clock::now();
            for (int _iter = 0; _iter < NUM_ITERS; ++_iter) {
                U_magnus = piecewise_magnus_solver(0.0, T, macro_dt, N, H0_test, Hmod_test, eps0, W, 4);
            }
            auto end_magnus = chrono::high_resolution_clock::now();
            double time_magnus_ms = chrono::duration_cast<chrono::microseconds>(end_magnus - start_magnus).count() / 1000.0 / NUM_ITERS;

            CMatrix diff_magnus(N * N, complexd(0.0, 0.0));
            matrix_ops::mat_sub(U_magnus, U_exact, diff_magnus, N);

            exp3_results << "Piecewise Magnus ACC | " << matrix_ops::max_element_diff(U_magnus, U_exact, N) << " | " << time_magnus_ms << "\n";
        }
        exp3_results.close();
    }

    // =====================================================================
    // EXPERIMENT 3b: Work-Precision Diagram
    // =====================================================================
    {
        std::ofstream exp3b_results(utils::results_path("experiment3b_work_precision.txt"));
        exp3b_results << std::scientific << std::setprecision(12);
        std::cout << "\nRunning Experiment 3b: Work-Precision Diagram (RK4 vs Magnus)...\n";
        exp3b_results << "Method | Step | Error | Time(ms)\n";
        exp3b_results << "------------------------------------------------\n";

        // Прогоняем RK4 с разными шагами
        std::vector<double> rk_steps = { 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5 };
        for (double s_dt : rk_steps) {
            int num_steps = (int)std::round(T / s_dt);
            CMatrix U_rk = utils::eye(N);
            
            auto start = chrono::high_resolution_clock::now();
            for (int step = 0; step < num_steps; ++step) {
                U_rk = runge_kutta_simple::runge_kutta_step(step * s_dt, H0_test, Hmod_test, W, U_rk, s_dt, N);
            }
            auto end = chrono::high_resolution_clock::now();
            double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;

            double err = matrix_ops::max_element_diff(U_rk, U_exact, N);
            exp3b_results << "RK4 | " << s_dt << " | " << err << " | " << time_ms << "\n";
        }

        // Прогоняем Magnus с разными макро-шагами
        std::vector<double> mag_steps = { 2e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3 };
        for (double m_dt : mag_steps) {
            CMatrix U_magnus(N * N);
            
            auto start = chrono::high_resolution_clock::now();
            for (int _iter = 0; _iter < NUM_ITERS; ++_iter) {
                U_magnus = piecewise_magnus_solver(0.0, T, m_dt, N, H0_test, Hmod_test, eps0, W, 4);
            }
            auto end = chrono::high_resolution_clock::now();
            double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0 / NUM_ITERS;

            double err = matrix_ops::max_element_diff(U_magnus, U_exact, N);
            exp3b_results << "Magnus | " << m_dt << " | " << err << " | " << time_ms << "\n";
        }
        
        exp3b_results.close();
        std::cout << "Experiment 3b results written to experiment3b_work_precision.txt\n";
    }

    // =====================================================================
    // ============= EXPERIMENT 4: Omega Stability Analysis ===============
    // =====================================================================
    {
        std::ofstream exp4_results(utils::results_path("experiment4_results.txt"));
        if (!exp4_results.is_open()) {
            std::cerr << "Error opening experiment4_results.txt" << std::endl;
            return 1;
        }
        exp4_results << std::scientific << std::setprecision(12);
        std::cout << "\nRunning Experiment 4: Omega Stability Analysis (H(t) dependent)...\n";

        exp4_results << "=================================================================\n";
        exp4_results << "EXPERIMENT 4: Omega Stability Analysis for Spectral Matrices (H(t) = A + B cos(Wt))\n";
        exp4_results << "N=" << N << ", dt=" << dt << "\n";
        exp4_results << "Reference: Recursive Magnus 10th order on fine grid\n";
        exp4_results << "-----------------------------------------------------------------\n";
        exp4_results << "Kappa | Method | Max Element Diff | Max Eig Diff | Time (ms)\n";
        exp4_results << "-----------------------------------------------------------------\n";

        for (size_t mat_idx = 0; mat_idx < spectral_matrices.size(); ++mat_idx) {
            double kappa = condition_numbers[mat_idx];
            const CMatrix& H_A = spectral_matrices[mat_idx];
            const CMatrix& H_B = Hmod_spectral;

            std::cout << "Computing Fine-Grid Reference Omega for kappa " << kappa << "...\n";

            CMatrix Omega_ref;
            for (int _iter = 0; _iter < NUM_ITERS; ++_iter) {
                Omega_ref = magnus_expansion(0.0, T_int, integration_dt * 0.1, N, H_A, H_B, eps0, W, 10);
            }

            vector<CMatrix> A_samples_k = generate_samples(0.0, T_int, integration_dt, N, H_A, H_B, eps0, W);
            CMatrix H_A_anti = matrix_ops::mat_scale(H_A, complexd(0.0, -1.0));
            CMatrix H_B_anti = matrix_ops::mat_scale(H_B, complexd(0.0, -1.0));

            // Classic Magnus (k=2)
            {
                CMatrix Omega_calc(N * N);
                auto start = chrono::high_resolution_clock::now();
                for (int _iter = 0; _iter < NUM_ITERS; ++_iter) {
                    Omega_calc = magnus_classic(0.0, T_int, integration_dt, N, H_A, H_B, eps0, W, 2);
                }
                auto end = chrono::high_resolution_clock::now();
                double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0 / NUM_ITERS;

                CMatrix diff(N * N, complexd(0.0, 0.0));
                matrix_ops::mat_sub(Omega_calc, Omega_ref, diff, N);
                exp4_results << kappa << " | Classic | " << matrix_ops::max_element_diff(Omega_calc, Omega_ref, N)
                    << " | " << matrix_ops::max_eigenvalue_modulus_hermitian(diff, N) << " | " << time_ms << "\n";
            }

            // ACC Magnus (k=5)
            {
                CMatrix Omega_calc(N * N);
                auto start = chrono::high_resolution_clock::now();
                for (int _iter = 0; _iter < NUM_ITERS; ++_iter) {
                    Omega_calc = magnus_ACC(A_samples_k, integration_dt, N, 5);
                }
                auto end = chrono::high_resolution_clock::now();
                double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0 / NUM_ITERS;

                CMatrix diff(N * N, complexd(0.0, 0.0));
                matrix_ops::mat_sub(Omega_calc, Omega_ref, diff, N);
                exp4_results << kappa << " | ACC | " << matrix_ops::max_element_diff(Omega_calc, Omega_ref, N)
                    << " | " << matrix_ops::max_eigenvalue_modulus_hermitian(diff, N) << " | " << time_ms << "\n";
            }

            // Recursive Magnus (k=8)
            {
                CMatrix Omega_calc(N * N);
                auto start = chrono::high_resolution_clock::now();
                for (int _iter = 0; _iter < NUM_ITERS; ++_iter) {
                    Omega_calc = magnus_expansion(0.0, T_int, integration_dt, N, H_A, H_B, eps0, W, 8);
                }
                auto end = chrono::high_resolution_clock::now();
                double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0 / NUM_ITERS;

                CMatrix diff(N * N, complexd(0.0, 0.0));
                matrix_ops::mat_sub(Omega_calc, Omega_ref, diff, N);
                exp4_results << kappa << " | Recursive | " << matrix_ops::max_element_diff(Omega_calc, Omega_ref, N)
                    << " | " << matrix_ops::max_eigenvalue_modulus_hermitian(diff, N) << " | " << time_ms << "\n";
            }

            // Formula 3.14
            {
                CMatrix Omega_calc(N * N);
                auto start = chrono::high_resolution_clock::now();
                for (int _iter = 0; _iter < NUM_ITERS; ++_iter) {
                    Omega_calc = magnus_3_14(0.0, T_int, integration_dt, N, H_A_anti, H_B_anti, eps0, W);
                }
                auto end = chrono::high_resolution_clock::now();
                double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0 / NUM_ITERS;

                CMatrix diff(N * N, complexd(0.0, 0.0));
                matrix_ops::mat_sub(Omega_calc, Omega_ref, diff, N);
                exp4_results << kappa << " | Formula 3.14 | " << matrix_ops::max_element_diff(Omega_calc, Omega_ref, N)
                    << " | " << matrix_ops::max_eigenvalue_modulus_hermitian(diff, N) << " | " << time_ms << "\n";
            }
        }
        exp4_results.close();
        std::cout << "Experiment 4 results written to experiment4_results.txt\n";
    }

    // =====================================================================
    // ============= EXPERIMENT 5: Matrix Exponent Stability ==============
    // =====================================================================
    {
        std::ofstream exp5_results(utils::results_path("experiment5_results.txt"));
        if (!exp5_results.is_open()) {
            std::cerr << "Error opening experiment5_results.txt" << std::endl;
            return 1;
        }
        exp5_results << std::scientific << std::setprecision(12);
        std::cout << "\nRunning Experiment 5: Matrix Exponent Stability Analysis...\n";

        exp5_results << "=================================================================\n";
        exp5_results << "EXPERIMENT 5: Exponentiation Stability for Dynamic Omega\n";
        exp5_results << "N=" << N << "\n";
        exp5_results << "Reference: Eigen Pade applied to Magnus-generated Omega\n";
        exp5_results << "-----------------------------------------------------------------\n";
        exp5_results << "Kappa | Method | Max Element Diff | Max Eig Diff | Time (ms)\n";
        exp5_results << "-----------------------------------------------------------------\n";

        for (size_t mat_idx = 0; mat_idx < spectral_matrices.size(); ++mat_idx) {
            double kappa = condition_numbers[mat_idx];
            const CMatrix& H_A = spectral_matrices[mat_idx];
            const CMatrix& H_B = Hmod_spectral;

            std::cout << "Testing matrix exponent stability for kappa " << kappa << "\n";

            vector<CMatrix> A_samp = generate_samples(0.0, T_int, integration_dt, N, H_A, H_B, eps0, W);
            CMatrix Omega_test = magnus_ACC(A_samp, integration_dt, N, 4);
            CMatrix U_ref = matrix_ops::expm_pade_eigen(Omega_test, N);

            // Taylor method (terms = 30)
            {
                CMatrix U_calc(N * N);
                auto start = chrono::high_resolution_clock::now();
                for (int _iter = 0; _iter < NUM_ITERS; ++_iter) {
                    U_calc = matrix_ops::expm_taylor(Omega_test, N, 30);
                }
                auto end = chrono::high_resolution_clock::now();
                double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0 / NUM_ITERS;

                CMatrix diff(N * N, complexd(0.0, 0.0));
                matrix_ops::mat_sub(U_calc, U_ref, diff, N);
                exp5_results << kappa << " | Taylor | " << matrix_ops::max_element_diff(U_calc, U_ref, N)
                    << " | " << matrix_ops::max_eigenvalue_modulus_hermitian(diff, N) << " | " << time_ms << "\n";
            }

            // Chebyshev method (M = 25)
            {
                CMatrix U_calc(N * N);
                auto start = chrono::high_resolution_clock::now();
                for (int _iter = 0; _iter < NUM_ITERS; ++_iter) {
                    U_calc = matrix_ops::expm_cheb(Omega_test, N, 25);
                }
                auto end = chrono::high_resolution_clock::now();
                double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0 / NUM_ITERS;

                CMatrix diff(N * N, complexd(0.0, 0.0));
                matrix_ops::mat_sub(U_calc, U_ref, diff, N);
                exp5_results << kappa << " | Chebyshev | " << matrix_ops::max_element_diff(U_calc, U_ref, N)
                    << " | " << matrix_ops::max_eigenvalue_modulus_hermitian(diff, N) << " | " << time_ms << "\n";
            }
        }
        exp5_results.close();
        std::cout << "Experiment 5 results written to experiment5_results.txt\n";
    }

    // =====================================================================
    // ============= EXPERIMENT 6: ODE Stability Analysis ================
    // =====================================================================
    {
        std::ofstream exp6_results(utils::results_path("experiment6_results.txt"));
        if (!exp6_results.is_open()) {
            std::cerr << "Error opening experiment6_results.txt" << std::endl;
            return 1;
        }
        exp6_results << std::scientific << std::setprecision(12);
        std::cout << "\nRunning Experiment 6: ODE Stability Analysis (H(t) dependent)...\n";

        exp6_results << "=================================================================\n";
        exp6_results << "EXPERIMENT 6: ODE Solution Stability Analysis for Spectral Matrices (H(t) = A + B cos(Wt))\n";
        exp6_results << "N=" << N << ", T=" << T << "\n";
        exp6_results << "Reference: Fine-grid RK4 (dt=1e-6)\n";
        exp6_results << "-----------------------------------------------------------------\n";
        exp6_results << "Kappa | Method | Max Element Diff | Max Eig Diff | Time (ms)\n";
        exp6_results << "-----------------------------------------------------------------\n";

        double fine_rk_dt = 1e-6;
        int exact_steps = (int)std::round(T / fine_rk_dt);

        for (size_t mat_idx = 0; mat_idx < spectral_matrices.size(); ++mat_idx) {
            double kappa = condition_numbers[mat_idx];
            const CMatrix& H_A = spectral_matrices[mat_idx];
            const CMatrix& H_B = Hmod_spectral;

            std::cout << "Computing Fine-Grid Reference U(T) for kappa " << kappa << "...\n";

            auto start_exact = chrono::high_resolution_clock::now();
            CMatrix U_exact_local = utils::eye(N);
            for (int step = 0; step < exact_steps; ++step) {
                U_exact_local = runge_kutta_simple::runge_kutta_step(step * fine_rk_dt, H_A, H_B, W, U_exact_local, fine_rk_dt, N);
            }
            auto end_exact = chrono::high_resolution_clock::now();
            std::cout << "Done in " << chrono::duration_cast<chrono::microseconds>(end_exact - start_exact).count() / 1000.0 << " ms\n";

            // Method 1: Runge-Kutta with normal dt = 1e-4
            {
                double rk_dt = 1e-4;
                int num_steps = (int)std::round(T / rk_dt);

                CMatrix U_rk = utils::eye(N);
                auto start_rk = chrono::high_resolution_clock::now();
                for (int step = 0; step < num_steps; ++step) {
                    U_rk = runge_kutta_simple::runge_kutta_step(step * rk_dt, H_A, H_B, W, U_rk, rk_dt, N);
                }
                auto end_rk = chrono::high_resolution_clock::now();
                double time_rk_ms = chrono::duration_cast<chrono::microseconds>(end_rk - start_rk).count() / 1000.0; // RK не усредняем

                CMatrix diff_rk(N * N, complexd(0.0, 0.0));
                matrix_ops::mat_sub(U_rk, U_exact_local, diff_rk, N);
                exp6_results << kappa << " | Runge-Kutta dt=1e-4 | " << matrix_ops::max_element_diff(U_rk, U_exact_local, N)
                    << " | " << matrix_ops::max_eigenvalue_modulus_hermitian(diff_rk, N) << " | " << time_rk_ms << "\n";
            }

            // Method 2: Piecewise Magnus ACC + Chebyshev
            {
                double macro_dt = 0.01;
                CMatrix U_magnus(N * N);
                auto start_magnus = chrono::high_resolution_clock::now();
                for (int _iter = 0; _iter < NUM_ITERS; ++_iter) {
                    U_magnus = piecewise_magnus_solver(0.0, T, macro_dt, N, H_A, H_B, eps0, W, 4);
                }
                auto end_magnus = chrono::high_resolution_clock::now();
                double time_magnus_ms = chrono::duration_cast<chrono::microseconds>(end_magnus - start_magnus).count() / 1000.0 / NUM_ITERS;

                CMatrix diff_magnus(N * N, complexd(0.0, 0.0));
                matrix_ops::mat_sub(U_magnus, U_exact_local, diff_magnus, N);

                exp6_results << kappa << " | Piecewise Magnus ACC | "
                    << matrix_ops::max_element_diff(U_magnus, U_exact_local, N) << " | "
                    << matrix_ops::max_eigenvalue_modulus_hermitian(diff_magnus, N) << " | "
                    << time_magnus_ms << "\n";
            }
        }
        exp6_results.close();
        std::cout << "Experiment 6 results written to experiment6_results.txt\n";
    }

    // =====================================================================
    // ============= EXPERIMENT 7: RK4 Accuracy vs dt ======================
    // =====================================================================
    {
        std::ofstream exp7_results(utils::results_path("experiment7_rk4_convergence.txt"));
        if (!exp7_results.is_open()) {
            std::cerr << "Error opening experiment7_rk4_convergence.txt" << std::endl;
            return 1;
        }
        exp7_results << std::scientific << std::setprecision(12);
        std::cout << "\nRunning Experiment 7: RK4 Accuracy vs dt...\n";

        exp7_results << "=================================================================\n";
        exp7_results << "EXPERIMENT 7: RK4 Accuracy vs Integration Step (dt)\n";
        exp7_results << "H(t) = H0 + Hmod * cos(W*t), N=" << N << ", T=" << T << "\n";
        exp7_results << "Reference: Piecewise Magnus Solver (macro_dt=1e-3, order=4)\n";
        exp7_results << "-----------------------------------------------------------------\n";
        exp7_results << "dt | Max Element Diff | Steps | Time (ms)\n";
        exp7_results << "-----------------------------------------------------------------\n";

        double macro_dt = 1e-3;
        std::cout << "Computing reference U(T) using piecewise_magnus_solver (Fast)...\n";
        CMatrix U_exact_magnus(N * N);
        
        auto start_exact = std::chrono::high_resolution_clock::now();
        for (int _iter = 0; _iter < NUM_ITERS; ++_iter) {
            U_exact_magnus = piecewise_magnus_solver(0.0, T, macro_dt, N, H0_test, Hmod_test, eps0, W, 4);
        }
        auto end_exact = std::chrono::high_resolution_clock::now();
        std::cout << "Done in " << std::chrono::duration_cast<std::chrono::microseconds>(end_exact - start_exact).count() / 1000.0 / NUM_ITERS << " ms (Avg)\n\n";

        double current_dt = 1e-3;
        double prev_error = 1e300; 

        while (current_dt >= 1e-8) {
            int steps = (int)std::round(T / current_dt);
            CMatrix U_rk = utils::eye(N);
            
            auto start = std::chrono::high_resolution_clock::now();
            for (int step = 0; step < steps; ++step) {
                U_rk = runge_kutta_simple::runge_kutta_step(step * current_dt, H0_test, Hmod_test, W, U_rk, current_dt, N);
            }
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

            // Сравниваем результат РК4 с эталоном Магнуса
            double current_error = matrix_ops::max_element_diff(U_rk, U_exact_magnus, N);

            exp7_results << current_dt << " | " << current_error << " | " << steps << " | " << time_ms << "\n";
            std::cout << "dt: " << std::setw(12) << current_dt << " | Error: " << current_error << "\n";

            if (current_error > prev_error && current_error < 1.0) {
                std::cout << "-> Stopping: Accuracy plateau or degradation reached due to floating-point round-off limits.\n";
                break;
            }

            prev_error = current_error;
            current_dt /= 2.0;
        }
        exp7_results.close();
        std::cout << "Experiment 7 results written to experiment7_rk4_convergence.txt\n";
    }

    // =====================================================================
    // ============= EXPERIMENT 8: Magnus macro_dt Self-Convergence ========
    // =====================================================================
    {
        std::ofstream exp8_results(utils::results_path("experiment8_macro_dt_self_conv.txt"));
        if (!exp8_results.is_open()) {
            std::cerr << "Error opening experiment8_macro_dt_self_conv.txt" << std::endl;
            return 1;
        }
        exp8_results << std::scientific << std::setprecision(12);
        std::cout << "\nRunning Experiment 8: Piecewise Magnus Self-Convergence vs macro_dt...\n";

        double inner_dt = 1e-6;          
        double current_macro_dt = 1e-1;  
        double tolerance = 1e-9;         

        CMatrix U_prev = utils::eye(N);  
        bool is_first_iteration = true;
        double prev_error = 1e300;

        exp8_results << "=================================================================\n";
        exp8_results << "EXPERIMENT 8: Piecewise Magnus Self-Convergence vs macro_dt\n";
        exp8_results << "Fixed inner integration step: " << inner_dt << "\n";
        exp8_results << "Convergence tolerance: " << tolerance << "\n";
        exp8_results << "-----------------------------------------------------------------\n";
        exp8_results << "macro_dt | Diff with prev step | Steps | Time (ms)\n";
        exp8_results << "-----------------------------------------------------------------\n";

        while (current_macro_dt >= inner_dt) {
            int steps = (int)std::round(T / current_macro_dt);
            CMatrix U_curr(N * N);

            auto start = std::chrono::high_resolution_clock::now();
            for (int _iter = 0; _iter < NUM_ITERS; ++_iter) {
                U_curr = utils::eye(N);
                for (int step = 0; step < steps; ++step) {
                    double t_curr = step * current_macro_dt;
                    double t_next = t_curr + current_macro_dt;
                    vector<CMatrix> step_samples = generate_samples(t_curr, t_next, inner_dt, N, H0_test, Hmod_test, eps0, W);
                    CMatrix Omega_local = magnus_ACC(step_samples, inner_dt, N, 4);
                    CMatrix U_local = matrix_ops::expm_cheb(Omega_local, N, 15);

                    CMatrix temp((size_t)N * N, complexd(0.0, 0.0));
                    matrix_ops::matmul(U_local, U_curr, temp, N);
                    U_curr = temp;
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 / NUM_ITERS;

            if (!is_first_iteration) {
                double diff = matrix_ops::max_element_diff(U_curr, U_prev, N);

                exp8_results << current_macro_dt << " | " << diff << " | " << steps << " | " << time_ms << "\n";
                std::cout << "macro_dt: " << std::setw(12) << current_macro_dt << " | Diff: " << diff << "\n";

                if (diff < tolerance) {
                    std::cout << "-> Converged! Accuracy tolerance reached.\n";
                    break;
                }
                if (diff > prev_error && diff < 1.0) {
                    std::cout << "-> Stopping: Accuracy plateau reached.\n";
                    break;
                }
                prev_error = diff;
            }
            else {
                exp8_results << current_macro_dt << " | N/A | " << steps << " | " << time_ms << "\n";
                std::cout << "macro_dt: " << std::setw(12) << current_macro_dt << " | Diff: N/A\n";
                is_first_iteration = false;
            }

            U_prev = U_curr;
            current_macro_dt /= 2.0; 
        }
        exp8_results.close();
        std::cout << "Experiment 8 results written to experiment8_macro_dt_self_conv.txt\n";
    }

    // =====================================================================
    // ============= EXPERIMENT 9: Adaptive Inner Integration ==============
    // =====================================================================
    {
        const string exp9_results_path = utils::results_path("experiment9_adaptive_inner.txt");
        std::ofstream exp9_results(exp9_results_path);
        if (!exp9_results.is_open()) {
            std::cerr << "Error opening " << exp9_results_path << "\n";
            return 1;
        }
        exp9_results << std::scientific << std::setprecision(12);
        std::cout << "\nRunning Experiment 9: Adaptive Inner Integration (Segment Refinement)...\n";

        double tolerance = 1e-12;
        int num_segments = 20;   
        double macro_dt = T / num_segments;
        CMatrix U_total = utils::eye(N);

        exp9_results << "=================================================================\n";
        exp9_results << "EXPERIMENT 9: Adaptive Inner Integration\n";
        exp9_results << "Total time T = " << T << ", Segments = " << num_segments << ", macro_dt = " << macro_dt << "\n";
        exp9_results << "Convergence tolerance = " << tolerance << "\n";
        exp9_results << "-----------------------------------------------------------------\n";
        exp9_results << "Segment | t_start | t_end | Converged inner_dt | Halvings | Time (ms)\n";
        exp9_results << "-----------------------------------------------------------------\n";

        auto start_total = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_segments; ++i) {
            double t_start = i * macro_dt;
            double t_end = t_start + macro_dt;

            double time_ms_seg = 0.0;
            int final_halving_count = 0;
            double final_inner_dt = 0.0;
            CMatrix final_U_local(N * N);

            std::cout << "Segment " << i + 1 << "/" << num_segments << " [" << t_start << ", " << t_end << "]:\n";

            // Усредняем работу алгоритма спуска по дереву адаптации для точности замеров
            for (int _iter = 0; _iter < NUM_ITERS; ++_iter) {
                double current_inner_dt = macro_dt;
                CMatrix U_local_old = utils::eye(N);
                CMatrix U_local_new = utils::eye(N);

                bool converged = false;
                int halving_count = 0;

                auto start_seg = std::chrono::high_resolution_clock::now();

                while (!converged && halving_count < 25) { 
                    vector<CMatrix> samples = generate_samples(t_start, t_end, current_inner_dt, N, H0_test, Hmod_test, eps0, W);
                    CMatrix Omega = magnus_ACC(samples, current_inner_dt, N, 4);
                    U_local_new = matrix_ops::expm_cheb(Omega, N, 15);

                    if (halving_count > 0) {
                        double diff = matrix_ops::max_element_diff(U_local_new, U_local_old, N);
                        if (diff < tolerance) {
                            converged = true;
                            if (_iter == 0) std::cout << "  -> Converged at inner_dt: " << std::setw(12) << current_inner_dt << " (diff: " << diff << ")\n";
                        }
                    }

                    U_local_old = U_local_new;
                    if (!converged) {
                        current_inner_dt /= 2.0; 
                        halving_count++;
                    }
                }

                auto end_seg = std::chrono::high_resolution_clock::now();
                time_ms_seg += std::chrono::duration_cast<std::chrono::microseconds>(end_seg - start_seg).count() / 1000.0;

                if (_iter == NUM_ITERS - 1) {
                    final_halving_count = halving_count;
                    final_inner_dt = current_inner_dt;
                    final_U_local = U_local_new;
                }
            }
            time_ms_seg /= NUM_ITERS;

            exp9_results << i + 1 << " | " << t_start << " | " << t_end << " | "
                << final_inner_dt << " | " << final_halving_count << " | " << time_ms_seg << "\n";

            CMatrix temp((size_t)N * N, complexd(0.0, 0.0));
            matrix_ops::matmul(final_U_local, U_total, temp, N);
            U_total = temp;
        }

        auto end_total = std::chrono::high_resolution_clock::now();
        double time_ms_total = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total).count() / 1000.0;

        exp9_results << "-----------------------------------------------------------------\n";
        exp9_results << "Total time for adaptive Magnus: " << time_ms_total << " ms\n";
        exp9_results.close();

        std::cout << "Experiment 9 finished. Total time: " << time_ms_total << " ms\n";
        std::cout << "Results written to " << exp9_results_path << "\n";
    }

}

#endif

#if 0

#include "AllComputationalMethods.h"
#include "Utils.h"
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
using namespace matrix_ops;

int main() {
    cout << "=== Starting HPC Roofline Benchmark ===" << endl;

    // Параметры системы
    int N = 2048; // Достаточно большой N, чтобы загрузить кэш и ядра
    double t0 = 0.0;
    double t1 = 0.1;      // Отрезок интегрирования
    double dt = 0.01;    // Шаг (10 сэмплов для ACC/Recursive)
    double eps0 = 0.5;
    double W = 2.0;
    int num_runs = 3;    // Количество прогонов каждого метода

    cout << "Matrix size: " << N << "x" << N << endl;
    cout << "Number of runs per method: " << num_runs << endl;

    // 1. Инициализация (используем функции из Utils)
    CMatrix H0 = utils::generate_hermitian_matrix(N, 1.0);
    CMatrix H_mod = utils::generate_hermitian_matrix(N, 1.0);

    // Подготовка сэмплов для ACC и Recursive
    vector<CMatrix> samples = generate_samples(t0, t1, dt, N, H0, H_mod, eps0, W);

    // =======================================================
    // Тест 1: ACC Magnus (до 6 порядка)
    // =======================================================
    cout << "\nRunning ACC Magnus (Order 6)..." << endl;
    CMatrix Omega_ACC(N * N);
    for (int i = 0; i < num_runs; ++i) {
        Omega_ACC = magnus_ACC(samples, dt, N, 6);
    }

    // =======================================================
    // Тест 2: Formula 3.14 (4 порядок)
    // =======================================================
    cout << "Running Formula 3.14..." << endl;
    CMatrix Omega_314(N * N);
    for (int i = 0; i < num_runs * 10; ++i) { // Крутим больше раз, т.к. она очень быстрая
        Omega_314 = magnus_3_14(t0, t1, t1 - t0, N, H0, H_mod, eps0, W);
    }

    // =======================================================
    // Тест 3: Recursive Magnus (например, 4 порядка)
    // =======================================================
    cout << "Running Recursive Magnus (Order 4)..." << endl;
    CMatrix Omega_Rec(N * N);
    for (int i = 0; i < num_runs; ++i) {
        Omega_Rec = magnus_expansion(t0, t1, dt, N, H0, H_mod, eps0, W, 4);
    }

    // =======================================================
    // Тест 4: Chebyshev Matrix Exponentiation
    // =======================================================
    cout << "Running Chebyshev Expm..." << endl;
    CMatrix U_Cheb(N * N);
    for (int i = 0; i < num_runs * 5; ++i) {
        // Считаем экспоненту от Omega_ACC (M=15 членов ряда)
        U_Cheb = expm_cheb(Omega_ACC, N, 15); 
    }

    cout << "\n=== Benchmark Finished Successfully ===" << endl;
    return 0;
}

#endif

#if 1

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>
#include <chrono>
#include <limits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mkl.h>
#include <omp.h>
#include <clocale>

#include "AllComputationalMethods.h"
#include "FlopCounter.h"

using namespace std;
using namespace utils;
using namespace matrix_ops;

namespace {

    constexpr double PI = 3.14159265358979323846;

    // -------------------------------------------------------------------------
    // Глобальные параметры эксперимента
    // -------------------------------------------------------------------------
    constexpr int N = 256;               // Размерность матриц N x N
    constexpr int NUM_ITERS = 3;         // Число прогонов для усреднения времени/FLOPs
    constexpr double T_TOTAL = 1e-3;     // Фиксированный отрезок интегрирования [0, 0.001]
    constexpr double EPS0 = 1.0;         // Амплитуда модуляции f(t) = EPS0 * cos(W * t)
    constexpr double W = 2.0 * PI;       // Частота модуляции (период T = 1.0)

    // Параметры эталонного расчета
    constexpr double REF_DT = T_TOTAL * 1e-3; // 1e-6 (1000 интервалов)
    constexpr int REF_MAGNUS_ORDER = 10;
    constexpr int EXACT_CHEB_ORDER = 32;      // Машинно-точная экспонента для изоляции ошибки Omega

    // Внутреннее разбиение одного шага [t_s, t_{s+1}] для Classic / Recursive / Formula314
    // 6 подынтервалов = 7 узлов (равные условия с ACC 6 и точная 4-го порядка ветка Симпсона)
    constexpr int INNER_SUB_DEFAULT = 6;

    // Точные целые числа шагов K на [0, 0.001]: dt = 0.001 / K (от 1e-3 до 1e-6)
    const vector<int> STEP_COUNTS = { 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000 };

    // Порядки разложения Чебышева для Части 2
    const vector<int> CHEB_ORDERS = { 2, 3, 4, 6, 8, 10, 12, 15, 20 };

    // -------------------------------------------------------------------------
    // Описание тестовой задачи и бинарного кэша эталонов
    // -------------------------------------------------------------------------
    struct BenchmarkProblem {
        string name;
        CMatrix H0;
        CMatrix Hmod;
        CMatrix H0_anti;   // -i * H0 (для magnus_3_14)
        CMatrix Hmod_anti; // -i * Hmod (для magnus_3_14)
        CMatrix Omega_ref; // Эталон Omega(0, T_TOTAL)
        CMatrix U_ref;     // Эталон U(T_TOTAL) = exp(Omega_ref)
    };

    struct ReferenceCacheHeader {
        uint64_t magic = 0x4D41474E55535246ULL; // "MAGNUSRF"
        uint32_t version = 3;                   // v3: Симпсон в эталоне + скорректированная 2_HighNonComm_DD
        int32_t N = 0;
        int32_t ref_order = 0;
        int32_t num_problems = 0;
        double T_total = 0.0;
        double eps0 = 0.0;
        double W = 0.0;
        double ref_dt = 0.0;
    };

    bool write_matrix_bin(ofstream& out, const CMatrix& mat) {
        const uint64_t count = static_cast<uint64_t>(mat.size());
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        out.write(reinterpret_cast<const char*>(mat.data()), static_cast<std::streamsize>(count * sizeof(complexd)));
        return out.good();
    }

    bool read_matrix_bin(ifstream& in, CMatrix& mat, size_t expected_count) {
        uint64_t count = 0;
        in.read(reinterpret_cast<char*>(&count), sizeof(count));
        if (!in.good() || count != expected_count) return false;
        mat.resize(expected_count);
        in.read(reinterpret_cast<char*>(mat.data()), static_cast<std::streamsize>(count * sizeof(complexd)));
        return in.good();
    }

    bool save_reference_cache(const string& path, const vector<BenchmarkProblem>& problems) {
        ofstream out(path, ios::binary | ios::trunc);
        if (!out.is_open()) return false;

        ReferenceCacheHeader hdr;
        hdr.N = N;
        hdr.ref_order = REF_MAGNUS_ORDER;
        hdr.num_problems = static_cast<int32_t>(problems.size());
        hdr.T_total = T_TOTAL;
        hdr.eps0 = EPS0;
        hdr.W = W;
        hdr.ref_dt = REF_DT;

        out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        for (const auto& p : problems) {
            uint32_t name_len = static_cast<uint32_t>(p.name.size());
            out.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
            out.write(p.name.data(), name_len);
            if (!write_matrix_bin(out, p.H0)) return false;
            if (!write_matrix_bin(out, p.Hmod)) return false;
            if (!write_matrix_bin(out, p.Omega_ref)) return false;
            if (!write_matrix_bin(out, p.U_ref)) return false;
        }
        return out.good();
    }

    bool load_reference_cache(const string& path, vector<BenchmarkProblem>& problems) {
        ifstream in(path, ios::binary);
        if (!in.is_open()) return false;

        ReferenceCacheHeader hdr;
        in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        if (!in.good()) return false;

        ReferenceCacheHeader expected;
        if (hdr.magic != expected.magic ||
            hdr.version != expected.version ||
            hdr.N != N ||
            hdr.ref_order != REF_MAGNUS_ORDER ||
            hdr.num_problems <= 0 ||
            hdr.T_total != T_TOTAL ||
            hdr.eps0 != EPS0 ||
            hdr.W != W ||
            hdr.ref_dt != REF_DT)
        {
            return false;
        }

        const size_t elems = static_cast<size_t>(N) * static_cast<size_t>(N);
        problems.clear();
        problems.resize(static_cast<size_t>(hdr.num_problems));

        for (int i = 0; i < hdr.num_problems; ++i) {
            uint32_t name_len = 0;
            in.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
            if (!in.good() || name_len == 0 || name_len > 256) return false;

            problems[i].name.resize(name_len);
            in.read(problems[i].name.data(), name_len);
            if (!in.good()) return false;

            if (!read_matrix_bin(in, problems[i].H0, elems)) return false;
            if (!read_matrix_bin(in, problems[i].Hmod, elems)) return false;
            if (!read_matrix_bin(in, problems[i].Omega_ref, elems)) return false;
            if (!read_matrix_bin(in, problems[i].U_ref, elems)) return false;

            problems[i].H0_anti = matrix_ops::mat_scale(problems[i].H0, complexd(0.0, -1.0));
            problems[i].Hmod_anti = matrix_ops::mat_scale(problems[i].Hmod, complexd(0.0, -1.0));
        }

        return true;
    }

    vector<BenchmarkProblem> load_or_generate_problems(bool force_rebuild) {
        const string cache_path = utils::results_path("reference_cache_N256_T1e-3.bin");
        vector<BenchmarkProblem> problems;

        if (!force_rebuild && load_reference_cache(cache_path, problems)) {
            cout << "[CACHE] Loaded reference matrices and solutions from: " << cache_path << "\n";
            return problems;
        }

        cout << "[CACHE] Generating 3 benchmark problems and computing references...\n";
        problems.resize(3);

        // Задача 1: Хорошо обусловленная спектральная матрица (kappa = 10)
        {
            auto [H0_spec10, V10, D10] = utils::generate_spectral_matrix(N, 10.0, 10.0);
            problems[0].name = "1_WellCond_Kappa10";
            problems[0].H0 = std::move(H0_spec10);
            problems[0].Hmod = utils::generate_hermitian_matrix(N, 2.0);
        }

        // Задача 2: Сильно некоммутирующая DD-пара со скорректированной нормой под [0, 0.001]
        // (||H0||_2 ~ 80, ||Hmod||_2 ~ 390 => T * ||H||_2 ~ 0.47 << pi)
        {
            auto [H_ddA, H_ddB] = utils::generate_hermitian_pair_dd(N, 0.25, 0.25, 6.0, 15.0);
            problems[1].name = "2_HighNonComm_DD";
            problems[1].H0 = std::move(H_ddA);
            problems[1].Hmod = std::move(H_ddB);
        }

        // Задача 3: Жесткая спектральная матрица (kappa = 10000)
        {
            auto [H0_spec10k, V10k, D10k] = utils::generate_spectral_matrix(N, 10000.0, 50.0);
            problems[2].name = "3_Stiff_Kappa10000";
            problems[2].H0 = std::move(H0_spec10k);
            problems[2].Hmod = utils::generate_hermitian_matrix(N, 5.0);
        }

        for (auto& prob : problems) {
            cout << "  Computing Omega_ref (Recursive order=" << REF_MAGNUS_ORDER
                 << ", dt=" << REF_DT << ") and U_ref (Pade) for [" << prob.name << "]...\n";
            prob.H0_anti = matrix_ops::mat_scale(prob.H0, complexd(0.0, -1.0));
            prob.Hmod_anti = matrix_ops::mat_scale(prob.Hmod, complexd(0.0, -1.0));
            prob.Omega_ref = magnus_expansion(0.0, T_TOTAL, REF_DT, N, prob.H0, prob.Hmod, EPS0, W, REF_MAGNUS_ORDER);
            prob.U_ref = matrix_ops::expm_pade_eigen(prob.Omega_ref, N);
        }

        if (save_reference_cache(cache_path, problems)) {
            cout << "[CACHE] Saved references to: " << cache_path << "\n";
        } else {
            cerr << "[WARN] Failed to write reference cache: " << cache_path << "\n";
        }

        return problems;
    }

    // -------------------------------------------------------------------------
    // Точные целочисленные границы шага s in [0, num_steps - 1]
    // -------------------------------------------------------------------------
    using StepOmegaFn = function<CMatrix(const BenchmarkProblem&, double t0, double t1)>;

    inline void step_bounds(int step_idx, int num_steps, double& t0, double& t1) {
        t0 = (step_idx == 0)
           ? 0.0
           : T_TOTAL * (static_cast<double>(step_idx) / static_cast<double>(num_steps));
        t1 = (step_idx + 1 == num_steps)
           ? T_TOTAL
           : T_TOTAL * (static_cast<double>(step_idx + 1) / static_cast<double>(num_steps));
    }

    // -------------------------------------------------------------------------
    // Часть 1: Прогон метода Магнуса по всем 3 задачам на сетке из K шагов
    // -------------------------------------------------------------------------
    struct Part1Record {
        string method;
        int order = 0;
        int num_steps = 1;
        double dt = 0.0;
        string problem;
        double global_error = 0.0;
        bool has_single_omega_err = false;
        double single_omega_err = 0.0;
        uint64_t flops = 0;
        double time_ms = 0.0;
        bool is_worst = false;

        double gflops_per_sec() const {
            return (time_ms > 0.0) ? (static_cast<double>(flops) * 1e-6 / time_ms) : 0.0;
        }
    };

    void run_part1_config(
        const string& method_name,
        int order,
        int num_steps,
        const vector<BenchmarkProblem>& problems,
        ofstream& csv,
        const StepOmegaFn& omega_step_fn)
    {
        vector<Part1Record> runs;
        runs.reserve(problems.size());

        const double dt_nominal = T_TOTAL / static_cast<double>(num_steps);
        size_t worst_idx = 0;
        double max_err = -1.0;

        for (size_t i = 0; i < problems.size(); ++i) {
            const auto& prob = problems[i];

            // 1) Проверка глобальной точности на [0, T_TOTAL] с машинно-точной экспонентой
            CMatrix U_curr = utils::eye(N);
            CMatrix U_next((size_t)N * N, complexd(0.0, 0.0));
            double single_omega_err = 0.0;

            for (int s = 0; s < num_steps; ++s) {
                double t0 = 0.0, t1 = 0.0;
                step_bounds(s, num_steps, t0, t1);
                CMatrix Om = omega_step_fn(prob, t0, t1);

                if (num_steps == 1) {
                    single_omega_err = matrix_ops::max_element_diff(Om, prob.Omega_ref, N);
                }

                CMatrix E_s = matrix_ops::expm_cheb(Om, N, EXACT_CHEB_ORDER);
                if (s == 0) {
                    U_curr = std::move(E_s);
                } else {
                    // Строго левое умножение: U(t_{s+1}) = exp(Omega_s) * U(t_s)
                    matrix_ops::matmul(E_s, U_curr, U_next, N);
                    std::swap(U_curr, U_next);
                }
            }

            const double global_err = matrix_ops::max_element_diff(U_curr, prob.U_ref, N);

            // 2) Профилирование ТОЛЬКО вычисления Omega на K шагах
            volatile double sink_real = 0.0;
            profiling::FlopScope scope;
            for (int iter = 0; iter < NUM_ITERS; ++iter) {
                for (int s = 0; s < num_steps; ++s) {
                    double t0 = 0.0, t1 = 0.0;
                    step_bounds(s, num_steps, t0, t1);
                    CMatrix Om = omega_step_fn(prob, t0, t1);
                    sink_real += Om[0].real();
                }
            }
            const double avg_time_ms = scope.elapsed_ms() / static_cast<double>(NUM_ITERS);
            const uint64_t avg_flops = scope.flops() / static_cast<uint64_t>(NUM_ITERS);
            (void)sink_real;

            Part1Record rec;
            rec.method = method_name;
            rec.order = order;
            rec.num_steps = num_steps;
            rec.dt = dt_nominal;
            rec.problem = prob.name;
            rec.global_error = global_err;
            rec.has_single_omega_err = (num_steps == 1);
            rec.single_omega_err = single_omega_err;
            rec.flops = avg_flops;
            rec.time_ms = avg_time_ms;

            if (global_err > max_err) {
                max_err = global_err;
                worst_idx = i;
            }
            runs.push_back(std::move(rec));
        }

        runs[worst_idx].is_worst = true;

        for (const auto& rec : runs) {
            csv << rec.method << ","
                << rec.order << ","
                << rec.num_steps << ","
                << scientific << setprecision(6) << rec.dt << ","
                << rec.problem << ","
                << scientific << setprecision(12) << rec.global_error << ",";
            if (rec.has_single_omega_err) {
                csv << scientific << setprecision(12) << rec.single_omega_err;
            }
            csv << ","
                << rec.flops << ","
                << fixed << setprecision(6) << rec.time_ms << ","
                << fixed << setprecision(3) << rec.gflops_per_sec() << ","
                << (rec.is_worst ? 1 : 0) << "\n";
        }
        csv.flush();

        const auto& w = runs[worst_idx];
        cout << left << setw(14) << w.method
             << setw(6) << w.order
             << setw(7) << w.num_steps
             << scientific << setprecision(1) << setw(10) << w.dt
             << scientific << setprecision(4) << setw(15) << w.global_error
             << defaultfloat << setw(16) << w.flops
             << fixed << setprecision(3) << setw(12) << w.time_ms
             << fixed << setprecision(2) << setw(9) << w.gflops_per_sec()
             << w.problem << "\n";
    }

    // -------------------------------------------------------------------------
    // Часть 2: Прогон связки (Magnus + Chebyshev) на 2D-сетке (dt, M)
    // -------------------------------------------------------------------------
    struct Part2Record {
        string method;
        int magnus_order = 0;
        int num_steps = 1;
        double dt = 0.0;
        int cheb_order = 0;
        string problem;
        double global_error = 0.0;
        double unitarity_defect = 0.0;
        uint64_t flops = 0;
        double time_ms = 0.0;
        bool is_worst = false;

        double gflops_per_sec() const {
            return (time_ms > 0.0) ? (static_cast<double>(flops) * 1e-6 / time_ms) : 0.0;
        }
    };

    void run_part2_config(
        const string& method_name,
        int magnus_order,
        int num_steps,
        int cheb_order,
        const vector<BenchmarkProblem>& problems,
        const CMatrix& eye_N,
        ofstream& csv,
        const StepOmegaFn& omega_step_fn)
    {
        vector<Part2Record> runs;
        runs.reserve(problems.size());

        const double dt_nominal = T_TOTAL / static_cast<double>(num_steps);
        size_t worst_idx = 0;
        double max_err = -1.0;

        for (size_t i = 0; i < problems.size(); ++i) {
            const auto& prob = problems[i];
            CMatrix U_final;

            profiling::FlopScope scope;
            for (int iter = 0; iter < NUM_ITERS; ++iter) {
                CMatrix U_curr = utils::eye(N);
                CMatrix U_next((size_t)N * N, complexd(0.0, 0.0));

                for (int s = 0; s < num_steps; ++s) {
                    double t0 = 0.0, t1 = 0.0;
                    step_bounds(s, num_steps, t0, t1);
                    CMatrix Om = omega_step_fn(prob, t0, t1);
                    CMatrix E_s = matrix_ops::expm_cheb(Om, N, cheb_order);

                    if (s == 0) {
                        U_curr = std::move(E_s);
                    } else {
                        matrix_ops::matmul(E_s, U_curr, U_next, N);
                        std::swap(U_curr, U_next);
                    }
                }
                U_final = std::move(U_curr);
            }
            const double avg_time_ms = scope.elapsed_ms() / static_cast<double>(NUM_ITERS);
            const uint64_t avg_flops = scope.flops() / static_cast<uint64_t>(NUM_ITERS);

            const double global_err = matrix_ops::max_element_diff(U_final, prob.U_ref, N);

            // Дефект унитарности max |U^* U - I| (вне FlopScope)
            CMatrix Uh = matrix_ops::dagger(U_final, N);
            CMatrix UhU((size_t)N * N, complexd(0.0, 0.0));
            matrix_ops::matmul(Uh, U_final, UhU, N);
            const double unit_defect = matrix_ops::max_element_diff(UhU, eye_N, N);

            Part2Record rec;
            rec.method = method_name;
            rec.magnus_order = magnus_order;
            rec.num_steps = num_steps;
            rec.dt = dt_nominal;
            rec.cheb_order = cheb_order;
            rec.problem = prob.name;
            rec.global_error = global_err;
            rec.unitarity_defect = unit_defect;
            rec.flops = avg_flops;
            rec.time_ms = avg_time_ms;

            if (global_err > max_err) {
                max_err = global_err;
                worst_idx = i;
            }
            runs.push_back(std::move(rec));
        }

        runs[worst_idx].is_worst = true;

        for (const auto& rec : runs) {
            csv << rec.method << ","
                << rec.magnus_order << ","
                << rec.num_steps << ","
                << scientific << setprecision(6) << rec.dt << ","
                << rec.cheb_order << ","
                << rec.problem << ","
                << scientific << setprecision(12) << rec.global_error << ","
                << scientific << setprecision(12) << rec.unitarity_defect << ","
                << rec.flops << ","
                << fixed << setprecision(6) << rec.time_ms << ","
                << fixed << setprecision(3) << rec.gflops_per_sec() << ","
                << (rec.is_worst ? 1 : 0) << "\n";
        }
        csv.flush();

        const auto& w = runs[worst_idx];
        cout << left << setw(22) << w.method
             << setw(7) << w.num_steps
             << scientific << setprecision(1) << setw(10) << w.dt
             << setw(6) << w.cheb_order
             << scientific << setprecision(4) << setw(15) << w.global_error
             << defaultfloat << setw(16) << w.flops
             << fixed << setprecision(3) << setw(12) << w.time_ms
             << w.problem << "\n";
    }

} // namespace

int main(int argc, char* argv[]) {
    std::setlocale(LC_ALL, "Russian");

    bool force_rebuild = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--rebuild-ref") == 0) {
            force_rebuild = true;
        }
    }

    mkl_set_dynamic(0);
    omp_set_dynamic(0);
    mkl_set_num_threads(1);
    omp_set_num_threads(1);

    cout << "========================================================================================================\n";
    cout << " EXPERIMENT: Global Error vs dt & Chebyshev Order on [0, " << T_TOTAL
         << "] (N = " << N << ", NUM_ITERS = " << NUM_ITERS << ")\n";
    cout << "========================================================================================================\n";

    // -------------------------------------------------------------------------
    // 1. Загрузка или генерация 3 задач и эталонов
    // -------------------------------------------------------------------------
    vector<BenchmarkProblem> problems = load_or_generate_problems(force_rebuild);
    const CMatrix eye_N = utils::eye(N);

    cout << "\n[SANITY CHECK] Norms and exact-expm accuracy (Cheb M=" << EXACT_CHEB_ORDER << " vs Pade):\n";
    for (const auto& p : problems) {
        CMatrix U_check = matrix_ops::expm_cheb(p.Omega_ref, N, EXACT_CHEB_ORDER);
        double diff = matrix_ops::max_element_diff(U_check, p.U_ref, N);
        double om_norm = matrix_ops::mat_one_norm(p.Omega_ref, N);
        cout << "  " << left << setw(22) << p.name
             << " ||Omega_ref||_1 = " << fixed << setprecision(4) << setw(10) << om_norm
             << " |Cheb(" << EXACT_CHEB_ORDER << ") - Pade|_max = " << scientific << setprecision(3) << diff << "\n";
    }

    // Хелперы для генерации одного шага Omega на [t0, t1]
    auto make_classic_fn = [](int k) -> StepOmegaFn {
        return [k](const BenchmarkProblem& p, double t0, double t1) {
            const double inner_dt = (t1 - t0) / static_cast<double>(INNER_SUB_DEFAULT);
            return magnus_classic(t0, t1, inner_dt, N, p.H0, p.Hmod, EPS0, W, k);
        };
    };

    auto make_acc_fn = [](int k) -> StepOmegaFn {
        return [k](const BenchmarkProblem& p, double t0, double t1) {
            const int sub_intervals = (k >= 6) ? 6 : 2;
            const double sub_dt = (t1 - t0) / static_cast<double>(sub_intervals);
            auto samples = generate_samples(t0, t1, sub_dt, N, p.H0, p.Hmod, EPS0, W);
            const double actual_sub_dt = (t1 - t0) / static_cast<double>(samples.size() - 1);
            return magnus_ACC(samples, actual_sub_dt, N, k);
        };
    };

    auto make_recursive_fn = [](int k) -> StepOmegaFn {
        return [k](const BenchmarkProblem& p, double t0, double t1) {
            const double inner_dt = (t1 - t0) / static_cast<double>(INNER_SUB_DEFAULT);
            return magnus_expansion(t0, t1, inner_dt, N, p.H0, p.Hmod, EPS0, W, k);
        };
    };

    StepOmegaFn formula314_fn = [](const BenchmarkProblem& p, double t0, double t1) {
        const double inner_dt = (t1 - t0) / static_cast<double>(INNER_SUB_DEFAULT);
        return magnus_3_14(t0, t1, inner_dt, N, p.H0_anti, p.Hmod_anti, EPS0, W);
    };

    // -------------------------------------------------------------------------
    // 2. ЧАСТЬ 1: Точность и FLOPs методов Магнуса в зависимости от dt
    // -------------------------------------------------------------------------
    const string part1_csv_path = utils::results_path("experiment_part1_omega_dt.csv");
    ofstream part1_csv(part1_csv_path);
    part1_csv << "Method,Order,NumSteps,dt,Problem,GlobalError,SingleStepOmegaError,FLOPs,Time_ms,GFLOPS_s,Is_Worst\n";

    cout << "\n========================================================================================================\n";
    cout << " PART 1: Magnus Methods vs dt on [0, " << T_TOTAL << "] (Console shows WORST-CASE only)\n";
    cout << "========================================================================================================\n";
    cout << left << setw(14) << "Method"
         << setw(6) << "Ord"
         << setw(7) << "Steps"
         << setw(10) << "dt"
         << setw(15) << "Worst Error"
         << setw(16) << "FLOPs (Omega)"
         << setw(12) << "Time (ms)"
         << setw(9) << "GFLOP/s"
         << "Worst Matrix\n";
    cout << "--------------------------------------------------------------------------------------------------------\n";

    // 2.1. Classic Magnus (k = 1, 2, 3)
    for (int k : {1, 2, 3}) {
        auto fn = make_classic_fn(k);
        for (int K : STEP_COUNTS) {
            run_part1_config("Classic", k, K, problems, part1_csv, fn);
        }
    }

    // 2.2. ACC Magnus (k = 1, 4, 6 — 5-й убран, 4-й добавлен, дубль 2-го убран)
    for (int k : {1, 4, 6}) {
        auto fn = make_acc_fn(k);
        for (int K : STEP_COUNTS) {
            run_part1_config("ACC", k, K, problems, part1_csv, fn);
        }
    }

    // 2.3. Recursive Magnus (k = 1..6)
    for (int k : {1, 2, 3, 4, 5, 6}) {
        auto fn = make_recursive_fn(k);
        for (int K : STEP_COUNTS) {
            run_part1_config("Recursive", k, K, problems, part1_csv, fn);
        }
    }

    // 2.4. Formula 3.14 (только в Части 1)
    for (int K : STEP_COUNTS) {
        run_part1_config("Formula314", 4, K, problems, part1_csv, formula314_fn);
    }

    part1_csv.close();

    // -------------------------------------------------------------------------
    // 3. ЧАСТЬ 2: 2D-сетка (dt x M) для ACC4+Cheb, ACC6+Cheb, Recursive4+Cheb
    // -------------------------------------------------------------------------
    const string part2_csv_path = utils::results_path("experiment_part2_step_dt_cheb.csv");
    ofstream part2_csv(part2_csv_path);
    part2_csv << "Method,MagnusOrder,NumSteps,dt,ChebOrder,Problem,GlobalError,UnitarityDefect,FLOPs,Time_ms,GFLOPS_s,Is_Worst\n";

    cout << "\n========================================================================================================\n";
    cout << " PART 2: Full Propagation (Magnus + Chebyshev) vs (dt, M) on [0, " << T_TOTAL << "] (WORST-CASE)\n";
    cout << "========================================================================================================\n";
    cout << left << setw(22) << "Method"
         << setw(7) << "Steps"
         << setw(10) << "dt"
         << setw(6) << "M"
         << setw(15) << "Worst Error"
         << setw(16) << "FLOPs (Total)"
         << setw(12) << "Time (ms)"
         << "Worst Matrix\n";
    cout << "--------------------------------------------------------------------------------------------------------\n";

    struct Part2MethodSpec {
        string name;
        int magnus_order;
        StepOmegaFn omega_fn;
    };

    vector<Part2MethodSpec> part2_methods = {
        { "ACC4+Chebyshev",       4, make_acc_fn(4) },
        { "ACC6+Chebyshev",       6, make_acc_fn(6) },
        { "Recursive4+Chebyshev", 4, make_recursive_fn(4) }
    };

    for (const auto& spec : part2_methods) {
        for (int K : STEP_COUNTS) {
            for (int M : CHEB_ORDERS) {
                run_part2_config(spec.name, spec.magnus_order, K, M, problems, eye_N, part2_csv, spec.omega_fn);
            }
        }
    }

    part2_csv.close();

    cout << "\n========================================================================================================\n";
    cout << "DONE. Raw datasets saved for Excel analysis:\n"
         << "  Part 1: " << part1_csv_path << "\n"
         << "  Part 2: " << part2_csv_path << "\n"
         << "========================================================================================================\n";

    return 0;
}

#endif