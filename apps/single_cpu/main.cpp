#if 1

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
