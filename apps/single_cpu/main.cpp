#if 1

#include <iomanip>
#include <chrono>
#include "AllComputationalMethods.h"
#include "RungeKutt.h"

using namespace std;
using namespace utils;
using namespace matrix_ops;
using namespace runge_kutta_simple;

using complexd = complex<double>;
using CMatrix = vector<complexd>;

const double PI = 3.14159265358979323846;

int main() {
    const int N = 100;
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
       auto start = chrono::high_resolution_clock::now();
       CMatrix Omega_calc = magnus_classic(0.0, T_int, integration_dt, N, H0_test, Hmod_test, eps0, W, k);
       auto end = chrono::high_resolution_clock::now();
       double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;

       CMatrix diff(N * N, complexd(0.0, 0.0));
       matrix_ops::mat_sub(Omega_calc, Omega_ref_dt, diff, N);
       results_txt << "Classic | " << k << " | " << matrix_ops::max_element_diff(Omega_calc, Omega_ref_dt, N)
           << " | " << matrix_ops::max_eigenvalue_modulus_hermitian(diff, N) << " | " << time_ms << "\n";
    }

    // 2. ACC Expansion
    for (int k = 1; k <= 5; ++k) {
       auto start = chrono::high_resolution_clock::now();
       CMatrix Omega_calc = magnus_ACC(A_samples, integration_dt, N, k);
       auto end = chrono::high_resolution_clock::now();
       double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;

       CMatrix diff(N * N, complexd(0.0, 0.0));
       matrix_ops::mat_sub(Omega_calc, Omega_ref_dt, diff, N);
       results_txt << "ACC | " << k << " | " << matrix_ops::max_element_diff(Omega_calc, Omega_ref_dt, N)
           << " | " << matrix_ops::max_eigenvalue_modulus_hermitian(diff, N) << " | " << time_ms << "\n";
    }

    // 3. Recursive Expansion
    for (int k = 1; k <= 8; ++k) {
       auto start = chrono::high_resolution_clock::now();
       CMatrix Omega_calc = magnus_expansion(0.0, T_int, integration_dt, N, H0_test, Hmod_test, eps0, W, k);
       auto end = chrono::high_resolution_clock::now();
       double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;

       CMatrix diff(N * N, complexd(0.0, 0.0));
       matrix_ops::mat_sub(Omega_calc, Omega_ref_dt, diff, N);
       results_txt << "Recursive | " << k << " | " << matrix_ops::max_element_diff(Omega_calc, Omega_ref_dt, N)
           << " | " << matrix_ops::max_eigenvalue_modulus_hermitian(diff, N) << " | " << time_ms << "\n";
    }

    // 4. Formula 3.14
    {
       auto start = chrono::high_resolution_clock::now();
       CMatrix Omega_314 = magnus_3_14(0.0, T_int, integration_dt, N, H0_anti_herm, Hmod_anti_herm, eps0, W);
       auto end = chrono::high_resolution_clock::now();
       double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;

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

       // Берем Omega_ref_dt из Эксперимента 1 как входные данные
       CMatrix U_ref = matrix_ops::expm_pade_eigen(Omega_ref_dt, N);

       // 1. Taylor Expansion
       std::vector<int> taylor_terms = { 1, 2, 3, 4, 5, 6, 7, 8 ,9 , 10, 20, 30 };
       for (int terms : taylor_terms) {
           auto start = chrono::high_resolution_clock::now();
           CMatrix U_calc = matrix_ops::expm_taylor(Omega_ref_dt, N, terms);
           auto end = chrono::high_resolution_clock::now();
           double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;

           CMatrix diff(N * N, complexd(0.0, 0.0));
           matrix_ops::mat_sub(U_calc, U_ref, diff, N);
           exp2_results << "Taylor | terms=" << terms << " | " << matrix_ops::max_element_diff(U_calc, U_ref, N)
               << " | " << matrix_ops::max_eigenvalue_modulus_hermitian(diff, N) << " | " << time_ms << "\n";
       }

       // 2. Chebyshev Expansion
       std::vector<int> cheb_terms = { 1, 2, 3, 4, 5, 6, 7, 8 ,9 ,  10, 15, 25 };
       for (int M : cheb_terms) {
           auto start = chrono::high_resolution_clock::now();
           CMatrix U_calc = matrix_ops::expm_cheb(Omega_ref_dt, N, M);
           auto end = chrono::high_resolution_clock::now();
           double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;

           CMatrix diff(N * N, complexd(0.0, 0.0));
           matrix_ops::mat_sub(U_calc, U_ref, diff, N);
           exp2_results << "Chebyshev | M=" << M << " | " << matrix_ops::max_element_diff(U_calc, U_ref, N)
               << " | " << matrix_ops::max_eigenvalue_modulus_hermitian(diff, N) << " | " << time_ms << "\n";
       }

       // 3. Стандартный Pade (если реализован отдельно от Eigen)
       {
           auto start = chrono::high_resolution_clock::now();
           CMatrix U_calc = matrix_ops::expm_pade_eigen(Omega_ref_dt, N); // В данном случае совпадет с референсом
           auto end = chrono::high_resolution_clock::now();
           double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
           exp2_results << "Eigen-Pade | -- | 0.000000e+00 | 0.000000e+00 | " << time_ms << "\n";
       }

       exp2_results.close();
       std::cout << "Experiment 2 results written to experiment2_results.txt\n";
    }

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
       CMatrix U_exact = utils::eye(N);

       std::cout << "Computing Exact U(T) via fine RK4 (dt=1e-6)... ";
       auto start_exact = chrono::high_resolution_clock::now();
       for (int step = 0; step < exact_steps; ++step) {
           U_exact = runge_kutta_simple::runge_kutta_step(step * fine_rk_dt, H0_test, Hmod_test, W, U_exact, fine_rk_dt, N);
       }
       auto end_exact = chrono::high_resolution_clock::now();
       std::cout << "Done in " << chrono::duration_cast<chrono::microseconds>(end_exact - start_exact).count() / 1000.0 << " ms\n";

       // Method 1: Standard Runge-Kutta
       {
           double rk_dt = 1e-4; // Обычный шаг
           int num_steps = (int)std::round(T / rk_dt);

           auto start_rk = chrono::high_resolution_clock::now();
           CMatrix U_rk = utils::eye(N);
           for (int step = 0; step < num_steps; ++step) {
               U_rk = runge_kutta_simple::runge_kutta_step(step * rk_dt, H0_test, Hmod_test, W, U_rk, rk_dt, N);
           }
           auto end_rk = chrono::high_resolution_clock::now();
           double time_rk_ms = chrono::duration_cast<chrono::microseconds>(end_rk - start_rk).count() / 1000.0;

           CMatrix diff_rk(N * N, complexd(0.0, 0.0));
           matrix_ops::mat_sub(U_rk, U_exact, diff_rk, N);
           exp3_results << "Runge-Kutta dt=1e-4 | " << matrix_ops::max_element_diff(U_rk, U_exact, N) << " | " << time_rk_ms << "\n";
       }

       // Method 2: Piecewise Magnus ACC + Chebyshev
       {
           // Используем macro_dt, определяющий размер одного "куска" Магнуса
           double macro_dt = 0.01;

           auto start_magnus = chrono::high_resolution_clock::now();

           // Вызов универсального солвера, который внутри себя разбивает T на шаги macro_dt,
           // для каждого шага считает локальную Омегу и перемножает экспоненты.
           CMatrix U_magnus = piecewise_magnus_solver(0.0, T, macro_dt, N, H0_test, Hmod_test, eps0, W, 4);

           auto end_magnus = chrono::high_resolution_clock::now();
           double time_magnus_ms = chrono::duration_cast<chrono::microseconds>(end_magnus - start_magnus).count() / 1000.0;

           CMatrix diff_magnus(N * N, complexd(0.0, 0.0));
           matrix_ops::mat_sub(U_magnus, U_exact, diff_magnus, N);

           exp3_results << "Piecewise Magnus ACC | " << matrix_ops::max_element_diff(U_magnus, U_exact, N) << " | " << time_magnus_ms << "\n";
       }
       exp3_results.close();
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

           // Эталонная Omega на мелкой сетке
           CMatrix Omega_ref = magnus_expansion(0.0, T_int, integration_dt * 0.1, N, H_A, H_B, eps0, W, 10);

           // Подготовка сэмплов и антиэрмитовых матриц для тестируемых методов
           vector<CMatrix> A_samples_k = generate_samples(0.0, T_int, integration_dt, N, H_A, H_B, eps0, W);
           CMatrix H_A_anti = matrix_ops::mat_scale(H_A, complexd(0.0, -1.0));
           CMatrix H_B_anti = matrix_ops::mat_scale(H_B, complexd(0.0, -1.0));

           // Classic Magnus (k=2)
           {
               auto start = chrono::high_resolution_clock::now();
               CMatrix Omega_calc = magnus_classic(0.0, T_int, integration_dt, N, H_A, H_B, eps0, W, 2);
               auto end = chrono::high_resolution_clock::now();
               double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;

               CMatrix diff(N * N, complexd(0.0, 0.0));
               matrix_ops::mat_sub(Omega_calc, Omega_ref, diff, N);
               exp4_results << kappa << " | Classic | " << matrix_ops::max_element_diff(Omega_calc, Omega_ref, N)
                   << " | " << matrix_ops::max_eigenvalue_modulus_hermitian(diff, N) << " | " << time_ms << "\n";
           }

           // ACC Magnus (k=5)
           {
               auto start = chrono::high_resolution_clock::now();
               CMatrix Omega_calc = magnus_ACC(A_samples_k, integration_dt, N, 5);
               auto end = chrono::high_resolution_clock::now();
               double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;

               CMatrix diff(N * N, complexd(0.0, 0.0));
               matrix_ops::mat_sub(Omega_calc, Omega_ref, diff, N);
               exp4_results << kappa << " | ACC | " << matrix_ops::max_element_diff(Omega_calc, Omega_ref, N)
                   << " | " << matrix_ops::max_eigenvalue_modulus_hermitian(diff, N) << " | " << time_ms << "\n";
           }

           // Recursive Magnus (k=8)
           {
               auto start = chrono::high_resolution_clock::now();
               CMatrix Omega_calc = magnus_expansion(0.0, T_int, integration_dt, N, H_A, H_B, eps0, W, 8);
               auto end = chrono::high_resolution_clock::now();
               double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;

               CMatrix diff(N * N, complexd(0.0, 0.0));
               matrix_ops::mat_sub(Omega_calc, Omega_ref, diff, N);
               exp4_results << kappa << " | Recursive | " << matrix_ops::max_element_diff(Omega_calc, Omega_ref, N)
                   << " | " << matrix_ops::max_eigenvalue_modulus_hermitian(diff, N) << " | " << time_ms << "\n";
           }

           // Formula 3.14
           {
               auto start = chrono::high_resolution_clock::now();
               CMatrix Omega_calc = magnus_3_14(0.0, T_int, integration_dt, N, H_A_anti, H_B_anti, eps0, W);
               auto end = chrono::high_resolution_clock::now();
               double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;

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

           // Генерируем "плохую" Omega с помощью Magnus ACC
           vector<CMatrix> A_samp = generate_samples(0.0, T_int, integration_dt, N, H_A, H_B, eps0, W);
           CMatrix Omega_test = magnus_ACC(A_samp, integration_dt, N, 4);

           // Эталонное экспоненцирование через Eigen Pade
           CMatrix U_ref = matrix_ops::expm_pade_eigen(Omega_test, N);

           // Taylor method (terms = 30)
           {
               auto start = chrono::high_resolution_clock::now();
               CMatrix U_calc = matrix_ops::expm_taylor(Omega_test, N, 30);
               auto end = chrono::high_resolution_clock::now();
               double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;

               CMatrix diff(N * N, complexd(0.0, 0.0));
               matrix_ops::mat_sub(U_calc, U_ref, diff, N);
               exp5_results << kappa << " | Taylor | " << matrix_ops::max_element_diff(U_calc, U_ref, N)
                   << " | " << matrix_ops::max_eigenvalue_modulus_hermitian(diff, N) << " | " << time_ms << "\n";
           }

           // Chebyshev method (M = 25)
           {
               auto start = chrono::high_resolution_clock::now();
               CMatrix U_calc = matrix_ops::expm_cheb(Omega_test, N, 25);
               auto end = chrono::high_resolution_clock::now();
               double time_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;

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

           std::cout << "Computing Fine-Grid Reference U(T) for kappa " << kappa << " (This will take a while)...\n";

           // Эталонное решение U(T)
           auto start_exact = chrono::high_resolution_clock::now();
           CMatrix U_exact = utils::eye(N);
           for (int step = 0; step < exact_steps; ++step) {
               U_exact = runge_kutta_simple::runge_kutta_step(step * fine_rk_dt, H_A, H_B, W, U_exact, fine_rk_dt, N);
           }
           auto end_exact = chrono::high_resolution_clock::now();
           std::cout << "Done in " << chrono::duration_cast<chrono::microseconds>(end_exact - start_exact).count() / 1000.0 << " ms\n";

           // Method 1: Runge-Kutta with normal dt = 1e-4
           {
               double rk_dt = 1e-4;
               int num_steps = (int)std::round(T / rk_dt);

               auto start_rk = chrono::high_resolution_clock::now();
               CMatrix U_rk = utils::eye(N);
               for (int step = 0; step < num_steps; ++step) {
                   U_rk = runge_kutta_simple::runge_kutta_step(step * rk_dt, H_A, H_B, W, U_rk, rk_dt, N);
               }
               auto end_rk = chrono::high_resolution_clock::now();
               double time_rk_ms = chrono::duration_cast<chrono::microseconds>(end_rk - start_rk).count() / 1000.0;

               CMatrix diff_rk(N * N, complexd(0.0, 0.0));
               matrix_ops::mat_sub(U_rk, U_exact, diff_rk, N);
               exp6_results << kappa << " | Runge-Kutta dt=1e-4 | " << matrix_ops::max_element_diff(U_rk, U_exact, N)
                   << " | " << matrix_ops::max_eigenvalue_modulus_hermitian(diff_rk, N) << " | " << time_rk_ms << "\n";
           }

           // Method 2: Piecewise Magnus ACC + Chebyshev
           {
               double macro_dt = 0.01; // Шаг кусочного Магнуса (в 100 раз больше Рунге-Кутты!)

               auto start_magnus = chrono::high_resolution_clock::now();

               // Вызов нашего нового солвера
               CMatrix U_magnus = piecewise_magnus_solver(0.0, T, macro_dt, N, H_A, H_B, eps0, W, 4);

               auto end_magnus = chrono::high_resolution_clock::now();
               double time_magnus_ms = chrono::duration_cast<chrono::microseconds>(end_magnus - start_magnus).count() / 1000.0;

               CMatrix diff_magnus(N * N, complexd(0.0, 0.0));
               matrix_ops::mat_sub(U_magnus, U_exact, diff_magnus, N);

               exp6_results << kappa << " | Piecewise Magnus ACC | "
                   << matrix_ops::max_element_diff(U_magnus, U_exact, N) << " | "
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

       // 1. Считаем эталонное решение с помощью кусочного метода Магнуса
       double macro_dt = 1e-3;
       std::cout << "Computing reference U(T) using piecewise_magnus_solver (Fast)...\n";
       auto start_exact = std::chrono::high_resolution_clock::now();

       CMatrix U_exact = piecewise_magnus_solver(0.0, T, macro_dt, N, H0_test, Hmod_test, eps0, W, 4);

       auto end_exact = std::chrono::high_resolution_clock::now();
       std::cout << "Done in " << std::chrono::duration_cast<std::chrono::microseconds>(end_exact - start_exact).count() / 1000.0 << " ms\n\n";

       // 2. Цикл проверки зависимости от dt для РК4
       double current_dt = 1e-3;
       double prev_error = 1e300; // Стартуем с огромного значения

       // Будем уменьшать dt в 2 раза, пока не достигнем машинного предела
       while (current_dt >= 1e-8) {
           int steps = (int)std::round(T / current_dt);
           CMatrix U_rk = utils::eye(N);

           auto start = std::chrono::high_resolution_clock::now();
           for (int step = 0; step < steps; ++step) {
               U_rk = runge_kutta_simple::runge_kutta_step(step * current_dt, H0_test, Hmod_test, W, U_rk, current_dt, N);
           }
           auto end = std::chrono::high_resolution_clock::now();
           double time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

           // Вычисляем ошибку по отношению к эталону Магнуса
           double current_error = matrix_ops::max_element_diff(U_rk, U_exact, N);

           exp7_results << current_dt << " | " << current_error << " | " << steps << " | " << time_ms << "\n";
           std::cout << "dt: " << std::setw(12) << current_dt << " | Error: " << current_error << "\n";

           // Условие выхода: 
           // Мы останавливаемся только если ошибка уже упала до нормальных значений (< 1.0) 
           // И при этом начала снова расти (или перестала падать) из-за накопления машинной погрешности.
           // Это позволяет "проскочить" взрывы на крупных шагах (когда error > 1.0).
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

       double inner_dt = 1e-6;          // Фиксированный мелкий внутренний шаг
       double current_macro_dt = 1e-1;  // Стартовый макро-шаг
       double tolerance = 1e-9;         // Порог сходимости

       CMatrix U_prev = utils::eye(N);  // Для хранения результата с предыдущим (более крупным) шагом
       bool is_first_iteration = true;
       double prev_error = 1e300;

       exp8_results << "=================================================================\n";
       exp8_results << "EXPERIMENT 8: Piecewise Magnus Self-Convergence vs macro_dt\n";
       exp8_results << "Fixed inner integration step: " << inner_dt << "\n";
       exp8_results << "Convergence tolerance: " << tolerance << "\n";
       exp8_results << "-----------------------------------------------------------------\n";
       exp8_results << "macro_dt | Diff with prev step | Steps | Time (ms)\n";
       exp8_results << "-----------------------------------------------------------------\n";

       // Будем измельчать макро-шаг, пока он не упрется во внутренний шаг
       while (current_macro_dt >= inner_dt) {
           int steps = (int)std::round(T / current_macro_dt);
           CMatrix U_curr = utils::eye(N);

           auto start = std::chrono::high_resolution_clock::now();

           for (int step = 0; step < steps; ++step) {
               double t_curr = step * current_macro_dt;
               double t_next = t_curr + current_macro_dt;

               // Генерируем сэмплы с фиксированным мелким шагом inner_dt
               vector<CMatrix> step_samples = generate_samples(t_curr, t_next, inner_dt, N, H0_test, Hmod_test, eps0, W);

               // Считаем локальную Омегу и берем экспоненту (Чебышев)
               CMatrix Omega_local = magnus_ACC(step_samples, inner_dt, N, 4);
               CMatrix U_local = matrix_ops::expm_cheb(Omega_local, N, 15);

               // Обновляем общее решение: U(t+dt) = U_local * U(t)
               CMatrix temp((size_t)N * N, complexd(0.0, 0.0));
               matrix_ops::matmul(U_local, U_curr, temp, N);
               U_curr = temp;
           }

           auto end = std::chrono::high_resolution_clock::now();
           double time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

           if (!is_first_iteration) {
               // Сравниваем результат для текущего macro_dt с результатом для macro_dt * 2
               double diff = matrix_ops::max_element_diff(U_curr, U_prev, N);

               exp8_results << current_macro_dt << " | " << diff << " | " << steps << " | " << time_ms << "\n";
               std::cout << "macro_dt: " << std::setw(12) << current_macro_dt << " | Diff: " << diff << "\n";

               // Критерий успешной сходимости
               if (diff < tolerance) {
                   std::cout << "-> Converged! Accuracy tolerance reached.\n";
                   break;
               }

               // Выход из цикла, если точность перестала улучшаться из-за машинной погрешности
               if (diff > prev_error && diff < 1.0) {
                   std::cout << "-> Stopping: Accuracy plateau reached.\n";
                   break;
               }
               prev_error = diff;
           }
           else {
               // Для первой итерации нам не с чем сравнивать
               exp8_results << current_macro_dt << " | N/A | " << steps << " | " << time_ms << "\n";
               std::cout << "macro_dt: " << std::setw(12) << current_macro_dt << " | Diff: N/A\n";
               is_first_iteration = false;
           }

           // Запоминаем текущую матрицу для следующей итерации
           U_prev = U_curr;
           current_macro_dt /= 2.0; // Измельчаем внешний шаг
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

        // Настройки адаптивности
        double tolerance = 1e-12;
        int num_segments = 20;   // Фиксируем количество макро-шагов
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

            double current_inner_dt = macro_dt; // Начинаем с шага, равного всему отрезку
            CMatrix U_local_old = utils::eye(N);
            CMatrix U_local_new = utils::eye(N);

            bool converged = false;
            int halving_count = 0;

            auto start_seg = std::chrono::high_resolution_clock::now();

            std::cout << "Segment " << i + 1 << "/" << num_segments << " [" << t_start << ", " << t_end << "]:\n";

            while (!converged && halving_count < 25) { // Ограничитель, чтобы избежать бесконечного цикла
                vector<CMatrix> samples = generate_samples(t_start, t_end, current_inner_dt, N, H0_test, Hmod_test, eps0, W);
                CMatrix Omega = magnus_ACC(samples, current_inner_dt, N, 4);

                // Берем экспоненту (Чебышев)
                U_local_new = matrix_ops::expm_cheb(Omega, N, 15);

                // Проверяем сходимость
                if (halving_count > 0) {
                    // Сравниваем текущую U_local с той, что была рассчитана на предыдущем, более крупном шаге
                    double diff = matrix_ops::max_element_diff(U_local_new, U_local_old, N);

                    if (diff < tolerance) {
                        converged = true;
                        std::cout << "  -> Converged at inner_dt: " << std::setw(12) << current_inner_dt
                            << " (diff: " << diff << ")\n";
                    }
                }

                // Запоминаем текущий результат для следующей итерации
                U_local_old = U_local_new;

                if (!converged) {
                    current_inner_dt /= 2.0; // Измельчаем шаг
                    halving_count++;
                }
            }

            auto end_seg = std::chrono::high_resolution_clock::now();
            double time_ms_seg = std::chrono::duration_cast<std::chrono::microseconds>(end_seg - start_seg).count() / 1000.0;

            // Записываем статистику по сегменту
            exp9_results << i + 1 << " | " << t_start << " | " << t_end << " | "
                << current_inner_dt << " | " << halving_count << " | " << time_ms_seg << "\n";

            // Обновляем общее решение: U(t+dt) = U_local * U(t)
            CMatrix temp((size_t)N * N, complexd(0.0, 0.0));
            matrix_ops::matmul(U_local_new, U_total, temp, N);
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

    results_txt.close();
    std::cout << "\nExperiment results written." << std::endl;
    return 0;
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
