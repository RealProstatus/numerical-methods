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

int main() {
    const int N = 100;
    const double dt = 1e-5;
    const double T = 1.0;
    const double diag_min = 10.0;

    // =====================================================================
    // Generate and save test matrices
    // =====================================================================
    std::cout << "Generating test matrices N=" << N << "...\n";

    // Test matrix for all experiments
    CMatrix H_test = utils::generate_hermitian_matrix(N, diag_min);
    utils::save_matrix(H_test, N, "H_test_matrix.txt");
    std::cout << "Test matrix saved to H_test_matrix.txt\n";

    // Generate spectral matrices for stability experiments
    std::cout << "Generating spectral matrices for stability experiments...\n";
    std::vector<double> condition_numbers = {10.0, 100.0, 1000.0, 10000.0};
    std::vector<CMatrix> spectral_matrices;
    std::vector<CMatrix> spectral_V_matrices;
    std::vector<CMatrix> spectral_D_matrices;

    for (size_t i = 0; i < condition_numbers.size(); ++i) {
        double kappa = condition_numbers[i];
        auto [H, V, D] = utils::generate_spectral_matrix(N, kappa, 3.0);

        std::string filename = "H_spectral_kappa_" + std::to_string((int)kappa) + ".txt";
        utils::save_matrix(H, N, filename);
        std::cout << "Spectral matrix with kappa=" << kappa << " saved to " << filename << "\n";

        spectral_matrices.push_back(H);
        spectral_V_matrices.push_back(V);
        spectral_D_matrices.push_back(D);
    }

    // =====================================================================
    // EXPERIMENTS RESULTS FILE
    // =====================================================================
    std::ofstream results_txt("magnus_experiment_results.txt");
    if (!results_txt.is_open()) {
        std::cerr << "Error opening magnus_experiment_results.txt" << std::endl;
        return 1;
    }
    results_txt << std::scientific << std::setprecision(12);
    std::cout << std::scientific << std::setprecision(12);

    // =====================================================================
    // Generate constant Hamiltonian H_const (for experiment 1)
    // =====================================================================
    CMatrix H_const = utils::generate_hermitian_matrix(N, diag_min);

    // Reference Omega for dt: Omega_ref = -i * H * dt (Exact solution for constant H)
    CMatrix Omega_ref_dt = matrix_ops::mat_scale(H_const, complexd(0.0, -dt));

    // Parameters for methods
    // We set H0 = H_const (to be multiplied by -i inside functions if needed)
    // But wait: generate_samples multiplies by -i. 
    // magnus_3_14 expects H0 and H_mod as raw matrices and handles -i internally or expects them pre-scaled?
    // Looking at your MatrixOperations.cpp: magnus_3_14 computes commutators directly.
    // Usually H in physics is Hermitian. The exponent is -i*Omega.
    // Let's stick to the convention: Input matrices are Hermitian.

    // For Experiment 1 (Constant H), we map to the H(t) = H0 + f(t)Hmod form:
    CMatrix H0_input = H_const;                // The constant part
    CMatrix H_mod_zero(N * N, complexd(0.0, 0.0)); // No modulation part
    const double eps0 = 0.0;                   // Amplitude 0
    const double W = 0.0;                      // Frequency 0

    // For other methods that take samples A(t) = -i H(t):
    // Since eps0=0, A(t) is constantly -i * H_const.
    const double T_int = dt;
    const double integration_dt = dt * 0.01;

    // We pass H0_input. generate_samples multiplies it by -i * (1 + ...).
    // Since eps0=0, samples will be correctly -i * H_const.
    vector<CMatrix> A_samples = generate_samples(0.0, T_int, integration_dt, N,
        H0_input, H_mod_zero, eps0, W);

    // However, magnus_3_14 implementation in MatrixOperations.cpp (from previous prompts)
    // assumes A(t) construction inside or takes H0/Hmod. 
    // It constructs Omega directly. We need to ensure it scales by -i if it's not done inside.
    // Checking MatrixOperations.cpp logic for magnus_3_14:
    // It sums terms like "h * H0". If H0 is Hermitian, Omega is Hermitian.
    // But the result of magnus is usually the exponent argument. 
    // If U = exp(Omega), and U = exp(-iHt), then Omega must be -iHt.
    // Let's pass pre-scaled H0 to magnus_3_14 to be safe, OR check if it multiplies by -i.
    // Looking at snippet: term = H0; scale(h); add(term). It does NOT multiply by -i.
    // So for magnus_3_14, we must pass the anti-hermitian version if we want the result to match Omega_ref_dt.

    CMatrix H0_anti_herm = matrix_ops::mat_scale(H_const, complexd(0.0, -1.0)); // -i * H


    // =====================================================================
    // ================== EXPERIMENT 1: Omega Accuracy Comparison ====================
    // =====================================================================
    results_txt << "=================================================================\n";
    results_txt << "EXPERIMENT 1: Comparison of Omega Matrix Calculation Accuracy (H = const)\n";
    results_txt << "N=" << N << ", dt=" << dt << "\n";
    results_txt << "Reference: Omega_ref = -i * H * dt\n";
    results_txt << "-----------------------------------------------------------------\n";
    results_txt << "Method | K | Max Element Diff | Max Eig Diff | Time (ms)\n";
    results_txt << "-----------------------------------------------------------------\n";

    // --- 1. Classic Magnus Expansion ---
    int max_classic_order = 2;
    for (int k = 1; k <= max_classic_order; ++k) {
        auto start = chrono::high_resolution_clock::now();

        // magnus_classic takes H0, H_mod and generates samples internally with -i factor
        CMatrix Omega_calc = magnus_classic(0.0, T_int, integration_dt, N, H0_input, H_mod_zero, eps0, W, k);

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        double time_ms = duration.count() / 1000.0;

        CMatrix diff(N * N, complexd(0.0, 0.0));
        matrix_ops::mat_sub(Omega_calc, Omega_ref_dt, diff, N);

        double max_el_diff = matrix_ops::max_element_diff(Omega_calc, Omega_ref_dt, N);
        double max_eig_diff = matrix_ops::max_eigenvalue_modulus_hermitian(diff, N);

        results_txt << "Classic | " << k << " | " << max_el_diff << " | " << max_eig_diff << " | " << time_ms << "\n";
    }

    // --- 2. Diploma Expansion (ACC) ---
    int max_acc_order = 5;
    for (int k = 1; k <= max_acc_order; ++k) {
        auto start = chrono::high_resolution_clock::now();

        CMatrix Omega_calc = magnus_ACC(A_samples, integration_dt, N, k);

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        double time_ms = duration.count() / 1000.0;

        CMatrix diff(N * N, complexd(0.0, 0.0));
        matrix_ops::mat_sub(Omega_calc, Omega_ref_dt, diff, N);

        double max_el_diff = matrix_ops::max_element_diff(Omega_calc, Omega_ref_dt, N);
        double max_eig_diff = matrix_ops::max_eigenvalue_modulus_hermitian(diff, N);

        results_txt << "ACC | " << k << " | " << max_el_diff << " | " << max_eig_diff << " | " << time_ms << "\n";
    }

    // --- 3. Recursive Expansion ---
    int max_recursive_order = 15;
    for (int k = 1; k <= max_recursive_order; ++k) {
        auto start = chrono::high_resolution_clock::now();

        CMatrix Omega_calc = magnus_expansion(0.0, T_int, integration_dt, N, H0_input, H_mod_zero, eps0, W, k);

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        double time_ms = duration.count() / 1000.0;

        CMatrix diff(N * N, complexd(0.0, 0.0));
        matrix_ops::mat_sub(Omega_calc, Omega_ref_dt, diff, N);

        double max_el_diff = matrix_ops::max_element_diff(Omega_calc, Omega_ref_dt, N);
        double max_eig_diff = matrix_ops::max_eigenvalue_modulus_hermitian(diff, N);

        results_txt << "Recursive | " << k << " | " << max_el_diff << " | " << max_eig_diff << " | " << time_ms << "\n";
    }

    // --- 4. Formula 3.14 (Special 4th order) ---
    //  - Formula 3.14 from dissertation
    {
        auto start = chrono::high_resolution_clock::now();

        // magnus_3_14 sums H0 terms directly. To get Omega = -i*H*dt, 
        // we must pass H0_anti_herm (-i*H) because the function doesn't add -i itself.
        // H_mod is zero, so commutators are zero.
        CMatrix Omega_314 = magnus_3_14(0.0, T_int, integration_dt, N, H0_anti_herm, H_mod_zero, eps0, W);

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        double time_ms = duration.count() / 1000.0;

        CMatrix diff(N * N, complexd(0.0, 0.0));
        matrix_ops::mat_sub(Omega_314, Omega_ref_dt, diff, N);

        double max_el_diff = matrix_ops::max_element_diff(Omega_314, Omega_ref_dt, N);
        double max_eig_diff = matrix_ops::max_eigenvalue_modulus_hermitian(diff, N);

        // K=4 is the theoretical order of this method
        results_txt << "Formula 3.14 | 4 | " << max_el_diff << " | " << max_eig_diff << " | " << time_ms << "\n";
    }
    // =====================================================================
    // ============= EXPERIMENT 2: Matrix Exponentiation Methods ============
    // =====================================================================
    {
        std::ofstream exp2_results("experiment2_results.txt");
        if (!exp2_results.is_open()) {
            std::cerr << "Error opening experiment2_results.txt" << std::endl;
            return 1;
        }
        exp2_results << std::scientific << std::setprecision(12);
        std::cout << "\nRunning Experiment 2: Matrix Exponentiation Comparison...\n";

        exp2_results << "=================================================================\n";
        exp2_results << "EXPERIMENT 2: Matrix Exponentiation Methods Comparison\n";
        exp2_results << "N=" << N << "\n";
        exp2_results << "Test matrix: H_test_matrix.txt\n";
        exp2_results << "Reference: Eigen Pade approximation\n";
        exp2_results << "-----------------------------------------------------------------\n";
        exp2_results << "Method | Param | Max Element Diff | Max Eig Diff | Time (ms)\n";
        exp2_results << "-----------------------------------------------------------------\n";

        // Create test Omega = -i * H_test (anti-Hermitian matrix)
        CMatrix Omega_test = matrix_ops::mat_scale(H_test, complexd(0.0, -1.0));

        // Reference: Eigen Pade approximation
        auto start_ref = chrono::high_resolution_clock::now();
        CMatrix U_ref = matrix_ops::expm_pade_eigen(Omega_test, N);
        auto end_ref = chrono::high_resolution_clock::now();
        auto duration_ref = chrono::duration_cast<chrono::microseconds>(end_ref - start_ref);
        double time_ref_ms = duration_ref.count() / 1000.0;

        std::cout << "Reference (Eigen Pade) computed in " << time_ref_ms << " ms\n";

        // Taylor method comparison (terms = 1..30)
        for (int k = 1; k <= 30; ++k) {
            auto start = chrono::high_resolution_clock::now();

            CMatrix U_taylor = matrix_ops::expm_taylor(Omega_test, N, k);

            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
            double time_ms = duration.count() / 1000.0;

            CMatrix diff(N * N, complexd(0.0, 0.0));
            matrix_ops::mat_sub(U_taylor, U_ref, diff, N);

            double max_el_diff = matrix_ops::max_element_diff(U_taylor, U_ref, N);
            double max_eig_diff = matrix_ops::max_eigenvalue_modulus_hermitian(diff, N);

            exp2_results << "Taylor | " << k << " | " << max_el_diff << " | " << max_eig_diff << " | " << time_ms << "\n";
        }

        // Chebyshev method comparison (M = 1..25)
        for (int M = 1; M <= 25; ++M) {
            auto start = chrono::high_resolution_clock::now();

            CMatrix U_cheb = matrix_ops::expm_cheb(Omega_test, N, M);

            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
            double time_ms = duration.count() / 1000.0;

            CMatrix diff(N * N, complexd(0.0, 0.0));
            matrix_ops::mat_sub(U_cheb, U_ref, diff, N);

            double max_el_diff = matrix_ops::max_element_diff(U_cheb, U_ref, N);
            double max_eig_diff = matrix_ops::max_eigenvalue_modulus_hermitian(diff, N);

            exp2_results << "Chebyshev | " << M << " | " << max_el_diff << " | " << max_eig_diff << " | " << time_ms << "\n";
        }

        exp2_results << "-----------------------------------------------------------------\n";
        exp2_results.close();
        std::cout << "Experiment 2 results written to experiment2_results.txt\n";
    }

    // =====================================================================
    // ============= EXPERIMENT 3: ODE Solving U'(t) = -iH*U(t) =============
    // =====================================================================

    {
        std::ofstream exp3_results("experiment3_results.txt");
        if (!exp3_results.is_open()) {
            std::cerr << "Error opening experiment3_results.txt" << std::endl;
            return 1;
        }
        exp3_results << std::scientific << std::setprecision(12);
        std::cout << "\nRunning Experiment 3: ODE Solving Comparison...\n";

        exp3_results << "=================================================================\n";
        exp3_results << "EXPERIMENT 3: ODE Solution U'(t) = -iH*U(t) Comparison\n";
        exp3_results << "N=" << N << ", T=" << T << "\n";
        exp3_results << "Test matrix: H_test_matrix.txt\n";
        exp3_results << "Reference: Exact solution U(T) = exp(-i*H*T) with Eigen Pade\n";
        exp3_results << "-----------------------------------------------------------------\n";
        exp3_results << "Method | Max Element Diff | Max Eig Diff | Time (ms)\n";
        exp3_results << "-----------------------------------------------------------------\n";

        // Exact solution U(T) = exp(-i * H_test * T) - reference
        CMatrix Omega_exact = matrix_ops::mat_scale(H_test, complexd(0.0, -T));
        auto start_exact = chrono::high_resolution_clock::now();
        CMatrix U_exact = matrix_ops::expm_pade_eigen(Omega_exact, N);
        auto end_exact = chrono::high_resolution_clock::now();
        auto duration_exact = chrono::duration_cast<chrono::microseconds>(end_exact - start_exact);
        double time_exact_ms = duration_exact.count() / 1000.0;

        std::cout << "Exact solution computed in " << time_exact_ms << " ms\n";

        // Method 1: Runge-Kutta with dt = 1e-5
        {
            double rk_dt = 1e-5;
            int num_steps = (int)std::round(T / rk_dt);
            if (num_steps == 0) num_steps = 1;

            std::cout << "Runge-Kutta will perform " << num_steps << " steps with dt=" << rk_dt << std::endl;

            auto start_rk = chrono::high_resolution_clock::now();

            CMatrix U_rk = utils::eye(N);
            for (int step = 0; step < num_steps; ++step) {
                U_rk = runge_kutta_simple::runge_kutta_step(H_test, U_rk, rk_dt, N);
            }

            auto end_rk = chrono::high_resolution_clock::now();
            auto duration_rk = chrono::duration_cast<chrono::microseconds>(end_rk - start_rk);
            double time_rk_ms = duration_rk.count() / 1000.0;

            // Compare with exact solution
            CMatrix diff_rk(N * N, complexd(0.0, 0.0));
            matrix_ops::mat_sub(U_rk, U_exact, diff_rk, N);

            double max_el_diff_rk = matrix_ops::max_element_diff(U_rk, U_exact, N);
            double max_eig_diff_rk = matrix_ops::max_eigenvalue_modulus_hermitian(diff_rk, N);

            exp3_results << "Runge-Kutta dt=1e-5 | " << max_el_diff_rk << " | " << max_eig_diff_rk << " | " << time_rk_ms << "\n";
            std::cout << "Runge-Kutta (dt=1e-5) computed in " << time_rk_ms << " ms\n";
        }

        // Method 2: Magnus ACC + Chebyshev exponentiation
        {
            // For constant H, we don't need many samples - just a few points for integration
            // Omega = integral_0^T (-i * H) dt = -i * H * T
            double integration_dt = 0.01;
            vector<CMatrix> A_samples = generate_samples(0.0, T, integration_dt, N,
                H_test, CMatrix(N*N, complexd(0.0, 0.0)), 0.0, 0.0); // eps0=0, W=0 for constant H

            std::cout << "Generated " << A_samples.size() << " samples for Magnus integration" << std::endl;

            auto start_magnus = chrono::high_resolution_clock::now();

            // Compute Omega using Magnus ACC (order 2 is sufficient for constant H)
            CMatrix Omega_magnus = magnus_ACC(A_samples, integration_dt, N, 2);

            // Exponentiate using Chebyshev method (M=15 terms is enough)
            CMatrix U_magnus = matrix_ops::expm_cheb(Omega_magnus, N, 15);

            auto end_magnus = chrono::high_resolution_clock::now();
            auto duration_magnus = chrono::duration_cast<chrono::microseconds>(end_magnus - start_magnus);
            double time_magnus_ms = duration_magnus.count() / 1000.0;

            // Compare with exact solution
            CMatrix diff_magnus(N * N, complexd(0.0, 0.0));
            matrix_ops::mat_sub(U_magnus, U_exact, diff_magnus, N);

            double max_el_diff_magnus = matrix_ops::max_element_diff(U_magnus, U_exact, N);
            double max_eig_diff_magnus = matrix_ops::max_eigenvalue_modulus_hermitian(diff_magnus, N);

            exp3_results << "Magnus ACC + Chebyshev | " << max_el_diff_magnus << " | " << max_eig_diff_magnus << " | " << time_magnus_ms << "\n";
            std::cout << "Magnus ACC + Chebyshev computed in " << time_magnus_ms << " ms\n";
        }

        exp3_results << "-----------------------------------------------------------------\n";
        exp3_results << "Note: Reference solution uses exp(-i*H*T) with Eigen Pade approximation\n";
        exp3_results.close();
        std::cout << "Experiment 3 results written to experiment3_results.txt\n";
    }
    

    // =====================================================================
    // ============= EXPERIMENT 4: Omega Stability Analysis ===============
    // =====================================================================
    {
        std::ofstream exp4_results("experiment4_results.txt");
        if (!exp4_results.is_open()) {
            std::cerr << "Error opening experiment4_results.txt" << std::endl;
            return 1;
        }
        exp4_results << std::scientific << std::setprecision(12);
        std::cout << "\nRunning Experiment 4: Omega Stability Analysis...\n";

        exp4_results << "=================================================================\n";
        exp4_results << "EXPERIMENT 4: Omega Stability Analysis for Spectral Matrices\n";
        exp4_results << "N=" << N << ", dt=" << dt << "\n";
        exp4_results << "Test matrices: spectral matrices with condition numbers 10, 100, 1000, 10000\n";
        exp4_results << "Reference: Omega = H * dt (exact for constant H)\n";
        exp4_results << "-----------------------------------------------------------------\n";
        exp4_results << "Kappa | Method | Max Element Diff | Max Eig Diff | Time (ms)\n";
        exp4_results << "-----------------------------------------------------------------\n";

        std::vector<std::string> methods = {"Classic", "ACC", "Recursive", "Formula 3.14"};
        std::vector<double> condition_numbers = {10.0, 100.0, 1000.0, 10000.0};

        for (size_t mat_idx = 0; mat_idx < spectral_matrices.size(); ++mat_idx) {
            double kappa = condition_numbers[mat_idx];
            const CMatrix& H = spectral_matrices[mat_idx];

            std::cout << "Testing matrix with condition number " << kappa << "\n";

            // Reference Omega = H * dt
            CMatrix Omega_ref = matrix_ops::mat_scale(H, complexd(dt, 0.0));

            // Test each Magnus method
            CMatrix H0_input = H;
            CMatrix H_mod_zero(N * N, complexd(0.0, 0.0));
            const double eps0 = 0.0;
            const double W = 0.0;
            const double T_int = dt;
            const double integration_dt = dt * 0.01;

            // Classic Magnus (k=2)
            {
                auto start = chrono::high_resolution_clock::now();
                CMatrix Omega_calc = magnus_classic(0.0, T_int, integration_dt, N, H0_input, H_mod_zero, eps0, W, 2);
                auto end = chrono::high_resolution_clock::now();
                auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
                double time_ms = duration.count() / 1000.0;

                CMatrix diff(N * N, complexd(0.0, 0.0));
                matrix_ops::mat_sub(Omega_calc, Omega_ref, diff, N);

                double max_el_diff = matrix_ops::max_element_diff(Omega_calc, Omega_ref, N);
                double max_eig_diff = matrix_ops::max_eigenvalue_modulus_hermitian(diff, N);

                exp4_results << kappa << " | Classic | " << max_el_diff << " | " << max_eig_diff << " | " << time_ms << "\n";
            }

            // ACC Magnus (k=5)
            {
                vector<CMatrix> A_samples = generate_samples(0.0, T_int, integration_dt, N, H0_input, H_mod_zero, eps0, W);

                auto start = chrono::high_resolution_clock::now();
                CMatrix Omega_calc = magnus_ACC(A_samples, integration_dt, N, 5);
                auto end = chrono::high_resolution_clock::now();
                auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
                double time_ms = duration.count() / 1000.0;

                CMatrix diff(N * N, complexd(0.0, 0.0));
                matrix_ops::mat_sub(Omega_calc, Omega_ref, diff, N);

                double max_el_diff = matrix_ops::max_element_diff(Omega_calc, Omega_ref, N);
                double max_eig_diff = matrix_ops::max_eigenvalue_modulus_hermitian(diff, N);

                exp4_results << kappa << " | ACC | " << max_el_diff << " | " << max_eig_diff << " | " << time_ms << "\n";
            }

            // Recursive Magnus (k=10)
            {
                auto start = chrono::high_resolution_clock::now();
                CMatrix Omega_calc = magnus_expansion(0.0, T_int, integration_dt, N, H0_input, H_mod_zero, eps0, W, 10);
                auto end = chrono::high_resolution_clock::now();
                auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
                double time_ms = duration.count() / 1000.0;

                CMatrix diff(N * N, complexd(0.0, 0.0));
                matrix_ops::mat_sub(Omega_calc, Omega_ref, diff, N);

                double max_el_diff = matrix_ops::max_element_diff(Omega_calc, Omega_ref, N);
                double max_eig_diff = matrix_ops::max_eigenvalue_modulus_hermitian(diff, N);

                exp4_results << kappa << " | Recursive | " << max_el_diff << " | " << max_eig_diff << " | " << time_ms << "\n";
            }

            // Formula 3.14
            {
                CMatrix H0_anti_herm = matrix_ops::mat_scale(H, complexd(0.0, -1.0)); // -i * H

                auto start = chrono::high_resolution_clock::now();
                CMatrix Omega_calc = magnus_3_14(0.0, T_int, integration_dt, N, H0_anti_herm, H_mod_zero, eps0, W);
                auto end = chrono::high_resolution_clock::now();
                auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
                double time_ms = duration.count() / 1000.0;

                CMatrix diff(N * N, complexd(0.0, 0.0));
                matrix_ops::mat_sub(Omega_calc, Omega_ref, diff, N);

                double max_el_diff = matrix_ops::max_element_diff(Omega_calc, Omega_ref, N);
                double max_eig_diff = matrix_ops::max_eigenvalue_modulus_hermitian(diff, N);

                exp4_results << kappa << " | Formula 3.14 | " << max_el_diff << " | " << max_eig_diff << " | " << time_ms << "\n";
            }
        }

        exp4_results << "-----------------------------------------------------------------\n";
        exp4_results.close();
        std::cout << "Experiment 4 results written to experiment4_results.txt\n";
    }

    // =====================================================================
    // ============= EXPERIMENT 5: Matrix Exponent Stability ==============
    // =====================================================================
    {
        std::ofstream exp5_results("experiment5_results.txt");
        if (!exp5_results.is_open()) {
            std::cerr << "Error opening experiment5_results.txt" << std::endl;
            return 1;
        }
        exp5_results << std::scientific << std::setprecision(12);
        std::cout << "\nRunning Experiment 5: Matrix Exponent Stability Analysis...\n";

        exp5_results << "=================================================================\n";
        exp5_results << "EXPERIMENT 5: Matrix Exponent Stability Analysis for Spectral Matrices\n";
        exp5_results << "N=" << N << "\n";
        exp5_results << "Test matrices: spectral matrices with condition numbers 10, 100, 1000, 10000\n";
        exp5_results << "Reference: exp(-i*H) = V*exp(-i*D)*V^(-1)\n";
        exp5_results << "-----------------------------------------------------------------\n";
        exp5_results << "Kappa | Method | Max Element Diff | Max Eig Diff | Time (ms)\n";
        exp5_results << "-----------------------------------------------------------------\n";

        std::vector<double> condition_numbers = {10.0, 100.0, 1000.0, 10000.0};

        for (size_t mat_idx = 0; mat_idx < spectral_matrices.size(); ++mat_idx) {
            double kappa = condition_numbers[mat_idx];
            const CMatrix& H = spectral_matrices[mat_idx];
            const CMatrix& V = spectral_V_matrices[mat_idx];
            const CMatrix& D = spectral_D_matrices[mat_idx];

            std::cout << "Testing matrix exponent with condition number " << kappa << "\n";

            // Reference: exp(-i*H) = V * exp(-i*D) * V^(-1)
            // Since D is diagonal, exp(-i*D) is diagonal with exp(-i*D_ii)
            CMatrix exp_minus_i_D(N * N, complexd(0.0, 0.0));
            for (int i = 0; i < N; ++i) {
                double lambda = D[idx(i, i, N)].real();
                complexd exp_val = exp(complexd(0.0, -lambda));
                exp_minus_i_D[idx(i, i, N)] = exp_val;
            }

            CMatrix Vh = matrix_ops::dagger(V, N);
            CMatrix V_exp_D(N * N, complexd(0.0, 0.0));
            matrix_ops::matmul(V, exp_minus_i_D, V_exp_D, N);
            CMatrix U_ref(N * N, complexd(0.0, 0.0));
            matrix_ops::matmul(V_exp_D, Vh, U_ref, N);

            // Test different exponentiation methods
            CMatrix Omega_test = matrix_ops::mat_scale(H, complexd(0.0, -1.0)); // -i * H

            // Taylor method (terms = 30)
            {
                auto start = chrono::high_resolution_clock::now();
                CMatrix U_calc = matrix_ops::expm_taylor(Omega_test, N, 30);
                auto end = chrono::high_resolution_clock::now();
                auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
                double time_ms = duration.count() / 1000.0;

                CMatrix diff(N * N, complexd(0.0, 0.0));
                matrix_ops::mat_sub(U_calc, U_ref, diff, N);

                double max_el_diff = matrix_ops::max_element_diff(U_calc, U_ref, N);
                double max_eig_diff = matrix_ops::max_eigenvalue_modulus_hermitian(diff, N);

                exp5_results << kappa << " | Taylor | " << max_el_diff << " | " << max_eig_diff << " | " << time_ms << "\n";
            }

            // Chebyshev method (M = 25)
            {
                auto start = chrono::high_resolution_clock::now();
                CMatrix U_calc = matrix_ops::expm_cheb(Omega_test, N, 25);
                auto end = chrono::high_resolution_clock::now();
                auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
                double time_ms = duration.count() / 1000.0;

                CMatrix diff(N * N, complexd(0.0, 0.0));
                matrix_ops::mat_sub(U_calc, U_ref, diff, N);

                double max_el_diff = matrix_ops::max_element_diff(U_calc, U_ref, N);
                double max_eig_diff = matrix_ops::max_eigenvalue_modulus_hermitian(diff, N);

                exp5_results << kappa << " | Chebyshev | " << max_el_diff << " | " << max_eig_diff << " | " << time_ms << "\n";
            }

            // Eigen Pade (reference method)
            {
                auto start = chrono::high_resolution_clock::now();
                CMatrix U_calc = matrix_ops::expm_pade_eigen(Omega_test, N);
                auto end = chrono::high_resolution_clock::now();
                auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
                double time_ms = duration.count() / 1000.0;

                CMatrix diff(N * N, complexd(0.0, 0.0));
                matrix_ops::mat_sub(U_calc, U_ref, diff, N);

                double max_el_diff = matrix_ops::max_element_diff(U_calc, U_ref, N);
                double max_eig_diff = matrix_ops::max_eigenvalue_modulus_hermitian(diff, N);

                exp5_results << kappa << " | Eigen Pade | " << max_el_diff << " | " << max_eig_diff << " | " << time_ms << "\n";
            }
        }

        exp5_results << "-----------------------------------------------------------------\n";
        exp5_results.close();
        std::cout << "Experiment 5 results written to experiment5_results.txt\n";
    }

    // =====================================================================
    // ============= EXPERIMENT 6: ODE Stability Analysis ================
    // =====================================================================
    {
        std::ofstream exp6_results("experiment6_results.txt");
        if (!exp6_results.is_open()) {
            std::cerr << "Error opening experiment6_results.txt" << std::endl;
            return 1;
        }
        exp6_results << std::scientific << std::setprecision(12);
        std::cout << "\nRunning Experiment 6: ODE Stability Analysis...\n";

        exp6_results << "=================================================================\n";
        exp6_results << "EXPERIMENT 6: ODE Solution Stability Analysis for Spectral Matrices\n";
        exp6_results << "N=" << N << ", T=" << T << "\n";
        exp6_results << "Test matrices: spectral matrices with condition numbers 10, 100, 1000, 10000\n";
        exp6_results << "Reference: Exact solution U(T) = exp(-i*H*T) with Eigen Pade\n";
        exp6_results << "-----------------------------------------------------------------\n";
        exp6_results << "Kappa | Method | Max Element Diff | Max Eig Diff | Time (ms)\n";
        exp6_results << "-----------------------------------------------------------------\n";

        std::vector<double> condition_numbers = {10.0, 100.0, 1000.0, 10000.0};

        for (size_t mat_idx = 0; mat_idx < spectral_matrices.size(); ++mat_idx) {
            double kappa = condition_numbers[mat_idx];
            const CMatrix& H = spectral_matrices[mat_idx];

            std::cout << "Testing ODE stability for matrix with condition number " << kappa << "\n";

            // Reference solution U(T) = exp(-i * H * T)
            CMatrix Omega_exact = matrix_ops::mat_scale(H, complexd(0.0, -T));
            CMatrix U_exact = matrix_ops::expm_pade_eigen(Omega_exact, N);

            // Method 1: Runge-Kutta with dt = 1e-5
            {
                double rk_dt = 1e-5;
                int num_steps = (int)std::round(T / rk_dt);
                if (num_steps == 0) num_steps = 1;

                auto start_rk = chrono::high_resolution_clock::now();

                CMatrix U_rk = utils::eye(N);
                for (int step = 0; step < num_steps; ++step) {
                    U_rk = runge_kutta_simple::runge_kutta_step(H, U_rk, rk_dt, N);
                }

                auto end_rk = chrono::high_resolution_clock::now();
                auto duration_rk = chrono::duration_cast<chrono::microseconds>(end_rk - start_rk);
                double time_rk_ms = duration_rk.count() / 1000.0;

                // Compare with exact solution
                CMatrix diff_rk(N * N, complexd(0.0, 0.0));
                matrix_ops::mat_sub(U_rk, U_exact, diff_rk, N);

                double max_el_diff_rk = matrix_ops::max_element_diff(U_rk, U_exact, N);
                double max_eig_diff_rk = matrix_ops::max_eigenvalue_modulus_hermitian(diff_rk, N);

                exp6_results << kappa << " | Runge-Kutta dt=1e-5 | " << max_el_diff_rk << " | " << max_eig_diff_rk << " | " << time_rk_ms << "\n";
            }

            // Method 2: Magnus ACC + Chebyshev exponentiation
            {
                // Use reasonable integration step for Magnus
                double integration_dt = 0.01;
                vector<CMatrix> A_samples = generate_samples(0.0, T, integration_dt, N,
                    H, CMatrix(N*N, complexd(0.0, 0.0)), 0.0, 0.0); // eps0=0, W=0 for constant H

                auto start_magnus = chrono::high_resolution_clock::now();

                // Compute Omega using Magnus ACC (order 2 is sufficient)
                CMatrix Omega_magnus = magnus_ACC(A_samples, integration_dt, N, 2);

                // Exponentiate using Chebyshev method (M=15 terms)
                CMatrix U_magnus = matrix_ops::expm_cheb(Omega_magnus, N, 15);

                auto end_magnus = chrono::high_resolution_clock::now();
                auto duration_magnus = chrono::duration_cast<chrono::microseconds>(end_magnus - start_magnus);
                double time_magnus_ms = duration_magnus.count() / 1000.0;

                // Compare with exact solution
                CMatrix diff_magnus(N * N, complexd(0.0, 0.0));
                matrix_ops::mat_sub(U_magnus, U_exact, diff_magnus, N);

                double max_el_diff_magnus = matrix_ops::max_element_diff(U_magnus, U_exact, N);
                double max_eig_diff_magnus = matrix_ops::max_eigenvalue_modulus_hermitian(diff_magnus, N);

                exp6_results << kappa << " | Magnus ACC + Chebyshev | " << max_el_diff_magnus << " | " << max_eig_diff_magnus << " | " << time_magnus_ms << "\n";
            }
        }

        exp6_results << "-----------------------------------------------------------------\n";
        exp6_results << "Note: Testing ODE solution stability for poorly conditioned matrices\n";
        exp6_results.close();
        std::cout << "Experiment 6 results written to experiment6_results.txt\n";
    }

    results_txt.close();
    std::cout << "\nExperiment results written to magnus_experiment_results.txt" << std::endl;

    return 0;
}