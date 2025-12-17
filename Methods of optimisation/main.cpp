#include <iostream>
#include <complex>
#include <vector>
#include <iomanip>
#include <chrono>
#include "Utils.h"
#include "MatrixOperations.h"

using namespace std;
using namespace utils;
using namespace matrix_ops;
using complexd = complex<double>;
using CMatrix = vector<complexd>;

void print_matrix(const CMatrix& mat, int N, const string& name) {
    cout << name << ":\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            complexd val = mat[i * N + j];
            cout << "(" << setprecision(6) << val.real() << ", " << val.imag() << ") ";
        }
        cout << "\n";
    }
    cout << "\n";
}

int main() {
    const int N = 10;
    const double dt = 1e-5;
    const double T = 1.0;
    const double diag_min = 10.0;

    // Set output to file
    std::ofstream results_txt("magnus_experiment_results.txt");
    if (!results_txt.is_open()) {
        std::cerr << "Error opening magnus_experiment_results.txt" << std::endl;
        return 1;
    }
    results_txt << std::scientific << std::setprecision(12);
    std::cout << std::scientific << std::setprecision(12);

    // =====================================================================
    // Generate constant Hamiltonian H_const
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
    // ============= EXPERIMENT 2: Integration over the full interval ========
    // =====================================================================
    results_txt << "\n=================================================================\n";
    results_txt << "EXPERIMENT 2: Comparison of Numerical Integration Accuracy (H = const)\n";
    results_txt << "N=" << N << ", dt=" << dt << ", T=" << T << "\n";

    // 1. Reference U_exact
    // Omega_ref_T = -i * H * T
    CMatrix Omega_ref_T = matrix_ops::mat_scale(H_const, complexd(0.0, -T));
    const int expm_ref_terms = 30;
    CMatrix U_exact = matrix_ops::expm_taylor(Omega_ref_T, N, expm_ref_terms);

    results_txt << "Reference: Omega = -i*H*T, Expm Method = Taylor(" << expm_ref_terms << " terms)\n";
    results_txt << "Omega Calculation Method: Recursive Expansion (10 levels)\n";
    results_txt << "-----------------------------------------------------------------\n";
    results_txt << "Expm Terms (K) | Max Element Diff (U) | Mat One Norm Diff (U_diff)\n";
    results_txt << "-----------------------------------------------------------------\n";

    // Omega(dt) - Recursive expansion, 10 levels
    // Use H0_input (Hermitian) because magnus_expansion handles -i internally via generate_samples
    const int omega_level = 10;
    CMatrix Omega_dt_magnus = magnus_expansion(0.0, dt, integration_dt, N, H0_input, H_mod_zero, eps0, W, omega_level);

    int num_steps = (int)std::round(T / dt);
    if (num_steps == 0) num_steps = 1;

    for (int k_taylor = 1; k_taylor <= 30; ++k_taylor) {
        // Step 1: U(dt) = expm_taylor(Omega_dt_magnus, N, k_taylor)
        CMatrix U_dt_calc = matrix_ops::expm_taylor(Omega_dt_magnus, N, k_taylor);

        // Step 2: U(T) = U(dt)^num_steps
        CMatrix U_calc = utils::eye(N);
        CMatrix temp(N * N, complexd(0.0, 0.0));

        for (int step = 0; step < num_steps; ++step) {
            matrix_ops::matmul(U_dt_calc, U_calc, temp, N);
            U_calc = temp;
        }

        CMatrix diff_U(N * N, complexd(0.0, 0.0));
        matrix_ops::mat_sub(U_calc, U_exact, diff_U, N);

        double max_el_diff = matrix_ops::max_element_diff(U_calc, U_exact, N);
        double mat_one_norm_diff = matrix_ops::mat_one_norm(diff_U, N);

        results_txt << "Taylor | " << std::setprecision(16) << k_taylor << " | " << max_el_diff << " | " << mat_one_norm_diff << "\n";
    }

    // =====================================================================
    // ============= EXPERIMENT 3: Chebyshev exp(Omega) Accuracy ============
    // =====================================================================
    results_txt << "\n=================================================================\n";
    results_txt << "EXPERIMENT 3: Accuracy of exp(Omega) using Chebyshev method\n";
    results_txt << "N=" << N << ", dt=" << dt << "\n";
    results_txt << "Reference: exp(Omega) = Taylor(30 terms)\n";
    results_txt << "-----------------------------------------------------------------\n";
    results_txt << "Method | K/M | Max Elem Diff | Max Eig Diff | Time (ms)\n";
    results_txt << "-----------------------------------------------------------------\n";

    // 1. Omega for full interval dt (same as in experiment 1)
    CMatrix Omega_dt = Omega_ref_dt;   // = -i * H * dt

    // 2. Reference exponential: Taylor with 30 terms
    CMatrix U_ref = matrix_ops::expm_taylor(Omega_dt, N, 30);

    // --------------- Taylor comparison (K = 1..10) --------------------
    for (int k = 1; k <= 10; k++)
    {
        auto start = chrono::high_resolution_clock::now();

        CMatrix U_taylor = matrix_ops::expm_taylor(Omega_dt, N, k);

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        double time_ms = duration.count() / 1000.0;

        CMatrix diff(N * N, complexd(0.0, 0.0));
        matrix_ops::mat_sub(U_taylor, U_ref, diff, N);

        double max_el_diff = matrix_ops::max_element_diff(U_taylor, U_ref, N);
        double max_eig_diff = matrix_ops::max_eigenvalue_modulus_hermitian(diff, N);

        results_txt << "Taylor | " << k << " | "
            << max_el_diff << " | "
            << max_eig_diff << " | " << time_ms << "\n";
    }

    // --------------- Chebyshev comparison (M = 1..25) --------------------
    for (int M = 1; M <= 25; M++)
    {
        auto start = chrono::high_resolution_clock::now();

        CMatrix U_cheb = expm_cheb(Omega_dt, N, M);

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        double time_ms = duration.count() / 1000.0;

        CMatrix diff(N * N, complexd(0.0, 0.0));
        matrix_ops::mat_sub(U_cheb, U_ref, diff, N);

        double max_el_diff = matrix_ops::max_element_diff(U_cheb, U_ref, N);
        double max_eig_diff = matrix_ops::max_eigenvalue_modulus_hermitian(diff, N);

        results_txt << "Cheb | " << M << " | "
            << max_el_diff << " | "
            << max_eig_diff << " | " << time_ms << "\n";
    }

    results_txt << "-----------------------------------------------------------------\n";

    results_txt.close();
    std::cout << "\nExperiment results written to magnus_experiment_results.txt" << std::endl;

    return 0;
}