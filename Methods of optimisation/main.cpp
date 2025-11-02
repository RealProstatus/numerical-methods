#include <mkl.h>
#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <fstream>

#include "Utils.h"
#include "MatrixOperations.h"
#include "RungeKutt.h"

using namespace std;
using namespace utils;
using namespace matrix_ops;
using namespace runge_kutta_simple;

// ------ Trapezoidal integration for matrix functions ------
vector<CMatrix> generate_samples(double t0, double t1, double dt, int N,
    const CMatrix& H0, const CMatrix& H_mod,
    double eps0, double W) {
    vector<CMatrix> samples;
    int n_steps = (int)((t1 - t0) / dt) + 1;

    cout << "Generating " << n_steps << " time samples with dt=" << dt << endl;

    for (int i = 0; i < n_steps; ++i) {
        double t = t0 + i * dt;
        double f_t = eps0 * cos(W * t);

        // A(t) = -i * (H0 + f(t) * H_mod)
        CMatrix A_t((size_t)N * N, complexd(0.0, 0.0));
        for (size_t k = 0; k < A_t.size(); ++k) {
            A_t[k] = complexd(0.0, -1.0) * (H0[k] + f_t * H_mod[k]);
        }

        samples.push_back(A_t);
    }

    return samples;
}

CMatrix trapezoidal_integral(const vector<CMatrix>& samples, double dt, int N) {
    CMatrix result((size_t)N * N, complexd(0.0, 0.0));
    int n_samples = samples.size();

    if (n_samples < 2) return result;

    // First and last samples with weight 0.5
    for (size_t k = 0; k < result.size(); ++k) {
        result[k] = (samples[0][k] + samples[n_samples - 1][k]) * 0.5;
    }

    // Middle samples with weight 1.0
    for (int i = 1; i < n_samples - 1; ++i) {
        for (size_t k = 0; k < result.size(); ++k) {
            result[k] += samples[i][k];
        }
    }

    // Multiply by dt
    mat_scale_inplace(result, N, complexd(dt, 0.0));

    return result;
}

// ------ Magnus expansion up to Omega_10 ------
CMatrix magnus_expansion(double t0, double t1, double integration_dt, int N,
    const CMatrix& H0, const CMatrix& H_mod,
    double eps0, double W, int max_n = 10) {
    vector<CMatrix> Omega_n(max_n + 1);

    // Генерируем выборки A(t)
    vector<CMatrix> A_samples = generate_samples(t0, t1, integration_dt, N, H0, H_mod, eps0, W);

    // Ω₁(t) = ∫₀ᵗ A(τ)dτ
    Omega_n[1] = trapezoidal_integral(A_samples, integration_dt, N);

    cout << "Computing Magnus expansion up to order " << max_n << "..." << endl;

    // Вычисляем члены высшего порядка согласно Wikipedia
    for (int n = 2; n <= max_n; ++n) {
        cout << "Computing Omega_" << n << "..." << endl;

        CMatrix sum_term((size_t)N * N, complexd(0.0, 0.0));

        for (int j = 1; j <= n - 1; ++j) {
            double B_j = matrix_ops::bernoulli_number(j);

            // Пропускаем нулевые числа Бернулли
            if (std::abs(B_j) < 1e-12) continue;

            // Вычисляем S_n^(j) согласно рекуррентной формуле
            vector<CMatrix> S_samples = matrix_ops::compute_S_n_j(n, j, Omega_n, A_samples, N);

            // ∫₀ᵗ S_n^(j)(τ)dτ
            CMatrix integral_term = trapezoidal_integral(S_samples, integration_dt, N);

            // B_j/j! * ∫₀ᵗ S_n^(j)(τ)dτ
            double factorial = 1.0;
            for (int k = 1; k <= j; ++k) factorial *= k;

            mat_scale_inplace(integral_term, N, complexd(B_j / factorial, 0.0));
            mat_add(sum_term, integral_term, sum_term, N);
        }

        Omega_n[n] = sum_term;
    }

    // Суммируем все члены разложения
    CMatrix Omega_total((size_t)N * N, complexd(0.0, 0.0));
    for (int n = 1; n <= max_n; ++n) {
        cout << "||Omega_" << n << "|| = " << mat_one_norm(Omega_n[n], N) << endl;
        mat_add(Omega_total, Omega_n[n], Omega_total, N);
    }

    return Omega_total;
}

// ------ Main function implementing the requested steps ------
int main() {
    const int N = 10;
    const double hbar = 1.0;
    const double eps0 = 1.0;
    const double W = 1.0;
    const double delta_t = 1e-5;
    const double integration_dt = 1e-8;

    cout << "=== Step 0: Generating Hermitian matrix H ===" << endl;

    // Generate Hermitian Hamiltonian H0
    CMatrix H0 = generate_hermitian_matrix(N, 5.0);

    // Generate modulation matrix H_mod (also Hermitian)
    CMatrix H_mod = generate_hermitian_matrix(N, 1.0);

    // Save matrices to shared resource (files)
    save_matrix(H0, N, "H0_matrix.txt");
    save_matrix(H_mod, N, "H_mod_matrix.txt");

    cout << "Hermitian matrices saved to H0_matrix.txt and H_mod_matrix.txt" << endl;

    cout << "\n=== Step 1: Runge-Kutta single step ===" << endl;

    // Create time-dependent Hamiltonian H(t) = H0 + eps0*cos(W*t)*H_mod
    double t_rk = 0.0;
    CMatrix H_current((size_t)N * N, complexd(0.0, 0.0));
    for (size_t k = 0; k < H_current.size(); ++k) {
        H_current[k] = H0[k] + eps0 * cos(W * t_rk) * H_mod[k];
    }

    // Initial state: identity matrix
    CMatrix U_rk = eye(N);

    // Perform one Runge-Kutta step using the provided function
    U_rk = runge_kutta_step(H_current, U_rk, delta_t, N);

    cout << "Runge-Kutta step completed. U(dt) computed." << endl;

    cout << "\n=== Step 2: Magnus expansion ===" << endl;

    cout << "2.1 Computing Magnus expansion up to Omega_10..." << endl;
    CMatrix Omega = magnus_expansion(0.0, delta_t, integration_dt, N, H0, H_mod, eps0, W, 10);

    cout << "2.2 Omega(0, " << delta_t << ") computed" << endl;

    cout << "2.3 Computing U(0," << delta_t << ") = exp(i*Omega) via Taylor series..." << endl;

    // Используем Omega напрямую без дополнительного i
    CMatrix U_magnus = expm_taylor(Omega, N, 30);

    cout << "Magnus expansion completed." << endl;

    // Compare results
    cout << "\n=== Comparison ===" << endl;

    // Compute difference between methods
    CMatrix diff((size_t)N * N, complexd(0.0, 0.0));
    mat_sub(U_rk, U_magnus, diff, N);

    double max_diff = 0.0;
    for (size_t k = 0; k < diff.size(); ++k) {
        max_diff = max(max_diff, abs(diff[k]));
    }

    cout << "Maximum difference between RK and Magnus methods: " << max_diff << endl;

    // Save results
    save_matrix(U_rk, N, "U_runge_kutta.txt");
    save_matrix(U_magnus, N, "U_magnus.txt");
    save_matrix(Omega, N, "Omega_magnus.txt");

    cout << "\nResults saved to files:" << endl;
    cout << "- U_runge_kutta.txt: Runge-Kutta result" << endl;
    cout << "- U_magnus.txt: Magnus expansion result" << endl;
    cout << "- Omega_magnus.txt: Magnus operator Omega" << endl;

    return 0;
}
//Использование метода Магнуса - Чебышева для решения системы линейных ОДУ первого порядка С ИСПОЛЬЗОВАНИЕ ТЕХНОЛОГИЙ ПАРАЛЛЕЛЬНОГО ПРОГРАММИРОВАНИЯ