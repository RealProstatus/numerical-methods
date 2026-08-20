#include <iostream>
#include <complex>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cmath>
#include "Utils.h"
#include "MatrixOperations.h"
#include "RungeKutt.h"
#include <clocale>

// Подключаем наш новый ультра-оптимизированный класс
#include "CudaMagnusChebyshev.cuh"
#include "AllComputationalMethods.h"

using namespace std;
using namespace utils;
using namespace matrix_ops;
using namespace runge_kutta_simple;

const double PI = 3.14159265358979323846;
/*
vector<CMatrix> generate_samples(const vector<double>& times, int N, const CMatrix& H0, const CMatrix& H_mod, double eps0, double W) {
    vector<CMatrix> samples;
    samples.reserve(times.size());
    for (double t : times) {
        double f_t = eps0 * cos(W * t);
        CMatrix A_t((size_t)N * N, complexd(0.0, 0.0));
        for (size_t k = 0; k < A_t.size(); ++k) {
            A_t[k] = complexd(0.0, -1.0) * (H0[k] + f_t * H_mod[k]);
        }
        samples.push_back(A_t);
    }
    return samples;
}

vector<double> generate_time_grid(double t0, double t1, double dt) {
    vector<double> times;
    double t = t0;
    while (t < t1 - 1e-12) {
        times.push_back(t);
        t += dt;
    }
    if (times.empty() || std::abs(times.back() - t1) > 1e-12) {
        times.push_back(t1);
    }
    return times;
}

CMatrix trapezoidal_integral(const vector<CMatrix>& samples, const vector<double>& times, int N) {
    CMatrix result((size_t)N * N, complexd(0.0, 0.0));
    int n_samples = samples.size();
    if (n_samples < 2) return result;

    for (int i = 0; i < n_samples - 1; ++i) {
        double dt_i = times[i + 1] - times[i];
        for (size_t k = 0; k < result.size(); ++k) {
            result[k] += (samples[i][k] + samples[i + 1][k]) * 0.5 * dt_i;
        }
    }
    return result;
}

CMatrix simpson_integral(const vector<CMatrix>& samples, const vector<double>& times, int N) {
    CMatrix result((size_t)N * N, complexd(0.0, 0.0));
    int n_samples = samples.size();
    if (n_samples < 3) return trapezoidal_integral(samples, times, N);

    double expected_dt = times[1] - times[0];
    double last_dt = times.back() - times[times.size() - 2];
    bool last_is_uniform = (std::abs(last_dt - expected_dt) < 1e-10);

    int max_uniform_points = last_is_uniform ? n_samples : n_samples - 1;
    int simpson_points = (max_uniform_points % 2 == 1) ? max_uniform_points : max_uniform_points - 1;

    for (size_t k = 0; k < result.size(); ++k) result[k] = samples[0][k] + samples[simpson_points - 1][k];
    for (int i = 1; i < simpson_points - 1; ++i) {
        double w = (i % 2 == 1) ? 4.0 : 2.0;
        for (size_t k = 0; k < result.size(); ++k) result[k] += samples[i][k] * w;
    }
    mat_scale_inplace(result, N, complexd(expected_dt / 3.0, 0.0));

    for (int i = simpson_points - 1; i < n_samples - 1; ++i) {
        double dt_i = times[i + 1] - times[i];
        for (size_t k = 0; k < result.size(); ++k) {
            result[k] += (samples[i][k] + samples[i + 1][k]) * 0.5 * dt_i;
        }
    }
    return result;
}

CMatrix magnus_ACC(const vector<CMatrix>& A_full_samples, const vector<double>& times, int N, int max_order) {
    double T_total = times.back() - times.front();
    CMatrix Omega_total(N * N, complexd(0.0, 0.0));

    CMatrix Omega1 = simpson_integral(A_full_samples, times, N);
    mat_add(Omega_total, Omega1, Omega_total, N);
    if (max_order < 2) return Omega_total;

    int M = A_full_samples.size();
    if (M < 3) return Omega_total;

    CMatrix A_0 = A_full_samples[0];
    CMatrix A_end = A_full_samples[M - 1];

    if (max_order >= 2) {
        CMatrix comm = commutator(A_end, A_0, N);
        mat_scale_inplace(comm, N, complexd(T_total * T_total / 12.0, 0.0));
        mat_add(Omega_total, comm, Omega_total, N);
    }
    return Omega_total;
}

CMatrix piecewise_magnus_solver(double t0, double T, double macro_dt, int N, const CMatrix& H0, const CMatrix& H_mod, double eps0, double W, int order = 4) {
    CMatrix U_total = utils::eye(N);
    int num_steps = (int)std::round((T - t0) / macro_dt);

    for (int step = 0; step < num_steps; ++step) {
        double t_curr = t0 + step * macro_dt;
        double t_next = std::min(t_curr + macro_dt, T);

        int sub_steps = (order >= 6) ? 6 : 2;
        double micro_dt = (t_next - t_curr) / sub_steps;

        vector<double> step_times = generate_time_grid(t_curr, t_next, micro_dt);
        vector<CMatrix> step_samples = generate_samples(step_times, N, H0, H_mod, eps0, W);

        CMatrix Omega_local = magnus_ACC(step_samples, step_times, N, order);
        CMatrix U_local = matrix_ops::expm_cheb(Omega_local, N, 15);

        CMatrix temp((size_t)N * N, complexd(0.0, 0.0));
        matrix_ops::matmul(U_local, U_total, temp, N);
        U_total = temp;
    }
    return U_total;
}*/

int main() {
    setlocale(LC_ALL, "Russian");

    // Идентификатор пространства имен для логов
    std::cout << "[spichek_d_magnus_cuda] Инициализация среды...\n\n";

    const int N = 100;
    const double T = 0.1;

    const double W = 2.0 * PI;
    const double eps0 = 1.0;
    const double diag_min = 10.0;

    std::cout << "=========================================================\n";
    std::cout << "  Подготовка к экспериментам N=" << N << "\n";
    std::cout << "=========================================================\n";

    // Генерация тестовых матриц
    CMatrix H0_test = utils::generate_hermitian_matrix(N, diag_min);
    CMatrix Hmod_test = utils::generate_hermitian_matrix(N, diag_min / 2.0);

    // =========================================================
    // 1. Вычисление эталонного решения (Сверхточный RK4)
    // =========================================================
    double dt_ref = 1e-6; // Уменьшили шаг эталона для абсолютной уверенности
    std::cout << "Вычисление эталонного решения (RK4, dt = " << scientific << dt_ref << ")...\n";

    CMatrix U_ref = utils::eye(N);
    int ref_steps = (int)std::round(T / dt_ref);

    auto start_ref = chrono::high_resolution_clock::now();
    for (int step = 0; step < ref_steps; ++step) {
        U_ref = runge_kutta_step(step * dt_ref, H0_test, Hmod_test, W, U_ref, dt_ref, N);
    }
    auto end_ref = chrono::high_resolution_clock::now();
    std::cout << "Эталон готов за " << chrono::duration_cast<chrono::microseconds>(end_ref - start_ref).count() / 1000.0 << " мс.\n\n";

    // =========================================================
    // Параметры для сравнения рабочих методов
    // =========================================================
    // RK4 требует мелкого шага для стабильности
    double dt_rk = 1e-4;

    // Магнус может позволить себе шаг крупнее, 
    // но он обязан лежать внутри радиуса сходимости (||H|| * dt < 3.4)
    double dt_magnus = 1e-3;

    std::cout << "=========================================================\n";
    std::cout << "  СРАВНЕНИЕ МЕТОДОВ\n";
    std::cout << "  RK4 dt = " << dt_rk << "  |  Magnus GPU dt = " << dt_magnus << "\n";
    std::cout << "=========================================================\n";

    // =========================================================
    // 2. Метод Рунге-Кутты 4 (CPU)
    // =========================================================
    CMatrix U_rk = utils::eye(N);
    int rk_steps = (int)std::round(T / dt_rk);

    auto start_rk = chrono::high_resolution_clock::now();
    for (int step = 0; step < rk_steps; ++step) {
        U_rk = runge_kutta_step(step * dt_rk, H0_test, Hmod_test, W, U_rk, dt_rk, N);
    }
    auto end_rk = chrono::high_resolution_clock::now();
    double time_rk = chrono::duration_cast<chrono::microseconds>(end_rk - start_rk).count() / 1000.0;

    // =========================================================
    // 3. УЛЬТРА-ОПТИМИЗИРОВАННЫЙ МАГНУС (GPU CUDA)
    // =========================================================
    auto start_magnus_new = chrono::high_resolution_clock::now();

    // Инициализация solver'а (передача H0, Hmod и вычисление коммутатора в GPU)
    cuda_optimized::MagnusChebyshevGPU solver(N, H0_test, Hmod_test);

    // Интегрирование полностью на стороне GPU
    CMatrix U_magnus_new = solver.solve(0.0, T, dt_magnus, eps0, W, 15);
    std::cout << "U_magnus_new[0] = " << U_magnus_new[0] << "\n";
    std::cout << "U_magnus_new[1] = " << U_magnus_new[1] << "\n";

    auto end_magnus_new = chrono::high_resolution_clock::now();
    double time_magnus_new = chrono::duration_cast<chrono::microseconds>(end_magnus_new - start_magnus_new).count() / 1000.0;

    // =========================================================
    // Подсчет ошибок и вывод результатов
    // =========================================================
    double err_rk = matrix_ops::max_element_diff(U_rk, U_ref, N);
    double err_new = matrix_ops::max_element_diff(U_magnus_new, U_ref, N);

    std::cout << left << setw(25) << "Метод" << setw(15) << "Шаг (dt)" << setw(15) << "Время (мс)" << "Ошибка (Max Diff)\n";
    std::cout << "---------------------------------------------------------------------------\n";
    std::cout << left << setw(25) << "Runge-Kutta 4 (CPU)" << setw(15) << scientific << setprecision(1) << dt_rk
        << setw(15) << fixed << setprecision(3) << time_rk << scientific << setprecision(6) << err_rk << "\n";

    std::cout << left << setw(25) << "Magnus ACC 4 (GPU)" << setw(15) << scientific << setprecision(1) << dt_magnus
        << setw(15) << fixed << setprecision(3) << time_magnus_new << scientific << setprecision(6) << err_new << "\n";
    std::cout << "---------------------------------------------------------------------------\n";

    return 0;
}
