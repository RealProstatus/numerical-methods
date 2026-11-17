#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <device_launch_parameters.h>

#include "MatrixOperations.h"

#define CHECK_CUDA(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err_) + \
            " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while (0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status_ = (call); \
    if (status_ != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error(std::string("cuBLAS Error: code ") + std::to_string(static_cast<int>(status_)) + \
            " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while (0)

using complexd = std::complex<double>;
using CMatrix = std::vector<complexd>;

namespace cuda_optimized {

    inline double bessel_j_host(int k, double x) {
#if defined(_WIN32)
        if (k == 0) return _j0(x);
        if (k == 1) return _j1(x);
        return _jn(k, x);
#else
        if (k == 0) return j0(x);
        if (k == 1) return j1(x);
        return jn(k, x);
#endif
    }

    // === CUDA ЯДРА ===

    // 1. Масштабирование матрицы на комплексный скаляр
    __global__ void scale_matrix_kernel(
        const cuDoubleComplex* src, cuDoubleComplex* dst, cuDoubleComplex alpha, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            dst[idx] = cuCmul(src[idx], alpha);
        }
    }

    // 2. Сумма двух матриц: dst = A + B
    __global__ void add_matrices_kernel(
        const cuDoubleComplex* A, const cuDoubleComplex* B, cuDoubleComplex* dst, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            dst[idx] = cuCadd(A[idx], B[idx]);
        }
    }

    // 3. Вычисление сумм модулей столбцов для 1-нормы на GPU (без выгрузки матрицы на CPU)
    __global__ void column_abs_sums_kernel(
        const cuDoubleComplex* M, double* col_sums, int N)
    {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col < N) {
            double sum = 0.0;
            for (int row = 0; row < N; ++row) {
                sum += cuCabs(M[row * N + col]);
            }
            col_sums[col] = sum;
        }
    }

    // 4. Слитое ядро 6-го порядка: сборка Omega из предвычисленных коммутаторов,
    // поворот спектра i*Omega*scale_factor и инициализация полиномов Чебышева
    __global__ void fused_omega6_cheb_init_kernel(
        const cuDoubleComplex* A0,
        const cuDoubleComplex* A1,
        const cuDoubleComplex* C1,
        const cuDoubleComplex* C2,
        const cuDoubleComplex* C3,
        const cuDoubleComplex* C4,
        const cuDoubleComplex* C56,
        const cuDoubleComplex* C7,
        double c_A0, double c_A1, double c_C1, double c_C2,
        double c_C3, double c_C4, double c_C56, double c_C7,
        double scale_factor,
        cuDoubleComplex C0_cheb, cuDoubleComplex C1_cheb,
        cuDoubleComplex* X, cuDoubleComplex* Tk_prev, cuDoubleComplex* Tk_curr, cuDoubleComplex* Res,
        int N)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int size = N * N;
        if (idx < size) {
            cuDoubleComplex a0 = A0[idx];
            cuDoubleComplex a1 = A1[idx];
            cuDoubleComplex c1 = C1[idx];
            cuDoubleComplex c2 = C2[idx];
            cuDoubleComplex c3 = C3[idx];
            cuDoubleComplex c4 = C4[idx];
            cuDoubleComplex c56 = C56[idx];
            cuDoubleComplex c7 = C7[idx];

            double re = c_A0 * cuCreal(a0) + c_A1 * cuCreal(a1)
                + c_C1 * cuCreal(c1) + c_C2 * cuCreal(c2)
                + c_C3 * cuCreal(c3) + c_C4 * cuCreal(c4)
                + c_C56 * cuCreal(c56) + c_C7 * cuCreal(c7);

            double im = c_A0 * cuCimag(a0) + c_A1 * cuCimag(a1)
                + c_C1 * cuCimag(c1) + c_C2 * cuCimag(c2)
                + c_C3 * cuCimag(c3) + c_C4 * cuCimag(c4)
                + c_C56 * cuCimag(c56) + c_C7 * cuCimag(c7);

            // X = Omega * i * scale_factor = (-im + i * re) * scale_factor
            cuDoubleComplex x_val = make_cuDoubleComplex(-im * scale_factor, re * scale_factor);
            X[idx] = x_val;

            int row = idx / N;
            int col = idx % N;
            cuDoubleComplex ident = make_cuDoubleComplex((row == col) ? 1.0 : 0.0, 0.0);

            Tk_prev[idx] = ident;
            Tk_curr[idx] = x_val;
            Res[idx] = cuCadd(cuCmul(C0_cheb, ident), cuCmul(C1_cheb, x_val));
        }
    }

    // 5. Ядро для шага полинома Чебышева: Tk_next = 2 * X_Tk - Tk_prev
    __global__ void cheb_step_update_kernel(
        const cuDoubleComplex* X_Tk, cuDoubleComplex* Tk_prev, cuDoubleComplex* Tk_curr,
        cuDoubleComplex* Res, cuDoubleComplex Ck, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            cuDoubleComplex xtk = X_Tk[idx];
            cuDoubleComplex prev = Tk_prev[idx];
            cuDoubleComplex curr = Tk_curr[idx];

            cuDoubleComplex next = make_cuDoubleComplex(
                2.0 * cuCreal(xtk) - cuCreal(prev),
                2.0 * cuCimag(xtk) - cuCimag(prev)
            );

            Res[idx] = cuCadd(Res[idx], cuCmul(Ck, next));
            Tk_prev[idx] = curr;
            Tk_curr[idx] = next;
        }
    }

    struct StepCoeffs {
        double c_A0, c_A1, c_C1, c_C2, c_C3, c_C4, c_C56, c_C7;
    };

    // === КЛАСС РЕШАТЕЛЯ ===
    class MagnusChebyshevGPU {
    private:
        int N = 0;
        int size = 0;
        size_t bytes = 0;
        cublasHandle_t handle = nullptr;

        // Антиэрмитовы генераторы и предвычисленные коммутаторы на GPU
        cuDoubleComplex* d_A0 = nullptr;
        cuDoubleComplex* d_A1 = nullptr;
        cuDoubleComplex* d_C1 = nullptr;
        cuDoubleComplex* d_C2 = nullptr;
        cuDoubleComplex* d_C3 = nullptr;
        cuDoubleComplex* d_C4 = nullptr;
        cuDoubleComplex* d_C56 = nullptr;
        cuDoubleComplex* d_C7 = nullptr;

        // Рабочие буферы Чебышева и пропагатора
        cuDoubleComplex* d_X = nullptr;
        cuDoubleComplex* d_Tk_prev = nullptr;
        cuDoubleComplex* d_Tk_curr = nullptr;
        cuDoubleComplex* d_Res = nullptr;
        cuDoubleComplex* d_Temp = nullptr;
        cuDoubleComplex* d_U_total = nullptr;
        double* d_col_sums = nullptr;

        // Кэшированные 1-нормы генераторов
        double norm_A0 = 0.0, norm_A1 = 0.0;
        double norm_C1 = 0.0, norm_C2 = 0.0, norm_C3 = 0.0;
        double norm_C4 = 0.0, norm_C56 = 0.0, norm_C7 = 0.0;

        void allocate_buffers() {
            bytes = static_cast<size_t>(size) * sizeof(cuDoubleComplex);
            CHECK_CUBLAS(cublasCreate(&handle));

            CHECK_CUDA(cudaMalloc(&d_A0, bytes));
            CHECK_CUDA(cudaMalloc(&d_A1, bytes));
            CHECK_CUDA(cudaMalloc(&d_C1, bytes));
            CHECK_CUDA(cudaMalloc(&d_C2, bytes));
            CHECK_CUDA(cudaMalloc(&d_C3, bytes));
            CHECK_CUDA(cudaMalloc(&d_C4, bytes));
            CHECK_CUDA(cudaMalloc(&d_C56, bytes));
            CHECK_CUDA(cudaMalloc(&d_C7, bytes));
            CHECK_CUDA(cudaMalloc(&d_X, bytes));
            CHECK_CUDA(cudaMalloc(&d_Tk_prev, bytes));
            CHECK_CUDA(cudaMalloc(&d_Tk_curr, bytes));
            CHECK_CUDA(cudaMalloc(&d_Res, bytes));
            CHECK_CUDA(cudaMalloc(&d_Temp, bytes));
            CHECK_CUDA(cudaMalloc(&d_U_total, bytes));
            CHECK_CUDA(cudaMalloc(&d_col_sums, static_cast<size_t>(N) * sizeof(double)));
        }

        // Умножение Row-Major матриц через cuBLAS (C = A * B <=> C^T = B^T * A^T)
        void gpu_matmul(const cuDoubleComplex* d_A, const cuDoubleComplex* d_B, cuDoubleComplex* d_C) {
            cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
            cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
            CHECK_CUBLAS(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N));
        }

        // Коммутатор Row-Major матриц: d_Out = [d_A, d_B] = d_A * d_B - d_B * d_A
        void gpu_commutator(const cuDoubleComplex* d_A, const cuDoubleComplex* d_B, cuDoubleComplex* d_Out) {
            cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
            cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
            cuDoubleComplex minus_one = make_cuDoubleComplex(-1.0, 0.0);

            CHECK_CUBLAS(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, d_B, N, d_A, N, &zero, d_Out, N));
            CHECK_CUBLAS(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &minus_one, d_A, N, d_B, N, &one, d_Out, N));
        }

        double compute_gpu_one_norm(const cuDoubleComplex* d_M) {
            int threads = 256;
            int blocks = (N + threads - 1) / threads;
            column_abs_sums_kernel << <blocks, threads >> > (d_M, d_col_sums, N);
            CHECK_CUDA(cudaGetLastError());

            std::vector<double> h_sums(N, 0.0);
            CHECK_CUDA(cudaMemcpy(h_sums.data(), d_col_sums, static_cast<size_t>(N) * sizeof(double), cudaMemcpyDeviceToHost));
            return *std::max_element(h_sums.begin(), h_sums.end());
        }

        void build_generators_on_gpu(double hbar) {
            int threads = 256;
            int blocks = (size + threads - 1) / threads;
            cuDoubleComplex gen_factor = make_cuDoubleComplex(0.0, -1.0 / hbar);

            // A0 = (-i / hbar) * H0, A1 = (-i / hbar) * Hmod
            scale_matrix_kernel << <blocks, threads >> > (d_A0, d_A0, gen_factor, size);
            scale_matrix_kernel << <blocks, threads >> > (d_A1, d_A1, gen_factor, size);
            CHECK_CUDA(cudaGetLastError());

            // Предвычисляем коммутаторы C1..C7 один раз на GPU
            gpu_commutator(d_A0, d_A1, d_C1);   // C1 = [A0, A1]
            gpu_commutator(d_A0, d_C1, d_C2);   // C2 = [A0, C1]
            gpu_commutator(d_A1, d_C1, d_C3);   // C3 = [A1, C1]
            gpu_commutator(d_A0, d_C2, d_C4);   // C4 = [A0, C2]
            gpu_commutator(d_A1, d_C2, d_C56);  // C5 = [A1, C2]
            gpu_commutator(d_A0, d_C3, d_Temp); // C6 = [A0, C3]
            add_matrices_kernel << <blocks, threads >> > (d_C56, d_Temp, d_C56, size); // C56 = C5 + C6
            CHECK_CUDA(cudaGetLastError());
            gpu_commutator(d_A1, d_C3, d_C7);   // C7 = [A1, C3]

            // Кэшируем 1-нормы на этапе инициализации
            norm_A0 = compute_gpu_one_norm(d_A0);
            norm_A1 = compute_gpu_one_norm(d_A1);
            norm_C1 = compute_gpu_one_norm(d_C1);
            norm_C2 = compute_gpu_one_norm(d_C2);
            norm_C3 = compute_gpu_one_norm(d_C3);
            norm_C4 = compute_gpu_one_norm(d_C4);
            norm_C56 = compute_gpu_one_norm(d_C56);
            norm_C7 = compute_gpu_one_norm(d_C7);
        }

        void run_propagation(double t0, double T, double dt, double eps0, double W, int M) {
            CMatrix U_init(size, complexd(0.0, 0.0));
            for (int i = 0; i < N; ++i) U_init[i * N + i] = complexd(1.0, 0.0);
            CHECK_CUDA(cudaMemcpy(d_U_total, U_init.data(), bytes, cudaMemcpyHostToDevice));

            const double h = dt;
            const double h2 = h * h;
            const double h3 = h2 * h;
            const double h5 = h3 * h2;
            const double abs_eps = std::abs(eps0);
            const double abs_W = std::abs(W);

            // Строгая верхняя оценка ||Omega||_1 для всех шагов
            const double max_c_A0 = h;
            const double max_c_A1 = h * abs_eps + (h3 / 60.0) * abs_eps * abs_W * abs_W;
            const double max_c_C1 = (h2 / 10.0) * abs_eps + (h3 / 30.0) * abs_eps * abs_W;
            const double max_c_C2 = (h5 / 720.0) * abs_eps * abs_W * abs_W;
            const double max_c_C3 = (h5 / 180.0) * abs_eps * abs_eps * abs_W * abs_W;
            const double max_c_C4 = (h5 / 720.0) * abs_eps * abs_W;
            const double max_c_C56 = (h5 / 720.0) * abs_eps * abs_eps * abs_W;
            const double max_c_C7 = (h5 / 720.0) * abs_eps * abs_eps * abs_eps * abs_W;

            double max_norm_omega = max_c_A0 * norm_A0 + max_c_A1 * norm_A1
                + max_c_C1 * norm_C1 + max_c_C2 * norm_C2
                + max_c_C3 * norm_C3 + max_c_C4 * norm_C4
                + max_c_C56 * norm_C56 + max_c_C7 * norm_C7;

            int s = 0;
            double alpha = max_norm_omega;
            while (alpha > 1.0) { alpha /= 2.0; s++; }
            if (alpha < 1e-14) alpha = 1e-14;

            double scale_factor = 1.0 / (std::pow(2.0, s) * alpha);

            cuDoubleComplex C0 = make_cuDoubleComplex(bessel_j_host(0, alpha), 0.0);
            cuDoubleComplex C1 = make_cuDoubleComplex(0.0, -2.0 * bessel_j_host(1, alpha));

            std::vector<cuDoubleComplex> Ck(M + 1);
            for (int k = 2; k <= M; ++k) {
                double Jk = bessel_j_host(k, alpha);
                int rem = k % 4;
                complexd i_pow_k;
                if (rem == 0) i_pow_k = complexd(1.0, 0.0);
                else if (rem == 1) i_pow_k = complexd(0.0, -1.0);
                else if (rem == 2) i_pow_k = complexd(-1.0, 0.0);
                else i_pow_k = complexd(0.0, 1.0);

                Ck[k] = make_cuDoubleComplex(2.0 * Jk * i_pow_k.real(), 2.0 * Jk * i_pow_k.imag());
            }

            int threads = 256;
            int blocks = (size + threads - 1) / threads;
            int num_steps = static_cast<int>(std::round((T - t0) / dt));

            // ================= ОСНОВНОЙ ЦИКЛ НА GPU (6-Й ПОРЯДОК) =================
            for (int step = 0; step < num_steps; ++step) {
                double t = t0 + step * dt;
                double t_half = t + 0.5 * dt;
                double t_end = t + dt;

                double f_t0 = eps0 * std::cos(W * t);
                double f_half = eps0 * std::cos(W * t_half);
                double f_t1 = eps0 * std::cos(W * t_end);
                double fp_half = -eps0 * W * std::sin(W * t_half);
                double f2_half = -eps0 * W * W * std::cos(W * t_half);

                double c_A0 = h;
                double c_A1 = h * (0.1 * f_t0 + 0.8 * f_half + 0.1 * f_t1) + (h3 / 60.0) * f2_half;
                double c_C1 = -(h2 / 20.0) * (f_t1 - f_t0) - (h3 / 30.0) * fp_half;
                double c_C2 = (h5 / 720.0) * f2_half;
                double c_C3 = (h5 / 720.0) * f_half * f2_half - (h5 / 240.0) * (fp_half * fp_half);
                double c_C4 = (h5 / 720.0) * fp_half;
                double c_C56 = (h5 / 720.0) * f_half * fp_half;
                double c_C7 = (h5 / 720.0) * (f_half * f_half) * fp_half;

                fused_omega6_cheb_init_kernel << <blocks, threads >> > (
                    d_A0, d_A1, d_C1, d_C2, d_C3, d_C4, d_C56, d_C7,
                    c_A0, c_A1, c_C1, c_C2, c_C3, c_C4, c_C56, c_C7,
                    scale_factor, C0, C1,
                    d_X, d_Tk_prev, d_Tk_curr, d_Res, N
                    );
                CHECK_CUDA(cudaGetLastError());

                for (int k = 2; k <= M; ++k) {
                    gpu_matmul(d_X, d_Tk_curr, d_Temp);
                    cheb_step_update_kernel << <blocks, threads >> > (
                        d_Temp, d_Tk_prev, d_Tk_curr, d_Res, Ck[k], size
                        );
                }

                for (int i = 0; i < s; ++i) {
                    gpu_matmul(d_Res, d_Res, d_Temp);
                    std::swap(d_Res, d_Temp);
                }

                gpu_matmul(d_Res, d_U_total, d_Temp);
                std::swap(d_U_total, d_Temp);
            }
        }

        void run_propagation_coeffs(const std::vector<StepCoeffs>& steps_coeffs, int M) {
            CMatrix U_init(size, complexd(0.0, 0.0));
            for (int i = 0; i < N; ++i) U_init[i * N + i] = complexd(1.0, 0.0);
            CHECK_CUDA(cudaMemcpy(d_U_total, U_init.data(), bytes, cudaMemcpyHostToDevice));

            double max_norm_omega = 0.0;
            for (const auto& sc : steps_coeffs) {
                double step_bound = std::abs(sc.c_A0) * norm_A0 + std::abs(sc.c_A1) * norm_A1
                    + std::abs(sc.c_C1) * norm_C1 + std::abs(sc.c_C2) * norm_C2
                    + std::abs(sc.c_C3) * norm_C3 + std::abs(sc.c_C4) * norm_C4
                    + std::abs(sc.c_C56) * norm_C56 + std::abs(sc.c_C7) * norm_C7;
                max_norm_omega = std::max(max_norm_omega, step_bound);
            }

            int s = 0;
            double alpha = max_norm_omega;
            while (alpha > 1.0) { alpha /= 2.0; s++; }
            if (alpha < 1e-14) alpha = 1e-14;

            double scale_factor = 1.0 / (std::pow(2.0, s) * alpha);

            cuDoubleComplex C0 = make_cuDoubleComplex(bessel_j_host(0, alpha), 0.0);
            cuDoubleComplex C1 = make_cuDoubleComplex(0.0, -2.0 * bessel_j_host(1, alpha));

            std::vector<cuDoubleComplex> Ck(M + 1);
            for (int k = 2; k <= M; ++k) {
                double Jk = bessel_j_host(k, alpha);
                int rem = k % 4;
                complexd i_pow_k;
                if (rem == 0) i_pow_k = complexd(1.0, 0.0);
                else if (rem == 1) i_pow_k = complexd(0.0, -1.0);
                else if (rem == 2) i_pow_k = complexd(-1.0, 0.0);
                else i_pow_k = complexd(0.0, 1.0);
                Ck[k] = make_cuDoubleComplex(2.0 * Jk * i_pow_k.real(), 2.0 * Jk * i_pow_k.imag());
            }

            int threads = 256;
            int blocks = (size + threads - 1) / threads;

            for (const auto& sc : steps_coeffs) {
                fused_omega6_cheb_init_kernel << <blocks, threads >> > (
                    d_A0, d_A1, d_C1, d_C2, d_C3, d_C4, d_C56, d_C7,
                    sc.c_A0, sc.c_A1, sc.c_C1, sc.c_C2, sc.c_C3, sc.c_C4, sc.c_C56, sc.c_C7,
                    scale_factor, C0, C1,
                    d_X, d_Tk_prev, d_Tk_curr, d_Res, N
                    );
                CHECK_CUDA(cudaGetLastError());

                for (int k = 2; k <= M; ++k) {
                    gpu_matmul(d_X, d_Tk_curr, d_Temp);
                    cheb_step_update_kernel << <blocks, threads >> > (
                        d_Temp, d_Tk_prev, d_Tk_curr, d_Res, Ck[k], size
                        );
                }

                for (int i = 0; i < s; ++i) {
                    gpu_matmul(d_Res, d_Res, d_Temp);
                    std::swap(d_Res, d_Temp);
                }

                gpu_matmul(d_Res, d_U_total, d_Temp);
                std::swap(d_U_total, d_Temp);
            }
        }

    public:
        // Конструктор из Host-векторов CMatrix
        MagnusChebyshevGPU(int N, const CMatrix& H0, const CMatrix& Hmod, double hbar = 1.0)
            : N(N), size(N* N)
        {
            allocate_buffers();
            CHECK_CUDA(cudaMemcpy(d_A0, H0.data(), bytes, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_A1, Hmod.data(), bytes, cudaMemcpyHostToDevice));
            build_generators_on_gpu(hbar);
        }

        // Zero-Copy конструктор напрямую из Device-указателей VRAM (CuPy / PyTorch / DLPack)
        MagnusChebyshevGPU(int N, const cuDoubleComplex* d_H0_ext, const cuDoubleComplex* d_Hmod_ext, double hbar = 1.0)
            : N(N), size(N* N)
        {
            allocate_buffers();
            CHECK_CUDA(cudaMemcpy(d_A0, d_H0_ext, bytes, cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(d_A1, d_Hmod_ext, bytes, cudaMemcpyDeviceToDevice));
            build_generators_on_gpu(hbar);
        }

        ~MagnusChebyshevGPU() {
            if (d_A0) cudaFree(d_A0);
            if (d_A1) cudaFree(d_A1);
            if (d_C1) cudaFree(d_C1);
            if (d_C2) cudaFree(d_C2);
            if (d_C3) cudaFree(d_C3);
            if (d_C4) cudaFree(d_C4);
            if (d_C56) cudaFree(d_C56);
            if (d_C7) cudaFree(d_C7);
            if (d_X) cudaFree(d_X);
            if (d_Tk_prev) cudaFree(d_Tk_prev);
            if (d_Tk_curr) cudaFree(d_Tk_curr);
            if (d_Res) cudaFree(d_Res);
            if (d_Temp) cudaFree(d_Temp);
            if (d_U_total) cudaFree(d_U_total);
            if (d_col_sums) cudaFree(d_col_sums);
            if (handle) cublasDestroy(handle);
        }

        // Zero-Copy запись результата напрямую в целевой буфер VRAM (d_U_out)
        void solve_to_device(double t0, double T, double dt, double eps0, double W, int M, cuDoubleComplex* d_U_out) {
            run_propagation(t0, T, dt, eps0, W, M);
            CHECK_CUDA(cudaMemcpy(d_U_out, d_U_total, bytes, cudaMemcpyDeviceToDevice));
        }

        // Выгрузка результата на Host (для CLI и NumPy)
        CMatrix solve(double t0, double T, double dt, double eps0, double W, int M = 15) {
            run_propagation(t0, T, dt, eps0, W, M);
            CMatrix U_final(size);
            CHECK_CUDA(cudaMemcpy(U_final.data(), d_U_total, bytes, cudaMemcpyDeviceToHost));
            return U_final;
        }

        CMatrix solve_coeffs(const std::vector<StepCoeffs>& steps_coeffs, int M = 16) {
            run_propagation_coeffs(steps_coeffs, M);
            CMatrix U_final(size);
            CHECK_CUDA(cudaMemcpy(U_final.data(), d_U_total, bytes, cudaMemcpyDeviceToHost));
            return U_final;
        }
    };
}