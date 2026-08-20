#pragma once
#include <vector>
#include <complex>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <device_launch_parameters.h>

// Внешняя функция для расчета 1-нормы на CPU (используется для оценки)
#include "MatrixOperations.h" 

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

using complexd = std::complex<double>;
using CMatrix = std::vector<complexd>;

namespace cuda_optimized {

    // === CUDA ЯДРА ===

    // 1. Слитое ядро: строит Omega, масштабирует её в X и инициализирует матрицы Чебышева
    __global__ void fused_omega_cheb_init_kernel(
        const cuDoubleComplex* H0, const cuDoubleComplex* Hmod, const cuDoubleComplex* Ccomm,
        double f_int, double c_coeff, double dt, double scale_factor,
        cuDoubleComplex C0, cuDoubleComplex C1,
        cuDoubleComplex* X, cuDoubleComplex* Tk_prev, cuDoubleComplex* Tk_curr, cuDoubleComplex* Res,
        int N)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int size = N * N;
        if (idx < size) {
            cuDoubleComplex h0 = H0[idx];
            cuDoubleComplex hm = Hmod[idx];
            cuDoubleComplex cc = Ccomm[idx];

            // Omega = -i * dt * H0 - i * f_int * Hmod + c_coeff * Ccomm
            // Напоминание: -i * (a + i*b) = b - i*a
            cuDoubleComplex term1 = make_cuDoubleComplex(cuCimag(h0) * dt, -cuCreal(h0) * dt);
            cuDoubleComplex term2 = make_cuDoubleComplex(cuCimag(hm) * f_int, -cuCreal(hm) * f_int);
            cuDoubleComplex term3 = make_cuDoubleComplex(cuCreal(cc) * c_coeff, cuCimag(cc) * c_coeff);

            cuDoubleComplex omega = cuCadd(cuCadd(term1, term2), term3);

            // ВАЖНО: Умножаем на 'i' (0 + 1i), чтобы повернуть спектр на вещественную ось!
            // X = omega * i * scale_factor
            // Математика: (re + im*i) * i = -im + re*i
            cuDoubleComplex x_val = make_cuDoubleComplex(
                -cuCimag(omega) * scale_factor,
                cuCreal(omega) * scale_factor
            );
            X[idx] = x_val;

            int row = idx / N;
            int col = idx % N;
            cuDoubleComplex ident = make_cuDoubleComplex((row == col) ? 1.0 : 0.0, 0.0);

            // Инициализация Чебышева
            Tk_prev[idx] = ident;
            Tk_curr[idx] = x_val;

            // Res = C0*I + C1*X
            cuDoubleComplex r = cuCadd(cuCmul(C0, ident), cuCmul(C1, x_val));
            Res[idx] = r;
        }
    }

    // 2. Ядро для шага полинома Чебышева: Tk_next = 2 * X_Tk - Tk_prev
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

            // Сдвиг для следующей итерации (перекладываем прямо в памяти)
            Tk_prev[idx] = curr;
            Tk_curr[idx] = next;
        }
    }

    // === КЛАСС РЕШАТЕЛЯ ===
    class MagnusChebyshevGPU {
    private:
        int N, size;
        size_t bytes;
        cublasHandle_t handle;

        // Данные на GPU
        cuDoubleComplex* d_H0, * d_Hmod, * d_Ccomm;
        cuDoubleComplex* d_X, * d_Tk_prev, * d_Tk_curr;
        cuDoubleComplex* d_Res, * d_Temp, * d_U_total;

        // Вспомогательная функция умножения
        void gpu_matmul(cuDoubleComplex* d_A, cuDoubleComplex* d_B, cuDoubleComplex* d_C) {
            cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
            cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
            CHECK_CUBLAS(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N));
        }

    public:
        MagnusChebyshevGPU(int N, const CMatrix& H0, const CMatrix& Hmod) : N(N), size(N* N) {
            bytes = size * sizeof(cuDoubleComplex);
            cublasCreate(&handle);

            // Выделяем память один раз
            cudaMalloc(&d_H0, bytes);
            cudaMalloc(&d_Hmod, bytes);
            cudaMalloc(&d_Ccomm, bytes);
            cudaMalloc(&d_X, bytes);
            cudaMalloc(&d_Tk_prev, bytes);
            cudaMalloc(&d_Tk_curr, bytes);
            cudaMalloc(&d_Res, bytes);
            cudaMalloc(&d_Temp, bytes);
            cudaMalloc(&d_U_total, bytes);

            // Копируем H0 и Hmod
            cudaMemcpy(d_H0, H0.data(), bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_Hmod, Hmod.data(), bytes, cudaMemcpyHostToDevice);

            // Предвычисляем коммутатор Ccomm = [H0, Hmod] на CPU и кидаем в GPU
            CMatrix Ccomm = matrix_ops::commutator(H0, Hmod, N);
            cudaMemcpy(d_Ccomm, Ccomm.data(), bytes, cudaMemcpyHostToDevice);
        }

        ~MagnusChebyshevGPU() {
            cudaFree(d_H0); cudaFree(d_Hmod); cudaFree(d_Ccomm);
            cudaFree(d_X); cudaFree(d_Tk_prev); cudaFree(d_Tk_curr);
            cudaFree(d_Res); cudaFree(d_Temp); cudaFree(d_U_total);
            cublasDestroy(handle);
        }

        CMatrix solve(double t0, double T, double dt, double eps0, double W, int M = 15) {
            // Инициализация U_total единичной матрицей
            CMatrix U_init(size, complexd(0.0, 0.0));
            for (int i = 0; i < N; ++i) U_init[i * N + i] = complexd(1.0, 0.0);
            cudaMemcpy(d_U_total, U_init.data(), bytes, cudaMemcpyHostToDevice);

            // Оценка нормы для Чебышева (чтобы не гонять данные CPU <-> GPU внутри цикла)
            // norm(Omega) <= dt * (norm(H0) + |f| * norm(Hmod)) + (dt^2 / 12) * |df| * norm(Ccomm)
            // Грубая, но надежная оценка максимума:
            CMatrix H0(size), Hmod(size), Ccomm(size);
            cudaMemcpy(H0.data(), d_H0, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(Hmod.data(), d_Hmod, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(Ccomm.data(), d_Ccomm, bytes, cudaMemcpyDeviceToHost);

            double norm_H0 = matrix_ops::mat_one_norm(H0, N);
            double norm_Hmod = matrix_ops::mat_one_norm(Hmod, N);
            double norm_Ccomm = matrix_ops::mat_one_norm(Ccomm, N);

            double max_f = eps0;
            double max_df = 2.0 * eps0;
            double max_norm_omega = dt * (norm_H0 + max_f * norm_Hmod) + (dt * dt / 12.0) * max_df * norm_Ccomm;

            int s = 0;
            double alpha = max_norm_omega;
            while (alpha > 1.0) { alpha /= 2.0; s++; }
            if (alpha < 1e-14) alpha = 1e-14;

            // Подготовка коэффициентов Чебышева на CPU
            double scale_factor = 1.0 / (std::pow(2.0, s) * alpha);

            // Используем _j0, _j1, _jn из <cmath> (специфичные для MSVC/POSIX), 
            // чтобы не зависеть от стандарта C++17
            cuDoubleComplex C0 = make_cuDoubleComplex(_j0(alpha), 0.0);
            cuDoubleComplex C1 = make_cuDoubleComplex(0.0, -2.0 * _j1(alpha)); // Исправлена опечатка

            std::vector<cuDoubleComplex> Ck(M + 1);
            for (int k = 2; k <= M; ++k) {
                double Jk = _jn(k, alpha); // Исправлен вызов функции Бесселя
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
            int num_steps = (int)std::round((T - t0) / dt);

            // ================= ОСНОВНОЙ ЦИКЛ НА GPU =================
            for (int step = 0; step < num_steps; ++step) {
                double t = t0 + step * dt;

                // Вычисляем косинусы на CPU и просто передаем как аргументы (это бесплатно)
                double f0 = eps0 * cos(W * t);
                double fm = eps0 * cos(W * (t + dt / 2.0));
                double fe = eps0 * cos(W * (t + dt));

                double f_int = (dt / 6.0) * (f0 + 4.0 * fm + fe);
                double c_coeff = (dt * dt / 12.0) * (fe - f0);

                // 1. Сборка Omega и Инициализация Чебышева (Все в одном ядре!)
                fused_omega_cheb_init_kernel <<<blocks, threads>>> (
                    d_H0, d_Hmod, d_Ccomm, f_int, c_coeff, dt, scale_factor,
                    C0, C1, d_X, d_Tk_prev, d_Tk_curr, d_Res, N);
                CHECK_CUDA(cudaGetLastError());


                // 2. Итерации Чебышева
                for (int k = 2; k <= M; ++k) {
                    gpu_matmul(d_X, d_Tk_curr, d_Temp); // d_Temp = X * Tk_curr
                    cheb_step_update_kernel <<<blocks, threads>>> (
                        d_Temp, d_Tk_prev, d_Tk_curr, d_Res, Ck[k], size);
                }

                // 3. Возведение в квадрат (Squaring)
                for (int i = 0; i < s; ++i) {
                    gpu_matmul(d_Res, d_Res, d_Temp);
                    std::swap(d_Res, d_Temp); // Просто меняем указатели, без копирования!
                }

                // 4. Обновление общего состояния
                gpu_matmul(d_Res, d_U_total, d_Temp);
                std::swap(d_U_total, d_Temp);
            }

            // Скачиваем готовый результат
            CMatrix U_final(size);
            cudaMemcpy(U_final.data(), d_U_total, bytes, cudaMemcpyDeviceToHost);
            return U_final;
        }
    };
}
