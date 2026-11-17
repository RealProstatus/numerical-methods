#include <cmath>
#include <complex>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <tuple>
#include <utility>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "magnus/solvers.hpp"
#include "CudaMagnusChebyshev.cuh"

namespace magnus {

    bool has_cuda_backend() noexcept {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        return (err == cudaSuccess && device_count > 0);
    }

    void propagate_cuda_raw(
        std::size_t dim,
        const Complex* d_H0_ptr,
        const Complex* d_H1_ptr,
        double t_start,
        double t_final,
        int steps,
        double epsilon,
        double omega,
        const SolverOptions& opts,
        Complex* d_U_out_ptr,
        bool pointers_on_device
    ) {
        if (dim == 0) {
            throw std::invalid_argument("propagate_cuda_raw: Matrix dimension cannot be 0.");
        }
        if (!d_H0_ptr || !d_H1_ptr || !d_U_out_ptr) {
            throw std::invalid_argument("propagate_cuda_raw: Null pointer passed.");
        }

        const std::size_t size = dim * dim;

        if (std::abs(t_final - t_start) < 1e-15) {
            std::vector<Complex> h_eye(size, Complex(0.0, 0.0));
            for (std::size_t i = 0; i < dim; ++i) {
                h_eye[i * dim + i] = Complex(1.0, 0.0);
            }
            if (pointers_on_device) {
                cudaError_t err = cudaMemcpy(d_U_out_ptr, h_eye.data(), size * sizeof(Complex), cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    throw std::runtime_error(std::string("CUDA memcpy failed: ") + cudaGetErrorString(err));
                }
            }
            else {
                std::memcpy(d_U_out_ptr, h_eye.data(), size * sizeof(Complex));
            }
            return;
        }

        const int actual_steps = (steps > 0) ? steps : 100;
        const double dt = (t_final - t_start) / static_cast<double>(actual_steps);
        const int cheb_degree = (opts.chebyshev_degree > 0) ? opts.chebyshev_degree : 16;

        if (pointers_on_device) {
            // Полный Zero-Copy в VRAM (CuPy / PyTorch / JAX через DLPack без выхода на CPU)
            cuda_optimized::MagnusChebyshevGPU gpu_solver(
                static_cast<int>(dim),
                reinterpret_cast<const cuDoubleComplex*>(d_H0_ptr),
                reinterpret_cast<const cuDoubleComplex*>(d_H1_ptr),
                opts.hbar
            );
            gpu_solver.solve_to_device(
                t_start, t_final, dt, epsilon, omega, cheb_degree,
                reinterpret_cast<cuDoubleComplex*>(d_U_out_ptr)
            );
        }
        else {
            CMatrix h_H0(d_H0_ptr, d_H0_ptr + size);
            CMatrix h_H1(d_H1_ptr, d_H1_ptr + size);

            cuda_optimized::MagnusChebyshevGPU gpu_solver(static_cast<int>(dim), h_H0, h_H1, opts.hbar);
            CMatrix U_res = gpu_solver.solve(t_start, t_final, dt, epsilon, omega, cheb_degree);
            std::memcpy(d_U_out_ptr, U_res.data(), size * sizeof(Complex));
        }
    }

    std::tuple<double, double, double> benchmark_gpu_events_raw(
        int N,
        const Complex* H0_ptr,
        const Complex* H1_ptr,
        double t0,
        double tf,
        int steps,
        double eps,
        double omega,
        int cheb_degree,
        Complex* U_out_ptr
    ) {
        const std::size_t size = static_cast<std::size_t>(N) * N;
        const std::size_t bytes = size * sizeof(cuDoubleComplex);
        const double dt = (tf - t0) / static_cast<double>(steps);

        cudaEvent_t ev0, ev1, ev2, ev3;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        cudaEventCreate(&ev2);
        cudaEventCreate(&ev3);

        cuDoubleComplex* d_H0 = nullptr;
        cuDoubleComplex* d_H1 = nullptr;
        cuDoubleComplex* d_U = nullptr;

        cudaEventRecord(ev0);
        cudaMalloc(&d_H0, bytes);
        cudaMalloc(&d_H1, bytes);
        cudaMalloc(&d_U, bytes);
        cudaMemcpy(d_H0, H0_ptr, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_H1, H1_ptr, bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(ev1);

        {
            cuda_optimized::MagnusChebyshevGPU solver(N, d_H0, d_H1, 1.0);
            solver.solve_to_device(t0, tf, dt, eps, omega, cheb_degree, d_U);
        }
        cudaEventRecord(ev2);

        cudaMemcpy(U_out_ptr, d_U, bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_H0);
        cudaFree(d_H1);
        cudaFree(d_U);
        cudaEventRecord(ev3);
        cudaEventSynchronize(ev3);

        float ms_h2d = 0.0f, ms_comp = 0.0f, ms_d2h = 0.0f;
        cudaEventElapsedTime(&ms_h2d, ev0, ev1);
        cudaEventElapsedTime(&ms_comp, ev1, ev2);
        cudaEventElapsedTime(&ms_d2h, ev2, ev3);

        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        cudaEventDestroy(ev2);
        cudaEventDestroy(ev3);

        return { static_cast<double>(ms_h2d), static_cast<double>(ms_comp), static_cast<double>(ms_d2h) };
    }

    std::pair<std::size_t, std::size_t> get_cuda_mem_info_raw() {
        std::size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        return { free_b, total_b };
    }

    void propagate_cuda_gl3_raw(
        std::size_t dim,
        const Complex* H0_ptr,
        const Complex* H1_ptr,
        const double* time_nodes,
        std::size_t num_time_nodes,
        const double* gl_nodes, // 3 * steps элементов: [f(t_n + c1*h), f(t_n + c2*h), f(t_n + c3*h)]
        const SolverOptions& opts,
        Complex* U_out_ptr
    ) {
        const std::size_t steps = num_time_nodes - 1;
        const std::size_t size = dim * dim;
        const double s15 = std::sqrt(15.0);

        std::vector<cuda_optimized::StepCoeffs> coeffs(steps);
        for (std::size_t n = 0; n < steps; ++n) {
            const double h = time_nodes[n + 1] - time_nodes[n];
            const double fv1 = gl_nodes[3 * n + 0];
            const double fv2 = gl_nodes[3 * n + 1];
            const double fv3 = gl_nodes[3 * n + 2];

            const double fm = fv2;
            const double d1 = (s15 * h / 3.0) * (fv3 - fv1);
            const double d2 = (10.0 * h / 3.0) * (fv3 - 2.0 * fv2 + fv1);

            coeffs[n] = cuda_optimized::StepCoeffs{
                h,
                h * fm + (1.0 / 12.0) * d2,
                -(1.0 / 12.0) * h * d1,
                (1.0 / 360.0) * (h * h) * d2,
                -(1.0 / 240.0) * h * (d1 * d1) + (1.0 / 360.0) * (h * h) * fm * d2,
                (1.0 / 720.0) * (h * h * h) * d1,
                (1.0 / 720.0) * (h * h * h) * fm * d1,
                (1.0 / 720.0) * (h * h * h) * (fm * fm) * d1
            };
        }

        CMatrix h_H0(H0_ptr, H0_ptr + size);
        CMatrix h_H1(H1_ptr, H1_ptr + size);
        const int cheb_degree = (opts.chebyshev_degree > 0) ? opts.chebyshev_degree : 16;

        cuda_optimized::MagnusChebyshevGPU gpu_solver(static_cast<int>(dim), h_H0, h_H1, opts.hbar);
        CMatrix U_res = gpu_solver.solve_coeffs(coeffs, cheb_degree);
        std::memcpy(U_out_ptr, U_res.data(), size * sizeof(Complex));
    }

} // namespace magnus