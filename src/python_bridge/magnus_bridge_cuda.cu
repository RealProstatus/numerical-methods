#include <iostream>
#include <vector>
#include <complex>
#include <cstring>
#include <stdexcept>
#include <cmath>

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

        // Вырожденный случай: нулевой отрезок времени
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

        int actual_steps = (steps > 0) ? steps : 100;
        double dt = (t_final - t_start) / static_cast<double>(actual_steps);
        int cheb_degree = (opts.chebyshev_degree > 0) ? opts.chebyshev_degree : 15;

        // Инициализация входных матриц для конструктора MagnusChebyshevGPU
        CMatrix h_H0(size);
        CMatrix h_H1(size);

        if (pointers_on_device) {
            cudaError_t err0 = cudaMemcpy(h_H0.data(), d_H0_ptr, size * sizeof(Complex), cudaMemcpyDeviceToHost);
            if (err0 != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA memcpy d_H0 to host failed: ") + cudaGetErrorString(err0));
            }
            cudaError_t err1 = cudaMemcpy(h_H1.data(), d_H1_ptr, size * sizeof(Complex), cudaMemcpyDeviceToHost);
            if (err1 != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA memcpy d_H1 to host failed: ") + cudaGetErrorString(err1));
            }
        }
        else {
            std::memcpy(h_H0.data(), d_H0_ptr, size * sizeof(Complex));
            std::memcpy(h_H1.data(), d_H1_ptr, size * sizeof(Complex));
        }

        // Запуск CUDA-решателя на GPU
        cuda_optimized::MagnusChebyshevGPU gpu_solver(static_cast<int>(dim), h_H0, h_H1);
        CMatrix U_res = gpu_solver.solve(t_start, t_final, dt, epsilon, omega, cheb_degree);

        // Выгрузка полученной унитарной матрицы в целевой буфер
        if (pointers_on_device) {
            cudaError_t errU = cudaMemcpy(d_U_out_ptr, U_res.data(), size * sizeof(Complex), cudaMemcpyHostToDevice);
            if (errU != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA memcpy U to device failed: ") + cudaGetErrorString(errU));
            }
        }
        else {
            std::memcpy(d_U_out_ptr, U_res.data(), size * sizeof(Complex));
        }
    }

} // namespace magnus