#include "magnus/solvers.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "core/matrix/MatrixOperations.h"
#include "core/expm/ChebyshevExpm.h"
#include "core/expm/EigenPadeExpm.h"
#include "core/expm/TaylorExpm.h"
#include "core/utils/ProblemSetup.h"
#include "solvers/single_cpu/Formula314ChebyshevSolver.h"
#include "solvers/single_cpu/PiecewiseMagnus.h"

#if defined(MAGNUS_HAS_MPI)
#include <mpi.h>
#include "solvers/multi_cpu/MpiMagnusChebyshev.h"
#endif

#if defined(MAGNUS_HAS_CUDA)
#include <cuda_runtime.h>
#include "solvers/single_gpu/CudaMagnusChebyshev.cuh"
#endif

namespace magnus {

    namespace {

        CMatrix apply_expm(const CMatrix& Omega, int N, const SolverOptions& opts) {
            switch (opts.expm_method) {
            case ExpmMethod::Chebyshev:
                return matrix_ops::expm_cheb(Omega, N, opts.chebyshev_degree);
            case ExpmMethod::PadeEigen:
                return matrix_ops::expm_pade_eigen(Omega, N);
            case ExpmMethod::Taylor:
                return matrix_ops::expm_taylor(Omega, N, opts.taylor_terms);
            default:
                return matrix_ops::expm_cheb(Omega, N, opts.chebyshev_degree);
            }
        }

        void propagate_arbitrary_modulation_mkl(
            int N,
            const CMatrix& H0,
            const CMatrix& H1,
            const double* time_nodes,
            std::size_t num_time_nodes,
            const double* mod_grid,
            std::size_t num_mod_points,
            const SolverOptions& opts,
            CMatrix& U_total
        ) {
            const std::size_t steps = num_time_nodes - 1;
            const std::size_t total_elems = static_cast<std::size_t>(N) * N;

            U_total.assign(total_elems, Complex(0.0, 0.0));
            for (int i = 0; i < N; ++i) {
                U_total[static_cast<std::size_t>(i) * N + i] = Complex(1.0, 0.0);
            }

            CMatrix C1(total_elems, Complex(0.0, 0.0));
            matrix_ops::commutator_inplace(H0, H1, C1, N);

            CMatrix C2, C3;
            const bool use_order6 = (opts.magnus_method == MagnusMethod::Formula314_6 ||
                opts.magnus_method == MagnusMethod::Acc6 ||
                opts.magnus_method == MagnusMethod::Recursive);
            if (use_order6) {
                C2.assign(total_elems, Complex(0.0, 0.0));
                C3.assign(total_elems, Complex(0.0, 0.0));
                matrix_ops::commutator_inplace(H0, C1, C2, N);
                matrix_ops::commutator_inplace(H1, C1, C3, N);
            }

            CMatrix Omega(total_elems, Complex(0.0, 0.0));
            CMatrix U_next(total_elems, Complex(0.0, 0.0));

            for (std::size_t step = 0; step < steps; ++step) {
                const double t0 = time_nodes[step];
                const double t1 = time_nodes[step + 1];
                const double dt = t1 - t0;

                double f0 = 0.0, f_half = 0.0, f1 = 0.0;
                if (num_mod_points >= 2 * steps + 1) {
                    f0 = mod_grid[2 * step];
                    f_half = mod_grid[2 * step + 1];
                    f1 = mod_grid[2 * step + 2];
                }
                else if (num_mod_points >= steps + 1) {
                    f0 = mod_grid[step];
                    f1 = mod_grid[step + 1];
                    f_half = 0.5 * (f0 + f1);
                }

                const double int_f = (dt / 6.0) * (f0 + 4.0 * f_half + f1);
                const double fp_half = (f1 - f0) / dt;
                const double fpp_half = 4.0 * (f0 - 2.0 * f_half + f1) / (dt * dt);

                std::fill(Omega.begin(), Omega.end(), Complex(0.0, 0.0));

                const Complex factor1(0.0, -1.0 / opts.hbar);
                matrix_ops::mat_axpy(H0, Omega, N, factor1 * dt);
                matrix_ops::mat_axpy(H1, Omega, N, factor1 * int_f);

                if (opts.magnus_method != MagnusMethod::Classic2) {
                    const double coeff_c1 = -(dt * dt * dt / 12.0) * fp_half / (opts.hbar * opts.hbar);
                    matrix_ops::mat_axpy(C1, Omega, N, Complex(coeff_c1, 0.0));
                }

                if (use_order6) {
                    const double hbar3 = opts.hbar * opts.hbar * opts.hbar;
                    const double dt5 = dt * dt * dt * dt * dt;
                    const Complex coeff_c2(0.0, (dt5 / 720.0) * fpp_half / hbar3);
                    const Complex coeff_c3(0.0, -(dt5 / 240.0) * f_half * fpp_half / hbar3);
                    matrix_ops::mat_axpy(C2, Omega, N, coeff_c2);
                    matrix_ops::mat_axpy(C3, Omega, N, coeff_c3);
                }

                CMatrix U_step = apply_expm(Omega, N, opts);
                matrix_ops::matmul(U_step, U_total, U_next, N);
                U_total.swap(U_next);
            }
        }

    } // namespace

    bool has_cuda_backend() noexcept {
#if defined(MAGNUS_HAS_CUDA)
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        return (err == cudaSuccess && device_count > 0);
#else
        return false;
#endif
    }

    bool has_mpi_backend() noexcept {
#if defined(MAGNUS_HAS_MPI)
        return true;
#else
        return false;
#endif
    }

    void compute_expm_cpu(
        std::size_t dim,
        const Complex* A_in,
        Complex* expA_out,
        ExpmMethod method,
        int degree_or_terms
    ) {
        const int N = static_cast<int>(dim);
        CMatrix A(A_in, A_in + dim * dim);
        CMatrix res;
        switch (method) {
        case ExpmMethod::Chebyshev:
            res = matrix_ops::expm_cheb(A, N, degree_or_terms);
            break;
        case ExpmMethod::PadeEigen:
            res = matrix_ops::expm_pade_eigen(A, N);
            break;
        case ExpmMethod::Taylor:
            res = matrix_ops::expm_taylor(A, N, degree_or_terms);
            break;
        }
        std::memcpy(expA_out, res.data(), dim * dim * sizeof(Complex));
    }

    void propagate_cpu_raw(
        std::size_t dim,
        const Complex* H0_ptr,
        const Complex* H1_ptr,
        const double* time_nodes,
        std::size_t num_time_nodes,
        const double* modulation_grid,
        std::size_t num_mod_points,
        bool is_harmonic,
        double epsilon,
        double omega,
        const SolverOptions& opts,
        Complex* U_out_ptr
    ) {
        if (num_time_nodes < 2) {
            throw std::invalid_argument("time_nodes must contain at least 2 points (t_start, t_final).");
        }

        const int N = static_cast<int>(dim);
        const std::size_t total_elems = dim * dim;
        const int steps = static_cast<int>(num_time_nodes - 1);
        const double t_start = time_nodes[0];
        const double t_final = time_nodes[num_time_nodes - 1];

        CMatrix H0(H0_ptr, H0_ptr + total_elems);
        CMatrix H1(H1_ptr, H1_ptr + total_elems);
        CMatrix U_res;

#if defined(MAGNUS_HAS_MPI)
        if (opts.backend == BackendType::MpiCpu && is_harmonic) {
            int initialized = 0;
            MPI_Initialized(&initialized);
            if (!initialized) {
                throw std::runtime_error("MPI is not initialized.");
            }
            MPI_Comm comm = (opts.mpi_comm_fortran >= 0)
                ? MPI_Comm_f2c(opts.mpi_comm_fortran)
                : MPI_COMM_WORLD;
            U_res = mpi_magnus_chebyshev_solver(
                H0, H1, N, t_start, t_final, steps, epsilon, omega, opts.chebyshev_degree, comm
            );
            if (!U_res.empty()) {
                std::memcpy(U_out_ptr, U_res.data(), total_elems * sizeof(Complex));
            }
            return;
        }
#endif

        if (is_harmonic) {
            utils::PeriodicHamiltonian ham;
            ham.dimension = N;
            ham.H0 = H0;
            ham.H1 = H1;
            ham.epsilon = epsilon;
            ham.omega = omega;
            ham.hbar = opts.hbar;

            if (opts.magnus_method == MagnusMethod::Formula314_6 &&
                opts.expm_method == ExpmMethod::Chebyshev) {
                single_cpu::Formula314ChebyshevOptions f_opts;
                f_opts.chebyshev_degree = opts.chebyshev_degree;
                U_res = single_cpu::solve_formula314_chebyshev(ham, t_start, t_final, steps, f_opts);
            }
            else {
                int order = 4;
                if (opts.magnus_method == MagnusMethod::Classic2) {
                    order = 2;
                }
                else if (opts.magnus_method == MagnusMethod::Classic4 || opts.magnus_method == MagnusMethod::Acc4) {
                    order = 4;
                }
                else if (opts.magnus_method == MagnusMethod::Acc6 || opts.magnus_method == MagnusMethod::Formula314_6) {
                    order = 6;
                }
                else if (opts.magnus_method == MagnusMethod::Recursive) {
                    order = opts.recursive_order;
                }
                const double macro_dt = (t_final - t_start) / steps;
                U_res = piecewise_magnus_solver(t_start, t_final, macro_dt, N, H0, H1, epsilon, omega, order);
            }
        }
        else {
            propagate_arbitrary_modulation_mkl(
                N, H0, H1, time_nodes, num_time_nodes, modulation_grid, num_mod_points, opts, U_res
            );
        }

        std::memcpy(U_out_ptr, U_res.data(), total_elems * sizeof(Complex));
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
#if defined(MAGNUS_HAS_CUDA)
        const int N = static_cast<int>(dim);
        const std::size_t total_elems = dim * dim;
        const std::size_t bytes = total_elems * sizeof(Complex);

        CMatrix h_H0(total_elems);
        CMatrix h_H1(total_elems);

        if (pointers_on_device) {
            cudaMemcpy(h_H0.data(), d_H0_ptr, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_H1.data(), d_H1_ptr, bytes, cudaMemcpyDeviceToHost);
        }
        else {
            std::memcpy(h_H0.data(), d_H0_ptr, bytes);
            std::memcpy(h_H1.data(), d_H1_ptr, bytes);
        }

        cuda_optimized::MagnusChebyshevGPU gpu_solver(N, h_H0, h_H1);
        CMatrix h_U = gpu_solver.solve(t_start, t_final, steps, epsilon, omega, opts.chebyshev_degree);

        if (pointers_on_device) {
            cudaMemcpy(d_U_out_ptr, h_U.data(), bytes, cudaMemcpyHostToDevice);
        }
        else {
            std::memcpy(d_U_out_ptr, h_U.data(), bytes);
        }
#else
        (void)dim; (void)d_H0_ptr; (void)d_H1_ptr; (void)t_start; (void)t_final;
        (void)steps; (void)epsilon; (void)omega; (void)opts; (void)d_U_out_ptr; (void)pointers_on_device;
        throw std::runtime_error("pymagnus_hpc was compiled without CUDA support.");
#endif
    }

} // namespace magnus