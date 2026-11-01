#pragma once
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace magnus {

    using Complex = std::complex<double>;

    enum class MagnusMethod {
        Classic2,
        Classic4,
        Acc4,
        Acc6,
        Formula314_4,
        Formula314_6,
        Recursive
    };

    enum class ExpmMethod {
        Chebyshev,
        PadeEigen,
        Taylor
    };

    enum class BackendType {
        Auto,
        Cpu,
        Cuda,
        MpiCpu
    };

    struct SolverOptions {
        MagnusMethod magnus_method = MagnusMethod::Formula314_6;
        ExpmMethod expm_method = ExpmMethod::Chebyshev;
        BackendType backend = BackendType::Auto;
        int chebyshev_degree = 16;
        int taylor_terms = 30;
        int recursive_order = 8;
        double hbar = 1.0;
        bool return_trajectory = false;
        int num_threads = 0;
        int mpi_comm_fortran = -1;
    };

    void compute_expm_cpu(
        std::size_t dim,
        const Complex* A_in,
        Complex* expA_out,
        ExpmMethod method,
        int degree_or_terms = 16
    );

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
    );

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
        bool pointers_on_device = true
    );

    bool has_cuda_backend() noexcept;
    bool has_mpi_backend() noexcept;

} // namespace magnus