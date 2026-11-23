#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include "section1_kernels.hpp"
#include "section2_kernels.hpp"
#include <nanobind/stl/string.h>
#include "magnus/solvers.hpp"

namespace nb = nanobind;
using namespace nb::literals;

using CpuMatrix2D = nb::ndarray<magnus::Complex, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using CpuConstMatrix2D = nb::ndarray<const magnus::Complex, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using CpuConstVec1D = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using GpuMatrix2D = nb::ndarray<magnus::Complex, nb::ndim<2>, nb::c_contig, nb::device::cuda>;
using GpuConstMatrix2D = nb::ndarray<const magnus::Complex, nb::ndim<2>, nb::c_contig, nb::device::cuda>;

NB_MODULE(_magnus_py, m) {
    m.doc() = "C++ / Intel MKL / CUDA / MPI Magnus-Chebyshev Extension";

    nb::enum_<magnus::MagnusMethod>(m, "MagnusMethod")
        .value("Classic2", magnus::MagnusMethod::Classic2)
        .value("Classic4", magnus::MagnusMethod::Classic4)
        .value("Acc4", magnus::MagnusMethod::Acc4)
        .value("Acc6", magnus::MagnusMethod::Acc6)
        .value("Formula314_4", magnus::MagnusMethod::Formula314_4)
        .value("Formula314_6", magnus::MagnusMethod::Formula314_6)
        .value("Recursive", magnus::MagnusMethod::Recursive);

    nb::enum_<magnus::ExpmMethod>(m, "ExpmMethod")
        .value("Chebyshev", magnus::ExpmMethod::Chebyshev)
        .value("PadeEigen", magnus::ExpmMethod::PadeEigen)
        .value("Taylor", magnus::ExpmMethod::Taylor);

    nb::enum_<magnus::BackendType>(m, "BackendType")
        .value("Auto", magnus::BackendType::Auto)
        .value("Cpu", magnus::BackendType::Cpu)
        .value("Cuda", magnus::BackendType::Cuda)
        .value("MpiCpu", magnus::BackendType::MpiCpu);

    nb::class_<magnus::SolverOptions>(m, "SolverOptions")
        .def(nb::init<>())
        .def_rw("magnus_method", &magnus::SolverOptions::magnus_method)
        .def_rw("expm_method", &magnus::SolverOptions::expm_method)
        .def_rw("backend", &magnus::SolverOptions::backend)
        .def_rw("chebyshev_degree", &magnus::SolverOptions::chebyshev_degree)
        .def_rw("taylor_terms", &magnus::SolverOptions::taylor_terms)
        .def_rw("recursive_order", &magnus::SolverOptions::recursive_order)
        .def_rw("hbar", &magnus::SolverOptions::hbar)
        .def_rw("mpi_comm_fortran", &magnus::SolverOptions::mpi_comm_fortran);

    m.def("has_cuda_backend", &magnus::has_cuda_backend);
    m.def("has_mpi_backend", &magnus::has_mpi_backend);

    m.def("expm_cpu", [](
        CpuConstMatrix2D A,
        CpuMatrix2D out,
        magnus::ExpmMethod method,
        int degree_or_terms
        ) {
            std::size_t N = A.shape(0);
            if (A.shape(1) != N || out.shape(0) != N || out.shape(1) != N) {
                throw nb::value_error("Input and output matrices must have shape (N, N).");
            }
            nb::gil_scoped_release release;
            magnus::compute_expm_cpu(N, A.data(), out.data(), method, degree_or_terms);
        }, "A"_a, "out"_a, "method"_a = magnus::ExpmMethod::Chebyshev, "degree_or_terms"_a = 16);

    m.def("propagate_cpu_zero_copy", [](
        CpuConstMatrix2D H0,
        CpuConstMatrix2D H1,
        CpuConstVec1D time_nodes,
        CpuConstVec1D modulation_grid,
        bool is_harmonic,
        double epsilon,
        double omega,
        const magnus::SolverOptions& opts,
        CpuMatrix2D U_out
        ) {
            std::size_t N = H0.shape(0);
            if (H0.shape(1) != N || H1.shape(0) != N || H1.shape(1) != N ||
                U_out.shape(0) != N || U_out.shape(1) != N) {
                throw nb::value_error("Matrices must have square shape (N, N).");
            }

            nb::gil_scoped_release release;
            bool is_uniform = true;
            const std::size_t n_nodes = time_nodes.shape(0);
            if (n_nodes > 2) {
                const double* tn = time_nodes.data();
                const double h0 = tn[1] - tn[0];
                for (std::size_t s = 1; s + 1 < n_nodes; ++s) {
                    if (std::abs((tn[s + 1] - tn[s]) - h0) > 1e-12 * std::max(1.0, std::abs(h0))) {
                        is_uniform = false;
                        break;
                    }
                }
            }

            const bool use_cuda_harmonic =
                is_harmonic && is_uniform &&
                (opts.backend == magnus::BackendType::Cuda ||
                    (opts.backend == magnus::BackendType::Auto &&
                        N >= 128 &&
                        magnus::has_cuda_backend() &&
                        opts.magnus_method == magnus::MagnusMethod::Formula314_6 &&
                        opts.expm_method == magnus::ExpmMethod::Chebyshev));

            if (use_cuda_harmonic) {
                magnus::propagate_cuda_raw(
                    N, H0.data(), H1.data(),
                    time_nodes.data()[0], time_nodes.data()[n_nodes - 1],
                    static_cast<int>(n_nodes - 1),
                    epsilon, omega, opts, U_out.data(), false
                );
            }
            else {
                magnus::propagate_cpu_raw(
                    N, H0.data(), H1.data(),
                    time_nodes.data(), n_nodes,
                    modulation_grid.data(), modulation_grid.shape(0),
                    is_harmonic, epsilon, omega, opts, U_out.data()
                );
            }
        });

    m.def("propagate_cuda_dlpack", [](
        GpuConstMatrix2D d_H0,
        GpuConstMatrix2D d_H1,
        double t_start,
        double t_final,
        int steps,
        double epsilon,
        double omega,
        const magnus::SolverOptions& opts,
        GpuMatrix2D d_U_out
        ) {
            std::size_t N = d_H0.shape(0);
            if (d_H0.shape(1) != N || d_H1.shape(0) != N || d_H1.shape(1) != N ||
                d_U_out.shape(0) != N || d_U_out.shape(1) != N) {
                throw nb::value_error("GPU matrices must have square shape (N, N).");
            }
            nb::gil_scoped_release release;
            magnus::propagate_cuda_raw(
                N, d_H0.data(), d_H1.data(),
                t_start, t_final, steps, epsilon, omega, opts, d_U_out.data(), true
            );
        });

    // === Вспомогательные ядра для экспериментов Раздела 1 (EXP-01 .. EXP-10) ===
    m.def("compute_omega_step", [](
        const std::string& method, int order,
        CpuConstMatrix2D H0, CpuConstMatrix2D H1,
        double t0, double dt, double eps, double omega, int sub_steps,
        CpuMatrix2D out
        ) -> double {
            const int N = static_cast<int>(H0.shape(0));
            const std::size_t elems = static_cast<std::size_t>(N) * N;
            magnus_sec1::CMatrix h0(H0.data(), H0.data() + elems);
            magnus_sec1::CMatrix h1(H1.data(), H1.data() + elems);

            magnus_sec1::CMatrix Om;
            double ms = 0.0;
            {
                nb::gil_scoped_release release;
                auto res = magnus_sec1::compute_omega_step(method, order, N, h0, h1, t0, dt, eps, omega, sub_steps);
                Om = std::move(res.first);
                ms = res.second;
            }
            std::memcpy(out.data(), Om.data(), elems * sizeof(magnus::Complex));
            return ms;
        });

    m.def("propagate_rk4_mkl", [](
        CpuConstMatrix2D H0, CpuConstMatrix2D H1,
        double t0, double tf, int steps, double eps, double omega,
        CpuMatrix2D out
        ) -> double {
            const int N = static_cast<int>(H0.shape(0));
            const std::size_t elems = static_cast<std::size_t>(N) * N;
            magnus_sec1::CMatrix h0(H0.data(), H0.data() + elems);
            magnus_sec1::CMatrix h1(H1.data(), H1.data() + elems);

            magnus_sec1::CMatrix U;
            double ms = 0.0;
            {
                nb::gil_scoped_release release;
                auto res = magnus_sec1::propagate_rk4_mkl(N, h0, h1, t0, tf, steps, eps, omega);
                U = std::move(res.first);
                ms = res.second;
            }
            std::memcpy(out.data(), U.data(), elems * sizeof(magnus::Complex));
            return ms;
        });

    m.def("propagate_piecewise_acc_inner", [](
        CpuConstMatrix2D H0, CpuConstMatrix2D H1,
        double t0, double tf, int macro_steps, int inner_steps,
        double eps, double omega, int order, int cheb_degree,
        CpuMatrix2D out
        ) -> double {
            const int N = static_cast<int>(H0.shape(0));
            const std::size_t elems = static_cast<std::size_t>(N) * N;
            magnus_sec1::CMatrix h0(H0.data(), H0.data() + elems);
            magnus_sec1::CMatrix h1(H1.data(), H1.data() + elems);

            magnus_sec1::CMatrix U;
            double ms = 0.0;
            {
                nb::gil_scoped_release release;
                auto res = magnus_sec1::propagate_piecewise_acc_inner(
                    N, h0, h1, t0, tf, macro_steps, inner_steps, eps, omega, order, cheb_degree
                );
                U = std::move(res.first);
                ms = res.second;
            }
            std::memcpy(out.data(), U.data(), elems * sizeof(magnus::Complex));
            return ms;
        });

    // === Ядра Раздела 2 (EXP-1.1 .. EXP-5.1b) ===
    m.def("get_ptr_address", [](CpuConstMatrix2D arr) -> std::uintptr_t {
        return reinterpret_cast<std::uintptr_t>(arr.data());
        });

    m.def("get_cuda_mem_info", []() {
        auto [free_b, total_b] = magnus_sec2::get_cuda_mem_info();
        return nb::make_tuple(free_b, total_b);
        });

    m.def("generate_mt19937_hermitian_pair", [](int N, std::uint64_t seed, CpuMatrix2D H0_out, CpuMatrix2D H1_out) {
        magnus_sec2::CMatrix h0, h1;
        magnus_sec2::generate_mt19937_hermitian_pair(N, seed, h0, h1);
        const std::size_t bytes = static_cast<std::size_t>(N) * N * sizeof(magnus::Complex);
        std::memcpy(H0_out.data(), h0.data(), bytes);
        std::memcpy(H1_out.data(), h1.data(), bytes);
        });

    m.def("expm_cheb_scaled_stats", [](CpuConstMatrix2D Omega, int M, CpuMatrix2D out) {
        const int N = static_cast<int>(Omega.shape(0));
        const std::size_t elems = static_cast<std::size_t>(N) * N;
        magnus_sec2::CMatrix om(Omega.data(), Omega.data() + elems);
        magnus_sec2::CMatrix res;
        double ms = 0.0;
        int zgemm_cnt = 0, s_cnt = 0;
        {
            nb::gil_scoped_release release;
            std::tie(ms, zgemm_cnt, s_cnt) = magnus_sec2::expm_cheb_scaled_stats(om, N, M, res);
        }
        std::memcpy(out.data(), res.data(), elems * sizeof(magnus::Complex));
        return nb::make_tuple(ms, zgemm_cnt, s_cnt);
        });

    m.def("expm_cheb_no_scaling_stats", [](CpuConstMatrix2D Omega, CpuMatrix2D out) {
        const int N = static_cast<int>(Omega.shape(0));
        const std::size_t elems = static_cast<std::size_t>(N) * N;
        magnus_sec2::CMatrix om(Omega.data(), Omega.data() + elems);
        magnus_sec2::CMatrix res;
        double ms = 0.0;
        int zgemm_cnt = 0, deg_m = 0;
        {
            nb::gil_scoped_release release;
            std::tie(ms, zgemm_cnt, deg_m) = magnus_sec2::expm_cheb_no_scaling_stats(om, N, res);
        }
        std::memcpy(out.data(), res.data(), elems * sizeof(magnus::Complex));
        return nb::make_tuple(ms, zgemm_cnt, deg_m);
        });

    m.def("propagate_method_cpu", [](
        const std::string& method_name, const std::string& expm_name,
        CpuConstMatrix2D H0, CpuConstMatrix2D H1,
        double t0, double tf, int steps, double eps, double omega, int cheb_degree,
        CpuMatrix2D out
        ) -> double {
            const int N = static_cast<int>(H0.shape(0));
            const std::size_t elems = static_cast<std::size_t>(N) * N;
            magnus_sec2::CMatrix h0(H0.data(), H0.data() + elems);
            magnus_sec2::CMatrix h1(H1.data(), H1.data() + elems);
            magnus_sec2::CMatrix res;
            double ms = 0.0;
            {
                nb::gil_scoped_release release;
                ms = magnus_sec2::propagate_method_cpu(method_name, expm_name, N, h0, h1, t0, tf, steps, eps, omega, cheb_degree, res);
            }
            std::memcpy(out.data(), res.data(), elems * sizeof(magnus::Complex));
            return ms;
        });

    m.def("run_long_time_invariants", [](
        const std::string& solver_name,
        CpuConstMatrix2D H0, CpuConstMatrix2D H1,
        double dt, int total_steps, int checkpoint_every,
        double eps, double omega, int cheb_degree,
        nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu> log_out
        ) {
            const int N = static_cast<int>(H0.shape(0));
            const std::size_t elems = static_cast<std::size_t>(N) * N;
            magnus_sec2::CMatrix h0(H0.data(), H0.data() + elems);
            magnus_sec2::CMatrix h1(H1.data(), H1.data() + elems);
            std::vector<double> rows;
            {
                nb::gil_scoped_release release;
                rows = magnus_sec2::run_long_time_invariants(
                    solver_name, N, h0, h1, dt, total_steps, checkpoint_every, eps, omega, cheb_degree
                );
            }
            const std::size_t copy_elems = std::min(rows.size(), log_out.shape(0) * log_out.shape(1));
            std::memcpy(log_out.data(), rows.data(), copy_elems * sizeof(double));
        });

    m.def("benchmark_gpu_events", [](
        CpuConstMatrix2D H0, CpuConstMatrix2D H1,
        double t0, double tf, int steps, double eps, double omega, int cheb_degree,
        CpuMatrix2D out
        ) {
            const int N = static_cast<int>(H0.shape(0));
            const std::size_t elems = static_cast<std::size_t>(N) * N;
            magnus_sec2::CMatrix h0(H0.data(), H0.data() + elems);
            magnus_sec2::CMatrix h1(H1.data(), H1.data() + elems);
            magnus_sec2::CMatrix res;
            double ms_h2d = 0.0, ms_comp = 0.0, ms_d2h = 0.0;
            {
                nb::gil_scoped_release release;
                std::tie(ms_h2d, ms_comp, ms_d2h) = magnus_sec2::benchmark_gpu_events(
                    N, h0, h1, t0, tf, steps, eps, omega, cheb_degree, res
                );
            }
            std::memcpy(out.data(), res.data(), elems * sizeof(magnus::Complex));
            return nb::make_tuple(ms_h2d, ms_comp, ms_d2h);
        });

    m.def("propagate_cuda_gl3", [](
        CpuConstMatrix2D H0,
        CpuConstMatrix2D H1,
        CpuConstVec1D time_nodes,
        CpuConstVec1D gl_nodes,
        const magnus::SolverOptions& opts,
        CpuMatrix2D U_out
        ) {
            const std::size_t dim = H0.shape(0);
            if (H0.shape(1) != dim || H1.shape(0) != dim || H1.shape(1) != dim ||
                U_out.shape(0) != dim || U_out.shape(1) != dim) {
                throw std::invalid_argument("H0, H1, and U_out must be square (N, N) matrices.");
            }
            if (time_nodes.shape(0) < 2) {
                throw std::invalid_argument("time_nodes must have at least 2 elements.");
            }
            const std::size_t steps = time_nodes.shape(0) - 1;
            if (gl_nodes.shape(0) != 3 * steps) {
                throw std::invalid_argument("gl_nodes must have length 3 * steps.");
            }

            const auto* h0_ptr = reinterpret_cast<const magnus::Complex*>(H0.data());
            const auto* h1_ptr = reinterpret_cast<const magnus::Complex*>(H1.data());
            auto* out_ptr = reinterpret_cast<magnus::Complex*>(U_out.data());

            nb::gil_scoped_release release;
            magnus::propagate_cuda_gl3_raw(
                dim, h0_ptr, h1_ptr,
                time_nodes.data(), time_nodes.shape(0),
                gl_nodes.data(), opts, out_ptr
            );
        });

    m.def("propagate_general_gl_cpu", [](
        CpuConstVec1D time_nodes,
        nb::ndarray<const magnus::Complex, nb::ndim<4>, nb::c_contig, nb::device::cpu> A_gl_nodes,
        const magnus::SolverOptions& opts,
        CpuMatrix2D U_out
        ) {
            if (time_nodes.shape(0) < 2) {
                throw std::invalid_argument("time_nodes must have at least 2 elements.");
            }
            const std::size_t steps = time_nodes.shape(0) - 1;
            const std::size_t dim = U_out.shape(0);
            if (A_gl_nodes.shape(0) != steps || A_gl_nodes.shape(1) != 3 ||
                A_gl_nodes.shape(2) != dim || A_gl_nodes.shape(3) != dim || U_out.shape(1) != dim) {
                throw std::invalid_argument("A_gl_nodes must have shape (steps, 3, N, N) matching U_out (N, N).");
            }

            const auto* a_ptr = reinterpret_cast<const magnus::Complex*>(A_gl_nodes.data());
            auto* out_ptr = reinterpret_cast<magnus::Complex*>(U_out.data());

            nb::gil_scoped_release release;
            magnus::propagate_general_gl_mkl(
                dim, time_nodes.data(), time_nodes.shape(0),
                a_ptr, opts, out_ptr
            );
        });
}