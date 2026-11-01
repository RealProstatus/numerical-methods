#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
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
            if ((opts.backend == magnus::BackendType::Cuda ||
                (opts.backend == magnus::BackendType::Auto && N >= 128 && magnus::has_cuda_backend())) &&
                is_harmonic) {
                magnus::propagate_cuda_raw(
                    N, H0.data(), H1.data(),
                    time_nodes.data()[0], time_nodes.data()[time_nodes.shape(0) - 1],
                    static_cast<int>(time_nodes.shape(0) - 1),
                    epsilon, omega, opts, U_out.data(), false
                );
            }
            else {
                magnus::propagate_cpu_raw(
                    N, H0.data(), H1.data(),
                    time_nodes.data(), time_nodes.shape(0),
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
}