#include <windows.h>
#include <omp.h>
#include <mkl.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <mpi.h>
#include "MpiMagnusChebyshev.h"
#include "PiecewiseMagnus.h"
#include "RungeKutt.h"
#include "AllComputationalMethods.h"
#include "Utils.h"

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Привязка процессов к ядрам
    HANDLE process = GetCurrentProcess();
    
    // 1. Запоминаем изначальные маски ОС, чтобы потом вернуть всё как было, чтобы просчитывать РК4 и one-CPU магнуса на 6 потоках
    DWORD_PTR processAffinityMask, systemAffinityMask;
    GetProcessAffinityMask(process, &processAffinityMask, &systemAffinityMask);
    DWORD_PTR originalMask = processAffinityMask;

    // 2. Узнаем реальное количество логических процессоров в системе
    // для i5 11400f - 6 шутк без гипертрединга
    int max_threads = 6;

    // 3. Привязываем текущий MPI-процесс к своему физическому ядру
    DWORD_PTR mpiAffinityMask = (static_cast<DWORD_PTR>(1) << (rank * 2));
    if (!SetProcessAffinityMask(process, mpiAffinityMask)) {
        if (rank == 0) cerr << "Warning: Failed to set process affinity mask.\n";
    }

    int N = 2048;
    utils::PeriodicHamiltonian ham;
    ham.dimension = N;
    ham.H0.resize(N * N);
    ham.H1.resize(N * N);

    // Генерация матриц гамильтониана
    if (rank == 0) {
        auto [H0_gen, H1_gen] = utils::generate_hermitian_pair(N, 1.0 , 1.0);
        ham.H0 = H0_gen;
        ham.H1 = H1_gen;
    }

    MPI_Bcast(ham.H0.data(), N * N, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    MPI_Bcast(ham.H1.data(), N * N, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    ham.epsilon = 1.0;
    ham.omega = 2.0;
    ham.hbar = 1.0;

    int total_steps = 24;
    auto problem = utils::prepare_evolution_problem(ham, 0.0, 1.0, total_steps);
    int order = 6;

    if (rank == 0) {
        cout << "========================================\n";
        cout << "Starting MPI Scalability Verification...\n";
        cout << "Matrix size (N): " << N << "x" << N << "\n";
        cout << "Integration steps: " << total_steps << "\n";
        cout << "MPI processes: " << size << " (Pinned to physical cores)\n";
        cout << "Max available threads for Single CPU: " << max_threads << "\n";
        cout << "========================================\n";
    }

    // 1. MPI Solver (1 процесс на 1 ядро, 1 поток для 1 процесса для чистоты сравнения)
    omp_set_num_threads(1);
    mkl_set_num_threads(1);

    MPI_Barrier(MPI_COMM_WORLD); 
    double start_time_mpi = MPI_Wtime();
    CMatrix U_mpi = multi_cpu::mpi_magnus_chebyshev_solver(problem, order, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time_mpi = MPI_Wtime();
    double mpi_duration = end_time_mpi - start_time_mpi;

    MPI_Finalize();
    if (rank != 0) {
        return 0;
    }

    cout << "MPI Solver (1 thread/proc) finished in " << mpi_duration << " seconds.\n";

    // return access to all cores
    if (!SetProcessAffinityMask(process, originalMask)) {
        cerr << "Warning: Failed to restore process affinity mask.\n";
    }

    omp_set_num_threads(max_threads);
    mkl_set_num_threads(max_threads);

    // 2. Single CPU Piecewise Magnus
    double start_time_single = omp_get_wtime();
    CMatrix U_single = piecewise_magnus_solver(
        problem.t_start, problem.t_final, problem.step_size(), N,
        ham.H0, ham.H1, ham.epsilon, ham.omega, order
    );
    double end_time_single = omp_get_wtime();
    cout << "Single CPU Solver (" << max_threads << " threads) finished in " << (end_time_single - start_time_single) << " seconds.\n";

    // 3. RK4 Solver
    double start_time_rk4 = omp_get_wtime();
    CMatrix U_rk4 = utils::eye(N);

    int rk4_multiplier = 100; 
    double dt_rk4 = problem.step_size() / rk4_multiplier;
    int total_rk4_steps = total_steps * rk4_multiplier;

    cout << "Starting RK4 with " << total_rk4_steps << " micro-steps...\n";
    
    for(int i = 0; i < total_rk4_steps; ++i) {
        double current_t = problem.t_start + i * dt_rk4;
        U_rk4 = runge_kutta_simple::runge_kutta_step(
            current_t,
            ham.H0, ham.H1, ham.omega,
            U_rk4, dt_rk4, N
        );
    }
    double end_time_rk4 = omp_get_wtime();
    cout << "RK4 Solver (" << max_threads << " threads) finished in " << (end_time_rk4 - start_time_rk4) << " seconds.\n";


    // --- Сравнение результатов ---
    double err_mpi_single = matrix_ops::max_element_diff(U_mpi, U_single, N);
    double err_mpi_rk4    = matrix_ops::max_element_diff(U_mpi, U_rk4, N);
    double err_single_rk4 = matrix_ops::max_element_diff(U_single, U_rk4, N);
    cout << fixed << setprecision(16);
    cout << "========================================\n";
    cout << "Diff(MPI vs Single): " << err_mpi_single << endl;
    cout << "Diff(MPI vs RK4):    " << err_mpi_rk4 << endl;
    cout << "Diff(Single vs RK4): " << err_single_rk4 << endl;
    cout << "========================================\n";
    // Запись результатов в файл
    string report_path = utils::results_path("mpi_hpc_verification_report.txt");
    ofstream out(report_path);
    out << "MPI HPC Verification Report\n" << "======================\n";
    out << "Matrix size: " << N << "x" << N << "\n";
    out << "Processes: " << size << " (Pinned)\nSteps: " << total_steps << "\n\n";
    
    out << "--- Timings ---\n";
    out << "MPI Time (Pure, " << size << " procs): " << (end_time_mpi - start_time_mpi) << " s\n";
    out << "Single Time (MKL " << max_threads << " threads): " << (end_time_single - start_time_single) << " s\n";
    out << "RK4 Time (MKL " << max_threads << " threads):    " << (end_time_rk4 - start_time_rk4) << " s\n\n";
    
    out << "--- Accuracy ---\n";
    out << "Max Diff MPI-Single: " << err_mpi_single << "\n";
    out << "Max Diff MPI-RK4:    " << err_mpi_rk4 << "\n";
    out.close();
    bool is_unitary_mpi = matrix_ops::is_unitary(U_mpi, N);
    bool is_unitary_rk4 = matrix_ops::is_unitary(U_rk4, N);
    cout << "Is MPI solution unitary? " << (is_unitary_mpi ? "YES" : "NO") << endl;
    cout << "Is RK4 solution unitary? " << (is_unitary_rk4 ? "YES" : "NO") << endl; 

    return 0;
}