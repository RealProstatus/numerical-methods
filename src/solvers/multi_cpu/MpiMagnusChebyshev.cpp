#include "MpiMagnusChebyshev.h"
#include "PiecewiseMagnus.h"
#include <iostream>

using namespace matrix_ops;

CMatrix multi_cpu::mpi_magnus_chebyshev_solver(const utils::EvolutionProblem& problem, int order, MPI_Comm comm) 
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int N = problem.hamiltonian.dimension;
    int total_steps = problem.steps;
    
    // Проверка делимости
    if (total_steps % size != 0 && rank == 0) {
        std::cerr << "Warning: total_steps (" << total_steps << ") is not divisible by MPI size (" << size << ")\n";
    }

    int local_steps = total_steps / size;
    double macro_dt = problem.step_size();

    // Границы локального участка
    double local_t_start = problem.node(rank * local_steps);
    double local_t_final = problem.node((rank + 1) * local_steps);

    // 1. Локальный расчет: Magnus ACC + Chebyshev
    CMatrix U_local = piecewise_magnus_solver(
        local_t_start, local_t_final, macro_dt, N,
        problem.hamiltonian.H0, problem.hamiltonian.H1,
        problem.hamiltonian.epsilon, problem.hamiltonian.omega, order
    );

    // 2. Бинарное дерево редукции для перемножения матриц
    CMatrix U_recv(N * N);
    CMatrix U_temp(N * N);

    for (int step = 1; step < size; step *= 2) {
        if (rank % (2 * step) == 0) {
            int sender = rank + step;
            if (sender < size) {
                MPI_Recv(U_recv.data(), N * N, MPI_C_DOUBLE_COMPLEX, sender, 0, comm, MPI_STATUS_IGNORE);
                matmul(U_recv, U_local, U_temp, N);
                U_local = U_temp; 
            }
        } else if (rank % (2 * step) == step) {
            int receiver = rank - step;
            MPI_Send(U_local.data(), N * N, MPI_C_DOUBLE_COMPLEX, receiver, 0, comm);
            break; 
        }
    }

    return U_local; 
}