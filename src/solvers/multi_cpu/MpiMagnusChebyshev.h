#pragma once

#include <mpi.h>
#include "ProblemSetup.h"
#include "MatrixOperations.h"

namespace multi_cpu {
    CMatrix mpi_magnus_chebyshev_solver(
        const utils::EvolutionProblem& problem, 
        int order, 
        MPI_Comm comm
    );
}