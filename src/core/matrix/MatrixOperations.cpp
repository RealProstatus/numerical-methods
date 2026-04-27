#include "MatrixOperations.h"

#include <mkl.h>
#include <mkl_lapacke.h>

namespace matrix_ops {
    // 1. Basic matrix operations
    void matmul(const CMatrix& A, const CMatrix& B, CMatrix& C, int N,
        complexd alpha, complexd beta) {
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            N, N, N, &alpha,
            (const void*)A.data(), N,
            (const void*)B.data(), N, &beta,
            (void*)C.data(), N);
    }

    double mat_one_norm(const CMatrix& A, int N) {
        double maxcol = 0.0;
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int i = 0; i < N; ++i) sum += std::abs(A[utils::idx(i, j, N)]);
            if (sum > maxcol) maxcol = sum;
        }
        return maxcol;
    }

    void mat_add(const CMatrix& A, const CMatrix& B, CMatrix& C, int N) {
        size_t n = (size_t)N * N;
        for (size_t k = 0; k < n; ++k) C[k] = A[k] + B[k];
    }

    void mat_sub(const CMatrix& A, const CMatrix& B, CMatrix& C, int N) {
        size_t n = (size_t)N * N;
        for (size_t k = 0; k < n; ++k) C[k] = A[k] - B[k];
    }

    void mat_axpy(const CMatrix& X, CMatrix& Y, int N, complexd alpha) {
        const MKL_INT n = static_cast<MKL_INT>((size_t)N * N);
        cblas_zaxpy(n, &alpha, X.data(), 1, Y.data(), 1);
    }

    void mat_scale_inplace(CMatrix& A, int N, complexd alpha) {
        const MKL_INT n = static_cast<MKL_INT>((size_t)N * N);
        cblas_zscal(n, &alpha, A.data(), 1);
    }

    CMatrix mat_copy(const CMatrix& A) {
        return CMatrix(A);
    }

    CMatrix mat_scale(const CMatrix& A, complexd alpha) {
        CMatrix result = A;
        for (size_t k = 0; k < result.size(); ++k)
            result[k] *= alpha;
        return result;
    }

    // 2. Commutators
    CMatrix commutator(const CMatrix& A, const CMatrix& B, int N) {
        CMatrix AB((size_t)N * N, complexd(0.0, 0.0));
        CMatrix BA((size_t)N * N, complexd(0.0, 0.0));
        CMatrix result((size_t)N * N, complexd(0.0, 0.0));

        matmul(A, B, AB, N);
        matmul(B, A, BA, N);
        mat_sub(AB, BA, result, N);

        return result;
    }

    CMatrix iterated_commutator(const CMatrix& X, const CMatrix& Y, int k, int N) {
        CMatrix result = mat_copy(Y);
        for (int i = 0; i < k; ++i) {
            result = commutator(X, result, N);
        }
        return result;
    }

    // 3. Matrix exponential (implemented in separate modules)

    // 4. Magnus expansion helpers
    double bernoulli_number(int j) {
        switch (j) {
        case 1: return -1.0 / 2.0;
        case 2: return 1.0 / 6.0;
        case 3: return 0.0;
        case 4: return -1.0 / 30.0;
        case 5: return 0.0;
        case 6: return 1.0 / 42.0;
        case 7: return 0.0;
        case 8: return -1.0 / 30.0;
        case 9: return 0.0;
        case 10: return 5.0 / 66.0;
        default: return 0.0;
        }
    }

    CMatrix compute_ad_Omega_k(const CMatrix& Omega, const CMatrix& A, int k, int N) {
        CMatrix result = mat_copy(A);
        for (int i = 0; i < k; ++i) {
            result = commutator(Omega, result, N);
        }
        return result;
    }


    // Updated implementation: cache is provided from the outside (no static cache).
    vector<CMatrix> compute_S_n_j(int n, int j,
        const vector<CMatrix>& Omega,
        const vector<CMatrix>& A_samples,
        int N,
        std::map<std::pair<int, int>, vector<CMatrix>>& cache) // passed by reference
    {
        // We removed static cache inside the function; use the caller-provided cache.

        auto key = std::make_pair(n, j);
        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second; // Return from caller-provided cache
        }

        vector<CMatrix> S_samples(A_samples.size());

        if (j == 1) {
            for (size_t t = 0; t < A_samples.size(); ++t)
                S_samples[t] = commutator(Omega[n - 1], A_samples[t], N);
        }
        else if (j == n - 1) {
            for (size_t t = 0; t < A_samples.size(); ++t)
                S_samples[t] = compute_ad_Omega_k(Omega[1], A_samples[t], n - 1, N);
        }
        else {
            for (size_t t = 0; t < A_samples.size(); ++t) {
                CMatrix sum((size_t)N * N, complexd(0.0, 0.0));
                for (int m = 1; m <= n - j; ++m) {

                    // IMPORTANT: check cache before recursive call
                    auto key_sub = std::make_pair(n - m, j - 1);
                    vector<CMatrix> S_nm_j1_samples;

                    auto it2 = cache.find(key_sub);
                    if (it2 != cache.end()) {
                        S_nm_j1_samples = it2->second;
                    }
                    else {
                        // Pass cache further into recursion.
                        S_nm_j1_samples = compute_S_n_j(n - m, j - 1, Omega, A_samples, N, cache);
                        // cache is updated inside because it is passed by reference
                    }

                    CMatrix comm = commutator(Omega[m], S_nm_j1_samples[t], N);
                    mat_add(sum, comm, sum, N);
                }
                S_samples[t] = sum;
            }
        }

        // Store into caller-provided cache
        cache[key] = S_samples;
        return S_samples;
    }

    CMatrix matrix_exp_special(const CMatrix& A, int N, double dt)
    {
        // Copy A because zheev overwrites the input.
        CMatrix copy = mat_copy(A);

        std::vector<double> w(N);  // Real eigenvalues
        lapack_int info = LAPACKE_zheev(LAPACK_ROW_MAJOR, 'V', 'U', N,
            reinterpret_cast<lapack_complex_double*>(copy.data()), N, w.data());

        if (info != 0) {
            std::cerr << "LAPACKE_zheev failed with info = " << info << std::endl;
            return utils::eye(N);  // Return I on error
        }

        // Create diagonal exp(-i w_k dt)
        CMatrix exp_D((size_t)N * N, complexd(0.0, 0.0));
        for (int k = 0; k < N; ++k) {
            double phase = -w[k] * dt;
            exp_D[utils::idx(k, k, N)] = complexd(std::cos(phase), std::sin(phase));
        }

        // temp = V * exp_D
        CMatrix temp((size_t)N * N, complexd(0.0, 0.0));
        matmul(copy, exp_D, temp, N);

        // result = temp * V^\dagger = V exp_D V^\dagger
        // V^\dagger is the conjugate transpose of V.
        // Since V is row-major, use cblas_zgemm with CblasConjTrans.
        const complexd alpha(1.0, 0.0);
        const complexd beta(0.0, 0.0);

        CMatrix result((size_t)N * N, complexd(0.0, 0.0));
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans,
            N, N, N, &alpha,
            temp.data(), N,
            copy.data(), N, &beta,
            result.data(), N);

        return result;
    }

    CMatrix dagger(const CMatrix& A, int N)
    {
        CMatrix Adag(N * N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
            {
                // transpose + complex conjugation
                Adag[utils::idx(i, j, N)] = std::conj(A[utils::idx(j, i, N)]);
            }
        return Adag;
    }

    bool is_unitary(const CMatrix& U, int N, double tol)
    {
        // 1. U^\dagger
        CMatrix Ud = dagger(U, N);

        // 2. M = U^\dagger * U
        CMatrix M(N * N, complexd(0, 0));
        matmul(Ud, U, M, N);

        // 3. E = M - I
        CMatrix I = utils::eye(N);
        CMatrix E(N * N);
        mat_sub(M, I, E, N);

        // 4. ||E||_1 < tol ?
        double err = mat_one_norm(E, N);

        return (err < tol);
    }

    double max_element_diff(const CMatrix& A, const CMatrix& B, int N) {
        double max_diff = 0.0;
        // Assume CMatrix stores N*N elements
        for (int i = 0; i < N * N; ++i) {
            // std::abs(complex) returns its magnitude
            double current_diff = std::abs(A[i] - B[i]);
            if (current_diff > max_diff) {
                max_diff = current_diff;
            }
        }
        return max_diff;
    }

    double max_eigenvalue_modulus_hermitian(const CMatrix& M, int N) {
        // 1. LAPACKE_zheev overwrites the input matrix, so we copy it
        std::vector<complexd> A_copy = M;

        // 2. Eigenvalues are real for a Hermitian matrix
        std::vector<double> W(N);

        // 3. LAPACKE_zheev: 'N' = eigenvalues only, 'U' = upper triangle stored
        lapack_int info = LAPACKE_zheev(
            LAPACK_ROW_MAJOR,
            'N',              // 'N': compute eigenvalues only
            'U',              // 'U': upper triangle is stored
            N,                // N: size of matrix
            (MKL_Complex16*)A_copy.data(), // A: input matrix
            N,                // lda: leading dimension
            W.data()          // W: output eigenvalues (real)
        );

        if (info != 0) {
            std::cerr << "Warning: LAPACKE_zheev failed with info=" << info
                << ". Returning 0.0." << std::endl;
            return 0.0;
        }

        // 4. Max absolute eigenvalue
        double max_abs_eig = 0.0;
        for (double eig : W) {
            max_abs_eig = std::max(max_abs_eig, std::abs(eig));
        }

        return max_abs_eig;
    }

}
