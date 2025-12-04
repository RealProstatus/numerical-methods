#include <iostream>
#include <complex>
#include <vector>
#include <iomanip>
#include <chrono>
#include "Utils.h"
#include "MatrixOperations.h"

using namespace std;
using namespace utils;
using namespace matrix_ops;
using complexd = complex<double>;
using CMatrix = vector<complexd>;

void print_matrix(const CMatrix& mat, int N, const string& name) {
    cout << name << ":\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            complexd val = mat[i * N + j];
            cout << "(" << setprecision(6) << val.real() << ", " << val.imag() << ") ";
        }
        cout << "\n";
    }
    cout << "\n";
}

int main() {
    const int N = 10;
    double dominance_factor = 10.0;

    CMatrix H = generate_hermitian_matrix(N, dominance_factor);

    // A = -i H 
    CMatrix A = mat_scale(H, complexd(0.0, -1.0));

    // Референс: Taylor 50
    auto ref_start = chrono::high_resolution_clock::now();
    CMatrix ref_exp = expm_taylor(A, N, 50);
    auto ref_end = chrono::high_resolution_clock::now();
    double ref_time = chrono::duration<double>(ref_end - ref_start).count();
    cout << "Reference (Taylor 50 terms) time: " << scientific << setprecision(10) << ref_time << " s\n";
    bool ref_unitary = is_unitary(ref_exp, N, 1e-10);
    cout << "Reference unitary: " << (ref_unitary ? "Yes" : "No") << "\n\n";

    // Метод 1: Taylor 30
    auto taylor_start = chrono::high_resolution_clock::now();
    CMatrix exp_taylor = expm_taylor(A, N, 30);
    auto taylor_end = chrono::high_resolution_clock::now();
    double taylor_time = chrono::duration<double>(taylor_end - taylor_start).count();
    double taylor_diff = max_element_diff(exp_taylor, ref_exp, N);
    bool taylor_unitary = is_unitary(exp_taylor, N, 1e-10);
    cout << "Taylor 30:\n";
    cout << "Time: " << taylor_time << " s\n";
    cout << "Max diff with ref: " << taylor_diff << "\n";
    cout << "Unitary: " << (taylor_unitary ? "Yes" : "No") << "\n\n";

    // Метод 2: Chebyshev (K=30)
    auto cheb_start = chrono::high_resolution_clock::now();
    CMatrix exp_cheb = expm_cheb(A, N, 30);
    auto cheb_end = chrono::high_resolution_clock::now();
    double cheb_time = chrono::duration<double>(cheb_end - cheb_start).count();
    double cheb_diff = max_element_diff(exp_cheb, ref_exp, N);
    bool cheb_unitary = is_unitary(exp_cheb, N, 1e-10);
    cout << "Chebyshev (K=30):\n";
    cout << "Time: " << cheb_time << " s\n";
    cout << "Max diff with ref: " << cheb_diff << "\n";
    cout << "Unitary: " << (cheb_unitary ? "Yes" : "No") << "\n\n";

    // Метод 3: Pade
    auto pade_start = chrono::high_resolution_clock::now();
    CMatrix exp_pade = expm_pade_eigen(A, N);
    auto pade_end = chrono::high_resolution_clock::now();
    double pade_time = chrono::duration<double>(pade_end - pade_start).count();
    double pade_diff = max_element_diff(exp_pade, ref_exp, N);
    bool pade_unitary = is_unitary(exp_pade, N, 1e-10);
    cout << "Pade:\n";
    cout << "Time: " << pade_time << " s\n";
    cout << "Max diff with ref: " << pade_diff << "\n";
    cout << "Unitary: " << (pade_unitary ? "Yes" : "No") << "\n\n";

    //print_matrix(exp_taylor, N, "Taylor Result");
    //print_matrix(exp_cheb, N, "Chebyshev Result");
    //print_matrix(exp_pade, N, "Padé Result");

    cout << "Chebyshev error: " << cheb_diff << "\n";
    cout << "Pade error:      " << pade_diff;

    return 0;
}