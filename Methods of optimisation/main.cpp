#include <iostream>
#include <complex>
#include <vector>
#include <iomanip>
#include <cmath> // Для ceil, log2
#include "Utils.h"
#include "MatrixOperations.h"

using namespace std;
using namespace utils;
using namespace matrix_ops;
using complexd = complex<double>;
using CMatrix = vector<complexd>;

// Функция для печати матрицы
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
    const int N = 3; // Малый размер
    double dominance_factor = 0.01; // Меньше, норма ~0.5

    // Генерация Hermitian H (как в проекте)
    CMatrix H = generate_hermitian_matrix(N, dominance_factor);
    print_matrix(H, N, "Hermitian H");

    // A = -i H (anti-Hermitian, как в quantum: eigenvalues pure imaginary)
    CMatrix A = mat_scale(H, complexd(0.0, -1.0));
    print_matrix(A, N, "A = -i H (anti-Hermitian)");

    // Масштабируем A на 0.1 для малой нормы (тест scaling)
    mat_scale_inplace(A, N, complexd(0.1, 0.0));
    double norm_A = mat_one_norm(A, N);
    cout << "Norm of A: " << norm_A << "\n";

    // Вычисление exp(A) с помощью Padé [3/3] (обновите theta в коде на 1.4957e-2)
    CMatrix exp_pade_result = expm_pade_eigen(A, N);
    print_matrix(exp_pade_result, N, "exp(A) via Padé [3/3]");

    // Вычисление exp(A) с помощью Taylor (референс)
    CMatrix exp_taylor_result = expm_taylor(A, N, 30);
    print_matrix(exp_taylor_result, N, "exp(A) via Taylor (30 terms)");

    // Сравнение: max элементная разница
    double max_diff = max_element_diff(exp_pade_result, exp_taylor_result, N);
    cout << "Max element difference between Padé and Taylor: " << scientific << setprecision(10) << max_diff << "\n";

    // Проверка унитарности (теперь должно быть Yes)
    bool is_unit = is_unitary(exp_pade_result, N, 1e-10);
    cout << "Is Padé result unitary? " << (is_unit ? "Yes" : "No") << "\n";

    return 0;
}