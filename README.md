# numerical-methods

Численный C++ проект для экспериментов с методами Магнуса, матричными экспонентами и построением операторов эволюции для периодического гамильтониана. 

## Статус Реализаций

| Конфигурация / Модуль | Опция сборки | Статус |
|---|:---:|---|
| Один CPU-узел (CLI) | `BUILD_SINGLE_CPU` | Реализовано и собирается[cite: 14] |
| Один GPU-узел (CUDA CLI) | `BUILD_SINGLE_GPU` | Реализовано (CUDA + cuBLAS) и собирается[cite: 14] |
| Несколько CPU-узлов (MPI CLI) | `BUILD_MULTI_CPU` | Реализовано (декомпозиция по времени + дерево редукции)[cite: 2, 14] |
| Несколько GPU-узлов | `BUILD_MULTI_GPU` | Зарезервировано[cite: 14] |
| **Python-модуль и плагины (QuTiP 5 / SciPy / Qiskit)** | `BUILD_PYTHON_BINDINGS` | **Реализовано (`pymagnus-hpc` через `nanobind`)**[cite: 2] |

Основные исполняемые цели (Executables)[cite: 14]:
* `single_cpu`[cite: 14]
* `single_gpu_app`[cite: 14]
* `mpi_verification`[cite: 14]

Основные библиотеки (Libraries)[cite: 14]:
* `computational_methods` (MKL/CUDA ядро матричных операций, экспонент и коммутаторов)[cite: 14]
* `one_cpu_solver` (солверы для одного CPU-узла)[cite: 14]
* `multi_cpu_solver` (MPI-солвер для кластеров)[cite: 14]
* `magnus_bridge` (C++ фасад и адаптер для вызова численного ядра из Python)[cite: 2]
* `_magnus_py` (nanobind расширение Python)[cite: 2]

---

## Структура Проекта

```text
numerical-methods/
  CMakeLists.txt                 # Корневой CMake (сборка CLI и Python-модуля)
  pyproject.toml                 # Сборка wheel-пакета pymagnus-hpc через scikit-build-core
  README.md
  .gitignore

  cmake/
    Dependencies.cmake           # Поиск MKL, Eigen, OpenMP

  include/
    magnus/
      solvers.hpp                # Чистый C++20 фасад (типы, опции, сигнатуры вызовов)

  src/
    CMakeLists.txt               # Сборка статических библиотек ядра и моста

    core/
      include/
      matrix/                    # MKL BLAS/LAPACK обертки (cblas_zgemm, нормы, коммутаторы)
      utils/                     # Генераторы матриц, сетки, ProblemSetup
      magnus/                    # Разложения Магнуса (Classic, Acc, Formula314, Recursive)
      expm/                      # Вычисление exp(A) (Chebyshev, PadeEigen, Taylor)
      integrators/               # Опорные интеграторы (RungeKutt)

    solvers/
      single_cpu/                # PiecewiseMagnus и Formula314ChebyshevSolver
      single_gpu/                # CudaMagnusChebyshev (MagnusChebyshevGPU)
      multi_cpu/                 # MpiMagnusChebyshev (декомпозиция по времени)
      multi_gpu/                 # Зарезервировано

    python_bridge/
      magnus_bridge.cpp          # Адаптер вызовов MKL/CUDA/MPI ядер к фасаду
      magnus_bridge_cuda.cu      # GPU-мост вызова CUDA-солвера

  python/
    bindings/
      module.cpp                 # Биндинги nanobind (_magnus_py, zero-copy, release GIL)
    pymagnus_hpc/
      __init__.py                # Регистрация DLL Intel MKL/OpenMP и плагинов
      solver.py                  # Высокоуровневый API (solve_magnus, expm_magnus)
      qutip_plugin.py            # Интегратор MagnusChebyshevIntegrator для QuTiP 5
      scipy_plugin.py            # Подкласс OdeSolver для scipy.integrate.solve_ivp
      qiskit_plugin.py           # Метод решения для qiskit_dynamics.solve_lmde

  apps/
    single_cpu/                  # CLI эксперименты для одного CPU
    single_gpu/                  # CLI бенчмарк и верификация CUDA
    multi_cpu/                   # CLI запуск MPI-редукции

  scripts/                       # Скрипты построения графиков и запуска серий тестов
  tests/                         # Тестовый набор
  results/                       # Выходные данные экспериментов
  3rdparty/
    eigen/                       # Шаблоны Eigen (submodule)
    googletest/                  # GoogleTest (submodule)
```

---

## Интеграция с Python и QuTiP 5

Пакет `pymagnus-hpc` предоставляет бесшовный доступ к C++/MKL солверам напрямую из Python без промежуточного копирования матриц в памяти (Zero-Copy через buffer protocol и DLPack) и с автоматическим освобождением GIL[cite: 2].

### 1. Установка пакета

В активированном окружении Intel oneAPI[cite: 14]:

```bat
# Установка сборочных зависимостей
pip install nanobind scikit-build-core qutip

# Установка пакета в режиме разработки
pip install -e . --no-build-isolation
```

Для сборки с поддержкой CUDA GPU (RTX 4060):

```bat
pip install -e . --no-build-isolation --config-settings=cmake.args="-DBUILD_SINGLE_GPU=ON"
```

### 2. Использование через интерфейс QuTiP 5 (`sesolve` / `propagator`)

При импорте `pymagnus_hpc` интегратор `"magnus_chebyshev"` автоматически регистрируется в реестре `qutip.SESolver`[cite: 2]:

```python
import numpy as np
import qutip as qt
import pymagnus_hpc  # Регистрация MagnusChebyshevIntegrator в QuTiP 5

# Гамильтониан двухуровневой системы: H(t) = 0.5*omega0*sigma_z + eps*cos(omega*t)*sigma_x
omega0, eps, omega = 2.0 * np.pi, 0.2, 2.0 * np.pi
H0 = 0.5 * omega0 * qt.sigmaz()
H1 = qt.sigmax()

H_evo = [H0, [H1, lambda t: eps * np.cos(omega * t)]]
psi0 = qt.basis(2, 0)
tlist = np.linspace(0.0, 1.0, 51)

# Вызов C++/MKL/CUDA солвера Магнуса 6-го порядка через штатный интерфейс QuTiP
result = qt.sesolve(
    H_evo,
    psi0,
    tlist,
    options={
        "method": "magnus_chebyshev",
        "magnus_method": "Formula314_6",   # 'Classic2', 'Acc4', 'Acc6', 'Formula314_6'
        "expm_method": "Chebyshev",        # 'Chebyshev', 'PadeEigen', 'Taylor'
        "chebyshev_degree": 16,
        "dt": 0.01,
        "is_harmonic": True,
        "epsilon": eps,
        "omega": omega,
    },
)

print("Финальное состояние:", result.states[-1])
```

---

## CMake Targets

### `computational_methods`

Static library со всеми текущими математическими функциями проекта:

- матричные операции;
- коммутаторы;
- методы Магнуса;
- матричные экспоненты;
- reference RK4;
- вспомогательные utilities.

### `one_cpu_solver`

Static library для one-CPU solver layer.

Сейчас туда входит существующий `PiecewiseMagnus`.

Важно: полноценный единый Floquet solver пока не собирается в отдельную цель. Мы пока только переносим текущую реализацию на CMake и фиксируем архитектурные места.

### `single_cpu`

Executable target, который собирает текущий `main.cpp` и позволяет запускать существующие эксперименты.

### `single_gpu_app`

Executable target, который собирает бенчмарк CUDA-реализации и сравнивает её с эталонным CPU-решением.

## Зависимости

Обязательные:

- CMake `>= 3.24`;
- Intel oneAPI C++ compiler;
- Intel oneMKL с `MKLConfig.cmake`;
- Eigen submodule;
- Visual Studio 2022 на Windows.
- CUDA Toolkit (для сборки GPU таргетов).

Будущие:

- Intel MPI;
- GoogleTest.

## Подготовка Submodules

Перед первой сборкой выполните:

```bat
git submodule update --init --recursive
```


## Сборка На Windows Через Visual Studio 2022 + Intel oneAPI

Это основной рекомендуемый сценарий

### 1. Откройте Intel oneAPI окружение

Откройте:

```text
Intel oneAPI command prompt for Intel 64 for Visual Studio 2022
```

Или вручную активируйте окружение в cmd:

```bat
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```

### 2. Сконфигурируйте CMake

Рекомендуемый вариант с Visual Studio generator и Intel toolset:

```bat
cmake -S . -B build\single_cpu -G "Visual Studio 17 2022" -A x64 -T "Intel C++ Compiler 2026" -DBUILD_SINGLE_CPU=ON -DBUILD_SINGLE_GPU=OFF -DBUILD_MULTI_CPU=ON -DBUILD_MULTI_GPU=OFF -DBUILD_TESTS=OFF
```

Если версия Intel toolset в вашей установке называется иначе, посмотрите доступные toolsets в Visual Studio Installer или CMake output и замените:

```text
Intel C++ Compiler 2025
```

на актуальное имя.

Сборка через Ninja для MPI-реализации
```bat
cmake -S . -B build -G "Ninja" -DCMAKE_CXX_COMPILER=icx -DBUILD_SINGLE_CPU=ON -DBUILD_MULTI_CPU=ON
cmake --build build
mpiexec -n 6 build\apps\multi_cpu\mpi_verification.exe
```

### 3. Соберите `single_cpu`

```bat
cmake --build build\single_cpu --config Release --target single_cpu
cmake --build build\single_cpu --config RelWithDebInfo --target single_cpu
```

### 4. Запустите приложение

```bat
build\single_cpu\apps\single_cpu\Release\single_cpu.exe
```

## Сборка GPU-версии на Windows (CUDA)

Важно: Для компиляции CUDA-кода (BUILD_SINGLE_GPU) под Windows необходимо использовать стандартный компилятор MSVC, а не Intel C++ Compiler, чтобы избежать конфликтов системных библиотек.

### Конфигурация (без флага -T):

```bat
cmake -S . -B build\single_gpu -G "Visual Studio 17 2022" -A x64 -DBUILD_SINGLE_GPU=ON -DBUILD_SINGLE_CPU=OFF -DBUILD_MULTI_CPU=OFF -DBUILD_MULTI_GPU=OFF -DBUILD_TESTS=OFF
```

### Сборка:

```bat
cmake --build build\single_gpu --config Release
```

### Запуск

```bat
build\single_gpu\apps\single_gpu\Release\single_gpu_app.exe
```

## Альтернативная Ручная Сборка На Windows Через NMake


Окружение всё равно должно быть активировано:

```bat
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```

Конфигурация:

```bat
cmake -S . -B build\single_cpu_nmake ^
  -G "NMake Makefiles" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CXX_COMPILER=icx ^
  -DBUILD_SINGLE_CPU=ON ^
  -DBUILD_SINGLE_GPU=OFF ^
  -DBUILD_MULTI_CPU=OFF ^
  -DBUILD_MULTI_GPU=OFF ^
  -DBUILD_TESTS=OFF
```

Сборка:

```bat
cmake --build build\single_cpu_nmake --target single_cpu
```

Запуск:

```bat
build\single_cpu_nmake\apps\single_cpu\single_cpu.exe
```

## Сборка На Linux

Linux-сценарий предусмотрен, но основной текущий рабочий сценарий — Windows + VS 2022.

Активируйте oneAPI:

```bash
source /opt/intel/oneapi/setvars.sh
```

Конфигурация:

```bash
cmake -S . -B build/single_cpu_linux \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=icpx \
  -DBUILD_SINGLE_CPU=ON \
  -DBUILD_SINGLE_GPU=OFF \
  -DBUILD_MULTI_CPU=OFF \
  -DBUILD_MULTI_GPU=OFF \
  -DBUILD_TESTS=OFF
```

Сборка:

```bash
cmake --build build/single_cpu_linux --target single_cpu
```

Запуск:

```bash
./build/single_cpu_linux/apps/single_cpu/single_cpu
```

## CMake Опции

| Опция | По умолчанию | Назначение |
|---|:---:|---|
| `BUILD_SINGLE_CPU` | `ON` | Сборка одноузлового CPU исполняемого файла (`single_cpu`)[cite: 14] |
| `BUILD_SINGLE_GPU` | `OFF` | Сборка CUDA исполняемого файла (`single_gpu_app`)[cite: 14] |
| `BUILD_MULTI_CPU` | `OFF` | Сборка многоузлового MPI приложения (`mpi_verification`)[cite: 14] |
| `BUILD_MULTI_GPU` | `OFF` | Резерв под распределенную GPU-систему (MPI + CUDA)[cite: 14] |
| `BUILD_PYTHON_BINDINGS` | `ON` | Сборка C++ расширения `_magnus_py` для Python[cite: 2] |
| `BUILD_TESTS` | `OFF` | Сборка модульных тестов GoogleTest[cite: 14] |

---

## Управление Потоками MKL

Число потоков на CPU удобно задавать через переменные окружения.

Windows:

```bat
set MKL_NUM_THREADS=1
set OMP_NUM_THREADS=1
build\single_cpu\apps\single_cpu\Release\single_cpu.exe
```

Linux:

```bash
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
./build/single_cpu_linux/apps/single_cpu/single_cpu
```