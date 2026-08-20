# numerical-methods

Численный C++ проект для экспериментов с методами Магнуса, матричными экспонентами и построением операторов эволюции для периодического гамильтониана. 

## Статус Реализаций

| Конфигурация | CMake-опция | Статус |
|---|---:|---|
| Один CPU-узел | `BUILD_SINGLE_CPU` | Реализовано и собирается |
| Один GPU-узел | `BUILD_SINGLE_GPU` | Реализовано (CUDA) и собирается |
| Несколько CPU-узлов | `BUILD_MULTI_CPU` | Папка и опция зарезервированы |
| Несколько GPU-узлов | `BUILD_MULTI_GPU` | Папка и опция зарезервированы |

Сейчас рабочий executable target:

```text
single_cpu
single_gpu_app
```

Основные static libraries:

```text
computational_methods
one_cpu_solver
```

## Структура Проекта

```text
numerical-methods/
  CMakeLists.txt
  README.md
  .gitignore

  cmake/
    Dependencies.cmake

  apps/
    single_cpu/
      CMakeLists.txt             # CMake target для executable single_cpu
      main.cpp                   # main с экспериментами (пока что)
    single_gpu/
      CMakeLists.txt             # CMake target для executable single_gpu_app
      gpu_verification.cu        # CUDA-точка входа и бенчмарк

  scripts/                       # Вспомогательные скрипты
  src/
    CMakeLists.txt               # Сборка статических lib с численными методами

    core/
      include/
      matrix/                    # Базовая линейная алгебра, MKL BLAS/LAPACK wrappers, cuBLAS
      utils/                     # Генерация матриц, вывод, сохранение
      magnus/                    # Реализации метода Магнуса
      expm/                      # Методы экспонирования матриц
      integrators/               # Reference-методы

    solvers/
      single_cpu/                # Текущий one-CPU solver
      single_gpu/                # CUDA solver на одном GPU-узле (MagnusChebyshevGPU)
      multi_cpu/                 # Зарезервировано под MPI solver на CPU-кластере
      multi_gpu/                 # Зарезервировано под MPI + CUDA solver

  tests/                         # Зарезервировано под GoogleTest tests

  results/                       # Выходные файлы экспериментов

  3rdparty/
    eigen/                       # Eigen submodule, header-only dependency
    googletest/                  # Будущий GoogleTest submodule, пока может отсутствовать
```

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
|---|---:|---|
| `BUILD_SINGLE_CPU` | `ON` | Собрать текущий рабочий `single_cpu` executable |
| `BUILD_SINGLE_GPU` | `OFF` | Собрать CUDA executable (single_gpu_app) |
| `BUILD_MULTI_CPU` | `OFF` | Зарезервировано под Intel MPI CPU target |
| `BUILD_MULTI_GPU` | `OFF` | Зарезервировано под Intel MPI + CUDA target |
| `BUILD_TESTS` | `OFF` | Включить будущие GoogleTest tests |

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

## Будущие GPU И MPI Targets

Пока эти targets не реализованы.

Папки уже есть:

```text
src/solvers/multi_cpu
src/solvers/multi_gpu
```

Если включить одну из опций:

```bat
cmake -S . -B build\multi_cpu -DBUILD_MULTI_CPU=ON
```

CMake остановится с сообщением, что исходники соответствующей реализации ещё не добавлены. Это ожидаемое поведение