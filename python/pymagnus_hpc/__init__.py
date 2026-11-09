import os
import sys
import glob

# Удержание дескрипторов DLL в памяти для предотвращения их сборки сборщиком мусора
_dll_directories = []

if sys.platform == "win32":
    # 1. Пути к Intel oneMKL и Intel OpenMP
    mkl_candidates = [
        r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\mkl\2025.0\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib",
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\redist\intel64_win\compiler",
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\2025.0\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\2025.0\windows\redist\intel64_win\compiler",
    ]
    for path in mkl_candidates:
        if os.path.isdir(path):
            try:
                _dll_directories.append(os.add_dll_directory(path))
                os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
            except OSError:
                pass

    # 2. Пути к NVIDIA CUDA Toolkit (включая подкаталоги bin\x64 для CUDA 13+)
    cuda_candidates = []
    cuda_path_env = os.environ.get("CUDA_PATH")
    if cuda_path_env:
        cuda_candidates.append(os.path.join(cuda_path_env, "bin"))
        cuda_candidates.append(os.path.join(cuda_path_env, "bin", "x64"))

    cuda_candidates.extend(glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\bin"))
    cuda_candidates.extend(glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\bin\x64"))

    for path in cuda_candidates:
        if os.path.isdir(path):
            try:
                _dll_directories.append(os.add_dll_directory(path))
                os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
            except OSError:
                pass

# Импорт скомпилированного расширения
from ._magnus_py import (
    BackendType,
    ExpmMethod,
    MagnusMethod,
    SolverOptions,
    expm_cpu,
    has_cuda_backend,
    has_mpi_backend,
    propagate_cpu_zero_copy,
    propagate_cuda_dlpack,
)
from .solver import expm_magnus, solve_magnus


def register_qutip_plugin() -> bool:
    try:
        from . import qutip_plugin  # noqa: F401
        return True
    except ImportError:
        return False


register_qutip_plugin()

__all__ = [
    "BackendType",
    "ExpmMethod",
    "MagnusMethod",
    "SolverOptions",
    "expm_cpu",
    "expm_magnus",
    "has_cuda_backend",
    "has_mpi_backend",
    "propagate_cpu_zero_copy",
    "propagate_cuda_dlpack",
    "register_qutip_plugin",
    "solve_magnus",
]