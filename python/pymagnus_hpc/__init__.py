import os
import sys

# На Windows регистрируем пути к DLL Intel oneMKL и компилятора Intel OpenMP
if sys.platform == "win32":
    # 1. Поиск путей через переменные окружения или стандартную установку oneAPI
    mkl_root = os.environ.get("MKLROOT", r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest")
    mkl_bin = os.path.join(mkl_root, "bin")
    if os.path.isdir(mkl_bin):
        os.add_dll_directory(mkl_bin)

    oneapi_root = os.environ.get("ONEAPI_ROOT", r"C:\Program Files (x86)\Intel\oneAPI")
    compiler_candidates = [
        os.path.join(oneapi_root, "compiler", "latest", "bin"),
        os.path.join(oneapi_root, "compiler", "latest", "bin", "intel64"),
        os.path.join(oneapi_root, "compiler", "latest", "windows", "bin"),
        os.path.join(oneapi_root, "compiler", "latest", "windows", "redist", "intel64_win", "compiler"),
    ]
    for c_bin in compiler_candidates:
        if os.path.isdir(c_bin):
            try:
                os.add_dll_directory(c_bin)
            except OSError:
                pass

    # 2. Дополнительно регистрируем все папки Intel из PATH, настроенные setvars.bat
    for path_dir in os.environ.get("PATH", "").split(os.pathsep):
        if os.path.isdir(path_dir) and any(sub in path_dir.lower() for sub in ("mkl", "oneapi", "compiler")):
            try:
                os.add_dll_directory(path_dir)
            except OSError:
                pass

# Теперь импорт скомпилированного C++ модуля пройдет успешно
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