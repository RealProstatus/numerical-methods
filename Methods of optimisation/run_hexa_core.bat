@echo off
echo ========================================
echo Running Magnus Experiments on HEXA CORE
echo MKL_NUM_THREADS=6, OMP_NUM_THREADS=6
echo ========================================
echo.

set MKL_NUM_THREADS=6
set OMP_NUM_THREADS=6

echo Starting program...
Methods of optimisation.exe

echo.
echo ========================================
echo Program finished!
echo ========================================
pause
