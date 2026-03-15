@echo off
echo ========================================
echo Running Magnus Experiments on DUAL CORE
echo MKL_NUM_THREADS=2, OMP_NUM_THREADS=2
echo ========================================
echo.

set MKL_NUM_THREADS=2
set OMP_NUM_THREADS=2

echo Starting program...
Methods of optimisation.exe

echo.
echo ========================================
echo Program finished!
echo ========================================
pause
