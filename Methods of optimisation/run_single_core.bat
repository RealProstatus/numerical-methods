@echo off
echo ========================================
echo Running Magnus Experiments on SINGLE CORE
echo MKL_NUM_THREADS=1, OMP_NUM_THREADS=1
echo ========================================
echo.

set MKL_NUM_THREADS=1
set OMP_NUM_THREADS=1

echo Starting program...
Methods of optimisation.exe

echo.
echo ========================================
echo Program finished!
echo ========================================
pause
