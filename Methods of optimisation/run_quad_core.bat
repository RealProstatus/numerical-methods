@echo off
echo ========================================
echo Running Magnus Experiments on QUAD CORE
echo MKL_NUM_THREADS=4, OMP_NUM_THREADS=4
echo ========================================
echo.

set MKL_NUM_THREADS=4
set OMP_NUM_THREADS=4

echo Starting program...
Methods of optimisation.exe

echo.
echo ========================================
echo Program finished!
echo ========================================
pause
