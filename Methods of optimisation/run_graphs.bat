@echo off
echo ========================================
echo Starting graphics scripts for Magnus Experiments
echo ========================================
echo.

cd /d "%~dp0"
echo Current folder: %CD%
echo MKL_NUM_THREADS: %MKL_NUM_THREADS%
echo OMP_NUM_THREADS: %OMP_NUM_THREADS%
echo.

echo Starting graph_experiment_1.py...
python graph_experiment_1.py
echo.

echo Starting graph_experiment_2.py...
python graph_experiment_2.py
echo.

echo Starting graph_experiment_4.py...
python graph_experiment_4.py
echo.

echo Starting graph_experiment_5.py...
python graph_experiment_5.py
echo.

echo ========================================
echo All graphics scripts completed!
echo ========================================
pause
