@echo off
echo Starting graphics scripts...
echo.

cd /d "%~dp0"
echo Current folder: %CD%
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

echo Done!
pause
