@echo off
setlocal
if defined ISAAC_PATH (
    set "ISAAC_PYTHON=%ISAAC_PATH%\python.bat"
) else (
    set "ISAAC_PYTHON=D:\0-NVIDIA-Omniverse\isaac-sim\python.bat"
)

if not exist "%ISAAC_PYTHON%" (
    echo [ERROR] NVIDIA Isaac Sim python.bat not found at: "%ISAAC_PYTHON%"
    echo Set the ISAAC_PATH environment variable to your Isaac Sim installation root.
    exit /b 1
)

echo [VoyagerSim] Running Autonomous Headless Verification Suite...
call "%ISAAC_PYTHON%" "%~dp0voyager_interstellar_sim\main_voyager_sim.py" --headless --verify %*
endlocal
