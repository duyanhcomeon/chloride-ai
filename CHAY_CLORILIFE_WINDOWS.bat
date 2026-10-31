@echo off
setlocal
cd /d "%~dp0"
echo =============================================
echo CLORILIFE AI v1.0 - Cai dat va khoi dong
 echo =============================================
where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_LAUNCHER=py -3"
) else (
    set "PYTHON_LAUNCHER=python"
)
%PYTHON_LAUNCHER% --version
if errorlevel 1 (
    echo Chua cai Python. Cai Python 3.12/3.13 va chon Add Python to PATH.
    pause
    exit /b 1
)
%PYTHON_LAUNCHER% -m pip install -r requirements.txt
if errorlevel 1 (
    echo Loi cai thu vien. Kiem tra ket noi Internet va Python.
    pause
    exit /b 1
)
%PYTHON_LAUNCHER% -m streamlit run app.py
if errorlevel 1 (
    echo Loi khoi dong. Xem thong bao o phia tren.
    pause
    exit /b 1
)
endlocal
