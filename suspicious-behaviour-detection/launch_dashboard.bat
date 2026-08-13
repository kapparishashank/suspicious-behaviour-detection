@echo off
set "PYTHON_EXE=C:\Users\sadam\AppData\Local\Python\bin\python.exe"
echo [System] Starting Surveillance Dashboard using %PYTHON_EXE%
%PYTHON_EXE% -m streamlit run dashboard_app.py
pause
