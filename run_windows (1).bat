@echo off
echo === PaperForge Windows Setup ===
cd /d "%~dp0"

echo Checking Python...
python --version
if errorlevel 1 (
    echo Python not found, trying py launcher...
    py --version
    if errorlevel 1 (
        echo.
        echo [ERROR] Python is not installed or not in PATH.
        echo Please install Python 3.10+ from https://www.python.org/downloads/
        echo Make sure to tick "Add python.exe to PATH" during install.
        pause
        exit /b
    ) else (
        set PYCMD=py
    )
) else (
    set PYCMD=python
)

echo.
echo Installing requirements...
%PYCMD% -m pip install --upgrade pip
%PYCMD% -m pip install streamlit PyMuPDF python-docx scikit-learn numpy openai

echo.
echo Starting PaperForge...
%PYCMD% -m streamlit run app.py

pause
