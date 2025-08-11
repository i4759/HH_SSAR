@echo off
echo Starting HH_SSAR project...

REM Переход в директорию скрипта
cd /d "%~dp0"

REM Попытка найти Python разными способами
echo Searching Python...

REM Проверка python
python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    goto :python_found
)

REM Проверка python3
python3 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python3
    goto :python_found
)

REM Проверка py launcher
py --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py
    goto :python_found
)


echo Python not found, please make sure that:
echo 1. Python is installed from the official python.org website
echo 2. "Add Python to PATH" is checked during installation
pause
exit /b 1

:python_found
echo Python found: %PYTHON_CMD%
%PYTHON_CMD% --version

REM Создание виртуального окружения если его нет
if not exist ".venv" (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv .venv
)

REM Активация виртуального окружения
call .venv\Scripts\activate

REM Обновление pip
%PYTHON_CMD% -m pip install --upgrade pip

REM Установка зависимостей если есть requirements.txt
if exist "requirements.txt" (
    echo Installing dependencies...
    %PYTHON_CMD% -m pip install -r requirements.txt
)

REM Запуск основного скрипта
if exist "main.py" (
    echo Starting main.py...
    %PYTHON_CMD% main.py
) else (
    echo File main.py not found!
)

pause
