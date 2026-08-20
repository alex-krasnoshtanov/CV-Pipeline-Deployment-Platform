@echo off
REM One-command local startup for Windows Command Prompt.
REM Usage: scripts\start.bat

set REPO_ROOT=%~dp0..
set ENV_FILE=%REPO_ROOT%\configs\env\.env
set COMPOSE_FILE=%REPO_ROOT%\infra\local\docker-compose.yml

REM ---- prerequisites ----

where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker not found.
    echo Install Docker Desktop: https://docs.docker.com/get-docker/
    exit /b 1
)

REM ---- first-run .env setup ----

if not exist "%ENV_FILE%" (
    copy "%REPO_ROOT%\configs\env\.env.example" "%ENV_FILE%" >nul
    echo Created %ENV_FILE% from template.
    echo.
    echo Set API_KEY in %ENV_FILE% then run this script again.
    echo Generate a key: powershell -Command "[BitConverter]::ToString([Security.Cryptography.RandomNumberGenerator]::GetBytes(32)).Replace('-','').ToLower()"
    exit /b 0
)

for /f "tokens=1,* delims==" %%A in ('findstr /b "API_KEY=" "%ENV_FILE%"') do set API_KEY=%%B
if "%API_KEY%"=="" (
    echo ERROR: API_KEY is missing or empty in %ENV_FILE%.
    exit /b 1
)

REM ---- start ----

echo Starting CV Platform...
echo   Frontend : http://localhost:3000
echo   Backend  : http://localhost:8000/docs
echo.
docker compose -f "%COMPOSE_FILE%" up --build %*
