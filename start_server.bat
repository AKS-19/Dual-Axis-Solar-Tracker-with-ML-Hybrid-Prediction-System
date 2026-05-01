@echo off
title Solar Tracker — Full Stack
chcp 65001 > nul

echo ========================================================
echo   Solar Tracker  ^|  2-Axis Precision Tracker
echo ========================================================
echo.

echo [1/4] Clearing any stale server processes...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5050 "') do (
    taskkill /PID %%a /F > nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8080 "') do (
    taskkill /PID %%a /F > nul 2>&1
)
timeout /t 1 /nobreak > nul
echo    Done.

echo.
echo [2/4] Installing dependencies (silent)...
python -m pip install flask flask-cors numpy scikit-learn pandas --quiet --disable-pip-version-check
echo    Done.

echo.
echo [3/4] Starting ML API Backend on port 5050...
start "Solar Tracker ML API" cmd /c "cd /d %~dp0 && python api_server.py & pause"
timeout /t 4 /nobreak > nul

echo [4/4] Starting Frontend HTTP server on port 8080...
start "Solar Tracker Frontend" /MIN cmd /c "cd /d %~dp0 && python -m http.server 8080"
timeout /t 2 /nobreak > nul

echo.
echo ========================================================
echo   READY  --  Open these URLs in your browser:
echo.
echo   3D Simulator : http://localhost:8080/deepseek_html_20260423_9e251f.html
echo   Analytics    : http://localhost:8080/analytics.html
echo   API Ping     : http://localhost:5050/ping
echo ========================================================
echo.

start "" "http://localhost:8080/deepseek_html_20260423_9e251f.html"
timeout /t 2 /nobreak > nul
start "" "http://localhost:8080/analytics.html"

echo   Both pages launched. Close this window when done.
pause