@echo off
echo ==============================================
echo   Starting Chronos Multi-Agent Trading System
echo ==============================================
docker compose up -d
echo Waiting for services to be ready...
timeout /t 3 /nobreak >nul
start http://localhost:8000
echo.
echo Chronos is online!
echo Dashboard: http://localhost:8000
echo API Docs:  http://localhost:8000/docs
echo.
