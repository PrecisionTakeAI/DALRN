@echo off
echo Starting Groq LPU Services...
echo.

REM Load environment variables
if exist .env.groq (
    for /f "delims=" %%x in (.env.groq) do set %%x
)

REM Start services
echo Starting Groq Search Service on port 9001...
start /B python -m services.search.groq_search_service

echo Starting Groq FHE Service on port 9002...
start /B python -m services.fhe.groq_fhe_service

echo.
echo Groq services started!
echo Check http://localhost:9001/health for search service
echo Check http://localhost:9002/health for FHE service
