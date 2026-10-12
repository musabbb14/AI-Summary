@echo off
echo Pushing AI-Summary to GitHub (https://github.com/musabbb14/AI-Summary)...
git push -u origin main
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo SUCCESS: Repository pushed to GitHub!
    echo URL: https://github.com/musabbb14/AI-Summary
    echo ========================================
) else (
    echo.
    echo [ERROR] Push failed. Make sure you created the empty repository at https://github.com/new with name AI-Summary.
)
pause
