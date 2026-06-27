@echo off
chcp 65001 >nul
title 台股分析系統

echo ========================================
echo    台股分析系統 - 單機版啟動中
echo ========================================
echo.

REM 自動抓隨身碟代號（不管插在哪個 USB 孔）
set "DRIVE=%~d0"

REM Python 在隨身碟裡的路徑
set "PYTHON=%DRIVE%\Portable_V1.2.2\data\UserProfile\AppData\Local\Programs\Python\Python313\python.exe"

REM 進到專案目錄
pushd "%~dp0"

echo 隨身碟代號: %DRIVE%
echo Python路徑: %PYTHON%

REM 清理之前的 Streamlit 殘留進程（避免 port 8501 被佔用）
echo 清理舊進程...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501" ^| findstr "LISTENING"') do taskkill /f /pid %%a >nul 2>&1
echo.

echo 啟動 Streamlit...
echo.
echo 瀏覽器開啟中，請稍候...
echo (按 Ctrl+C 可關閉伺服器)
echo.

REM 直接開瀏覽器
start http://localhost:8501

REM 用完整路徑執行，不靠 PATH
"%PYTHON%" -m streamlit run app.py

if %errorlevel% neq 0 (
    echo.
    echo 啟動失敗！錯誤碼: %errorlevel%
    echo 請確認隨身碟已正確插入
    pause
)

popd
