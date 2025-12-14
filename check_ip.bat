@echo off
REM Windows için IP kontrolü

echo Sunucu IP Adresleri:
echo ===================
echo.

echo Public IP:
curl -s ifconfig.me
echo.
echo.

echo Local IP:
ipconfig | findstr /i "IPv4"
echo.

pause


