@echo off
call conda create -n ftirml -y
echo Finished creating environment, now activating
timeout /t 2 >nul
call conda activate ftirml
echo Environment activated, now installing python and spyder
timeout /t 2 >nul
call conda install python==3.10 spyder -y
echo Finished installing python V3.10 and spyder, now installing pypi packages
timeout /t 2 >nul
call pip install -r pip_requirements.txt

echo Finished installing pypi packages
timeout /t 2 >nul


set /p user_input="Install done. Do you want to launch Spyder? (Y/n): "
if /i "%user_input%"=="n" (
    echo Exiting installer...
    exit /b
) else (
    call spyder
)

pause