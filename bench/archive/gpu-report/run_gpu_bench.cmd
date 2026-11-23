@echo off
call "D:\Software\VS2022\VS2022\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d D:\Projects\downshift-server
set PYTHONWARNINGS=ignore
.venv-gpu\Scripts\python.exe -m bench.torch_vs_ort --device cuda %*
