@echo off
rem ---------------------------------------------------------------
rem  Launcher for cull.py
rem  Kept ASCII-only on purpose: cmd.exe reads .bat files using the
rem  console code page, so non-ASCII text here would get mangled.
rem  All Chinese prompts live in cull.py instead.
rem
rem  Usage:
rem    double-click           -> window opens, drag a photo folder in
rem    drag folder onto file  -> processes it, then waits
rem    run.bat "D:\pics" --full
rem ---------------------------------------------------------------
chcp 65001 >nul
setlocal
set "HERE=%~dp0"

if not "%~1"=="" goto WITHARGS

rem no argument: interactive prompt, then auto-open the report
"%HERE%venv\Scripts\python.exe" "%HERE%cull.py" --open --pause
goto END

:WITHARGS
rem only a folder was given (drag-and-drop) -> wait so the result stays visible
if "%~2"=="" (
  "%HERE%venv\Scripts\python.exe" "%HERE%cull.py" %* --pause
) else (
  "%HERE%venv\Scripts\python.exe" "%HERE%cull.py" %*
)
goto END

:END
endlocal
