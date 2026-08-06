@echo off
setlocal enabledelayedexpansion

:: run_hls.bat
:: Headless Vitis HLS run for conv_engine: C-sim + C-synthesis (+ optional
:: co-simulation with the "cosim" argument), driven via run_hls.tcl. Same
:: structure/conventions as mnist_fpga_ver11.0.0\run_regression.bat
:: (auto-detect the Xilinx tool, log to a dedicated folder, findstr-based
:: PASS/FAIL parsing, errorlevel-driven exit codes) - adapted from
:: xvlog/xelab/xsim to vitis_hls.
::
:: Usage (double-click, or from any cmd prompt - including a fresh one
:: where vitis_hls is not yet on PATH):
::   run_hls.bat                  C-sim + C-synthesis only (fast)
::   run_hls.bat cosim            also runs co-simulation (slow)
::   run_hls.bat package          also runs Package IP
::   run_hls.bat cosim package    both (either order)
::
:: NOTE: there is deliberately no "cosim-only" mode that skips C-sim/
:: C-synthesis and reuses an already-synthesized solution - tried once, real
:: run failed with `ERROR: [COSIM 212-40] ... Synthesis was not successful`
:: even though the RTL was still on disk from a prior successful run (see
:: run_hls.tcl's header comment for the full finding). Co-simulation always
:: needs C-synthesis to run in the SAME vitis_hls invocation.
::
:: NOTE: the findstr patterns below for "synthesis completed" / "cosim
:: passed" are best-effort, written without a log from a run that actually
:: reached completion to check exact wording against (early bring-up runs
:: hung on a license issue before getting this far) - if this reports FAIL
:: but conv_engine_prj\solution1\ itself looks fine, the pattern (not the
:: run) is probably what's wrong; open hls_logs\run_hls.log directly and
:: adjust the findstr lines below to match what your install actually prints.

set "SCRIPT_DIR=%~dp0"
set "LOGDIR=%SCRIPT_DIR%hls_logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

:: ---- Find vitis_hls: use it if already on PATH (e.g. this .bat was run
:: from a "Vitis HLS <version> Command Prompt" shortcut), otherwise search
:: common install locations and source that version's settings64.bat first.
where vitis_hls >nul 2>&1
if not errorlevel 1 goto vitis_ok
set VITIS_FOUND=0
for %%V in (2024.2 2024.1 2023.2 2023.1 2022.2 2022.1 2021.2 2021.1) do (
    if exist "C:\Xilinx\Vitis\%%V\settings64.bat" (
        call "C:\Xilinx\Vitis\%%V\settings64.bat" >nul 2>&1
        set VITIS_FOUND=1
    )
)
if "%VITIS_FOUND%"=="0" (
    echo ERROR: vitis_hls not in PATH and no install found under C:\Xilinx\Vitis\
    echo        Either run this from a "Vitis HLS Command Prompt" shortcut,
    echo        or edit the version list above to match where Vitis is
    echo        actually installed on this machine.
    exit /b 1
)
:vitis_ok

:: ---- A `vitis_hls` found on PATH is not necessarily 2024.2 - a machine
:: with multiple Vitis installs (or a PATH left over from an older one) will
:: silently resolve to whichever version sorts first, and older vitis_hls
:: CLIs reject this script's `--cosim` argv outright (`ERROR: [HLS 200-101]
:: Unknown option '--cosim'` on 2021.1 - see TROUBLESHOOTING.md) instead of
:: passing it through to run_hls.tcl's $argv, making every stage report FAIL
:: with no real synthesis/cosim ever attempted. Verify the version actually
:: resolved, and if it's wrong, try sourcing 2024.2 explicitly (which
:: prepends its own bin dir onto PATH, taking priority over whatever was
:: there before) rather than silently proceeding on the wrong toolchain.
vitis_hls -version 2>nul | findstr /C:"2024.2" >nul 2>&1
if errorlevel 1 (
    if exist "C:\Xilinx\Vitis\2024.2\settings64.bat" (
        call "C:\Xilinx\Vitis\2024.2\settings64.bat" >nul 2>&1
    )
    vitis_hls -version 2>nul | findstr /C:"2024.2" >nul 2>&1
    if errorlevel 1 (
        echo ERROR: vitis_hls on PATH is not version 2024.2, and sourcing
        echo        C:\Xilinx\Vitis\2024.2\settings64.bat did not fix it.
        echo        This project's run_hls.tcl targets 2024.2 specifically;
        echo        running any other version can fail in confusing ways
        echo        ^(e.g. an older vitis_hls rejecting --cosim as an unknown
        echo        option instead of passing it to the Tcl script^).
        echo        Open a "Vitis HLS 2024.2 Command Prompt" shortcut instead
        echo        of a plain cmd/PowerShell, then re-run this script.
        exit /b 1
    )
)

:: Cosim/Package IP are selected via the RUN_HLS_COSIM/RUN_HLS_PACKAGE
:: environment variables, not vitis_hls command-line arguments -
:: `vitis_hls -f run_hls.tcl --cosim` fails on both 2021.1 and 2024.2 with
:: `ERROR: [HLS 200-101] Unknown option '--cosim'` (vitis_hls's own CLI
:: parser validates every token after `-f <script>` against its fixed
:: option list; it does not pass unrecognized ones through to the Tcl
:: script's $argv - see TROUBLESHOOTING.md and run_hls.tcl's header
:: comment). Both args accepted in either order/combination.
set "RUN_HLS_COSIM="
set "RUN_HLS_PACKAGE="
if /i "%~1"=="cosim"   (set "RUN_HLS_COSIM=1")
if /i "%~1"=="package" (set "RUN_HLS_PACKAGE=1")
if /i "%~2"=="cosim"   (set "RUN_HLS_COSIM=1")
if /i "%~2"=="package" (set "RUN_HLS_PACKAGE=1")

echo.
if defined RUN_HLS_COSIM (
    if defined RUN_HLS_PACKAGE (
        echo =^> Running vitis_hls ^(C-sim + C-synthesis + co-simulation + Package IP^) - this is slow
    ) else (
        echo =^> Running vitis_hls ^(C-sim + C-synthesis + co-simulation^) - this is slow
    )
) else (
    if defined RUN_HLS_PACKAGE (
        echo =^> Running vitis_hls ^(C-sim + C-synthesis + Package IP^)
    ) else (
        echo =^> Running vitis_hls ^(C-sim + C-synthesis only; pass "cosim" and/or "package" as arguments to also run co-simulation / Package IP^)
    )
)

set "LOG=%LOGDIR%\run_hls.log"
cd /d "%SCRIPT_DIR%"
call vitis_hls -f run_hls.tcl > "%LOG%" 2>&1
if errorlevel 1 (
    echo FAIL  vitis_hls exited with an error -- see %LOG%
    powershell -NoProfile -Command "Get-Content -Tail 40 -Path '%LOG%'"
    exit /b 1
)

set OVERALL_FAIL=0

:: TWO separate bugs were fixed here on 2026-08-05 - see
:: doc/conv_engine_requant_troubleshooting.md 29-f.
::
:: The two bugs were MODE-DEPENDENT and complementary - which is why the
:: false FAIL got noticed and lived on the backlog for days while the far
:: more dangerous false PASS went unseen:
::
:: 1. FALSE POSITIVE, in NORMAL mode (the serious one): this used an
::    UNANCHORED `findstr /C:"ALL CONFIGS PASS"`. run_hls.tcl's own
::    announcement line ("...expect configs A through G to each print PASS,
::    then "ALL CONFIGS PASS"...") contains that exact substring and lands in
::    the same log, so in normal mode the pattern matched no matter what
::    C-sim actually did - including runs that mismatched, crashed, were
::    interrupted, or never started. Verified against the real 13:00 log: 0
::    occurrences of "Finished Command csim_design", yet 1 unanchored match;
::    and against a synthetic log containing a genuine "MISMATCH", which the
::    old gate also called PASS. `/B` anchors to start-of-line, where only
::    the testbench's own result is printed. Same bug and same fix already
::    applied to pool_upsample_route's run_hls.bat on 2026-07-29 -
::    conv_engine's copy never got it.
::
:: 2. FALSE NEGATIVE, in RUN_HLS_COSIM_LAYER mode: run_hls.tcl takes its
::    other branch there and never prints the announcement, while the
::    testbench's --cosim-only path prints "COSIM CONFIG PASS" instead of
::    "ALL CONFIGS PASS" (conv_engine_tb.cpp) - so real-layer cosim runs
::    matched neither and always reported FAIL. Both sentinels accepted
::    below.
set "CSIM_RESULT="
findstr /B /C:"ALL CONFIGS PASS" "%LOG%" >nul 2>&1
if not errorlevel 1 set "CSIM_RESULT=all"
findstr /B /C:"COSIM CONFIG PASS" "%LOG%" >nul 2>&1
if not errorlevel 1 set "CSIM_RESULT=cosim"

if "!CSIM_RESULT!"=="all" (
    echo PASS  C simulation -- all 7 configs ^(A-G^) matched the reference
) else if "!CSIM_RESULT!"=="cosim" (
    echo PASS  C simulation -- single real-layer/cosim config matched the reference ^(RUN_HLS_COSIM_LAYER mode^)
) else (
    echo FAIL  C simulation printed neither ALL CONFIGS PASS nor COSIM CONFIG PASS -- see %LOG%
    findstr /C:"MISMATCH" /C:"FAIL" /C:"Assertion" "%LOG%"
    set OVERALL_FAIL=1
)

:: NOTE: matching "Finished Command csynth_design" specifically, not just
:: "csynth_design" - that substring also appears the moment synthesis
:: STARTS ("Running: csynth_design"), which would false-positive PASS even
:: if synthesis stalled or errored partway through. "Finished Command
:: csynth_design" mirrors the exact pattern csim_design was confirmed to
:: print on completion in an earlier real run
:: ("Finished Command csim_design CPU user time...").
findstr /I /C:"Finished Command csynth_design" "%LOG%" >nul 2>&1
if not errorlevel 1 (
    echo PASS  C synthesis ran -- see conv_engine_prj\solution1\syn\report\conv_engine_csynth.rpt for DSP/BRAM/FF numbers
) else (
    echo FAIL  no C-synthesis completion marker found in %LOG% ^(pattern may need adjusting - see NOTE at top of this script^)
    set OVERALL_FAIL=1
)

if defined RUN_HLS_COSIM (
    findstr /I /C:"Finished Command cosim_design" /C:"co-simulation finished: PASS" "%LOG%" >nul 2>&1
    if not errorlevel 1 (
        echo PASS  C/RTL co-simulation
    ) else (
        echo FAIL  no co-simulation success marker found in %LOG% ^(pattern may need adjusting^)
        set OVERALL_FAIL=1
    )
)

if defined RUN_HLS_PACKAGE (
    :: No confirmed log marker for export_design's own success message
    :: (same caveat this file's NOTE at the top already gives for
    :: csynth/cosim patterns) - check for the export directory itself
    :: instead, same approach as pool_upsample_route/run_hls.bat.
    if exist "%SCRIPT_DIR%conv_engine_prj\solution1\impl\ip" (
        echo PASS  Package IP -- exported -- see conv_engine_prj\solution1\impl\ip\
    ) else (
        echo FAIL  Package IP -- impl\ip not found -- see %LOG%
        findstr /C:"PACKAGE-IP ERROR" "%LOG%"
        set OVERALL_FAIL=1
    )
)

echo.
echo ========================================
if "%OVERALL_FAIL%"=="0" (
    echo   conv_engine HLS run complete
) else (
    echo   One or more steps did not confirm success -- see %LOG%
)
echo   Full reports: %SCRIPT_DIR%conv_engine_prj\solution1\
echo ========================================
echo.

exit /b %OVERALL_FAIL%
