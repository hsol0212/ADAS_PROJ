#!/usr/bin/env bash
# run_hls.sh
# Headless Vitis HLS run for conv_engine: C-sim + C-synthesis (+ optional
# co-simulation with --cosim), driven via run_hls.tcl. Same automation
# pattern as mnist_fpga_ver11.0.0/run_regression.sh, adapted from
# xvlog/xelab/xsim to vitis_hls.
#
# Usage (from this directory, hls/conv_engine/):
#   bash run_hls.sh                     # C-sim + C-synthesis only (fast)
#   bash run_hls.sh --cosim             # also runs co-simulation (slow)
#   bash run_hls.sh --package           # also runs Package IP
#   bash run_hls.sh --cosim --package   # both (any order)
#
# NOTE: there is deliberately no "--cosim-only" mode that skips C-sim/
# C-synthesis and reuses an already-synthesized solution - tried once, real
# run failed with `ERROR: [COSIM 212-40] ... Synthesis was not successful`
# even though the RTL was still on disk from a prior successful run (see
# run_hls.tcl's header comment for the full finding). Co-simulation always
# needs C-synthesis to run in the SAME vitis_hls invocation.
#
# Prerequisites: `vitis_hls` must be on PATH - source Vitis's settings
# script first, e.g.:
#   source /tools/Xilinx/Vitis/2024.2/settings64.sh
#
# Exit code: 0 = csim+csynth (+cosim if requested) all completed AND csim
# printed "ALL CONFIGS PASS"; 1 otherwise.
#
# NOTE: the grep patterns below for "synthesis completed" / "cosim passed"
# are best-effort, written without a log from a run that actually reached
# completion to check against (early bring-up runs hung on a license issue
# before getting this far) - if this script reports FAIL but
# conv_engine_prj/solution1's own reports look fine, the pattern (not the
# run) is probably what's wrong; check hls_logs/run_hls.log directly and
# adjust the grep below to match what your install actually prints.

set -uo pipefail   # NOT -e: vitis_hls's exit code is checked explicitly below

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOGDIR="$SCRIPT_DIR/hls_logs"
mkdir -p "$LOGDIR"

if [ -t 1 ]; then
    C_RED='\033[0;31m'; C_GRN='\033[0;32m'; C_YLW='\033[1;33m'; C_RST='\033[0m'
else
    C_RED=''; C_GRN=''; C_YLW=''; C_RST=''
fi

step() { printf "\n${C_YLW}==> %s${C_RST}\n" "$*"; }
pass() { printf "${C_GRN}PASS${C_RST}  %s\n" "$*"; }
fail() { printf "${C_RED}FAIL${C_RST}  %s\n" "$*"; }

if ! command -v vitis_hls >/dev/null 2>&1; then
    fail "vitis_hls not found on PATH"
    printf "       source Vitis's settings script first, e.g.:\n"
    printf "       source /tools/Xilinx/Vitis/2024.2/settings64.sh\n"
    exit 1
fi

COSIM=0
PACKAGE=0
for arg in "$@"; do
    case "$arg" in
        --cosim)   COSIM=1 ;;
        --package) PACKAGE=1 ;;
    esac
done
if [ "$COSIM" -eq 1 ] && [ "$PACKAGE" -eq 1 ]; then
    step "Running vitis_hls (C-sim + C-synthesis + co-simulation + Package IP) - this is slow"
elif [ "$COSIM" -eq 1 ]; then
    step "Running vitis_hls (C-sim + C-synthesis + co-simulation) - this is slow"
elif [ "$PACKAGE" -eq 1 ]; then
    step "Running vitis_hls (C-sim + C-synthesis + Package IP)"
else
    step "Running vitis_hls (C-sim + C-synthesis only; pass --cosim and/or --package to also run co-simulation / Package IP)"
fi

LOG="$LOGDIR/run_hls.log"
# Cosim/Package IP are selected via RUN_HLS_COSIM/RUN_HLS_PACKAGE (read by
# run_hls.tcl through $::env(...)), not vitis_hls command-line arguments -
# `vitis_hls -f run_hls.tcl --cosim` fails with `ERROR: [HLS 200-101]
# Unknown option '--cosim'` on both 2021.1 and 2024.2: vitis_hls's own CLI
# parser validates every token after `-f <script>` against its fixed option
# list and does not pass unrecognized ones through to the Tcl script's
# $argv (see TROUBLESHOOTING.md and run_hls.tcl's header comment).
[ "$COSIM" -eq 1 ]   && export RUN_HLS_COSIM=1
[ "$PACKAGE" -eq 1 ] && export RUN_HLS_PACKAGE=1
vitis_hls -f run_hls.tcl >"$LOG" 2>&1
RC=$?

if [ "$RC" -ne 0 ]; then
    fail "vitis_hls exited with status $RC - see $LOG"
    tail -40 "$LOG"
    exit 1
fi

overall_fail=0

# Anchored with ^ and accepting both sentinels - see run_hls.bat's comment
# block here and doc/conv_engine_requant_troubleshooting.md 29-f. Unanchored,
# this matched run_hls.tcl's own announcement line (which quotes the string
# "ALL CONFIGS PASS") and so reported PASS on every run, including runs where
# C-sim never completed. RUN_HLS_COSIM_LAYER mode prints COSIM CONFIG PASS
# instead, which unanchored-and-single-sentinel reported as FAIL.
if grep -q "^ALL CONFIGS PASS" "$LOG"; then
    pass "C simulation - all 7 configs (A-G) matched the reference"
elif grep -q "^COSIM CONFIG PASS" "$LOG"; then
    pass "C simulation - single real-layer/cosim config matched the reference (RUN_HLS_COSIM_LAYER mode)"
else
    fail "C simulation printed neither ALL CONFIGS PASS nor COSIM CONFIG PASS - see $LOG"
    grep -E "MISMATCH|FAIL|Assertion|assert" "$LOG" | head -20 || true
    overall_fail=1
fi

# NOTE: matching "Finished Command csynth_design" specifically, not a bare
# "csynth_design" substring - that also appears the moment synthesis STARTS
# ("Running: csynth_design"), which would false-positive PASS even if
# synthesis stalled or errored partway through. "Finished Command
# csynth_design" mirrors the exact pattern csim_design was confirmed to
# print on completion in an earlier real run ("Finished Command csim_design
# CPU user time...").
if grep -qi "Finished Command csynth_design" "$LOG"; then
    pass "C synthesis ran - see conv_engine_prj/solution1/syn/report/conv_engine_csynth.rpt for DSP/BRAM/FF numbers"
else
    fail "no C-synthesis completion marker found in $LOG (pattern may need adjusting - see NOTE at top of this script)"
    overall_fail=1
fi

if [ "$COSIM" -eq 1 ]; then
    if grep -qi "Finished Command cosim_design\|co-simulation finished: PASS" "$LOG"; then
        pass "C/RTL co-simulation"
    else
        fail "no co-simulation success marker found in $LOG (pattern may need adjusting)"
        overall_fail=1
    fi
fi

if [ "$PACKAGE" -eq 1 ]; then
    # No confirmed log marker for export_design's own success message
    # (same caveat this file's NOTE at the top already gives for
    # csynth/cosim patterns) - check for the export directory itself
    # instead, same approach as pool_upsample_route/run_hls.sh.
    if [ -d "$SCRIPT_DIR/conv_engine_prj/solution1/impl/ip" ]; then
        pass "Package IP - exported - see conv_engine_prj/solution1/impl/ip/"
    else
        fail "Package IP: impl/ip not found - see $LOG"
        grep -E "PACKAGE-IP ERROR" "$LOG" | head -20 || true
        overall_fail=1
    fi
fi

printf "\n========================================\n"
if [ "$overall_fail" -eq 0 ]; then
    printf "${C_GRN}  conv_engine HLS run complete${C_RST}\n"
else
    printf "${C_RED}  One or more steps did not confirm success - see $LOG${C_RST}\n"
fi
printf "  Full reports: %s/conv_engine_prj/solution1/\n" "$SCRIPT_DIR"
printf "========================================\n\n"

exit "$overall_fail"
