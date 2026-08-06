# run_hls.tcl
#
# Headless Vitis HLS batch script for conv_engine - the Tcl-driven
# equivalent of clicking through the GUI's New Project / Add Files / Run
# C Simulation / Run C Synthesis buttons. Run via:
#
#   vitis_hls -f run_hls.tcl                        # C-sim + C-synthesis only
#   RUN_HLS_COSIM=1 vitis_hls -f run_hls.tcl         # also runs co-simulation (slow)
#   RUN_HLS_PACKAGE=1 vitis_hls -f run_hls.tcl       # also runs Package IP
#
# Package IP (`export_design`) needs csynth_design to have already run in
# THIS SAME invocation (same constraint cosim_design has - see the "TRIED
# AND REJECTED" note right below) - it only ever adds an export step after
# csim/csynth(/cosim), never a way to skip them. Exports to
# conv_engine_prj/solution1/impl/ip - that is the IP_REPO_PATH
# vivado/create_bd.tcl needs (that script's own header comment already
# pointed at this path, before Package IP had ever actually been run).
#
# TRIED AND REJECTED (2026-07-24): a "cosim-only" mode that skipped
# csim_design/csynth_design and just reopened the existing solution
# (open_project/open_solution WITHOUT -reset) to save time re-running
# cosim_design alone. Real run against this project: failed immediately with
# `ERROR: [COSIM 212-40] C/RTL co-simulation cannot be started, possible
# causes: 1) Synthesis was not successful; ...` even though solution1's RTL
# from the prior full run was still on disk and untouched. Conclusion:
# cosim_design's "synthesis succeeded" check is apparently a session-local
# flag set by csynth_design actually running in THAT SAME `vitis_hls -f ...`
# process, not something reconstructed from re-opening a solution that was
# already synthesized in a PRIOR process - reopening the project/solution via
# Tcl batch mode does not restore it. Do not re-attempt this shortcut without
# first confirming (in the *Vitis HLS Tcl API* docs, not by guessing again)
# that there's a supported way to mark a reopened solution "synthesized" for
# cosim_design's purposes. Until then, co-simulation always needs
# csynth_design to run first IN THE SAME invocation - i.e. always via
# RUN_HLS_COSIM=1 (the full C-sim + C-synthesis + co-simulation path below),
# never a synthesis-skipping shortcut.
#
# Cosim is selected through the RUN_HLS_COSIM *environment* variable, not a
# `vitis_hls` command-line argument - confirmed on both 2021.1 and 2024.2
# that `vitis_hls -f run_hls.tcl --cosim` fails with `ERROR: [HLS 200-101]
# Unknown option '--cosim'` (see TROUBLESHOOTING.md). vitis_hls's own CLI
# parser validates every token after `-f <script>` against its fixed option
# list (`-eval`/`-i`/`-l`/`-n`/`-nosplash`/`-p`/`-terse`/`-version`, see its
# `-help` SYNTAX output) - there is no passthrough to the Tcl script's
# `$argv` this way. An environment variable sidesteps vitis_hls's CLI parser
# entirely (Tcl reads it via `$::env(...)`, independent of how the process
# was invoked). run_hls.sh/run_hls.bat set this variable before calling
# `vitis_hls -f run_hls.tcl` with no extra arguments.
#
# (or use run_hls.sh, which wraps this with PASS/FAIL reporting). This is
# the same headless-automation idea as
# mnist_fpga_ver11.0.0/run_regression.sh uses for xvlog/xelab/xsim, just
# driving vitis_hls's own Tcl API instead.
#
# ---------------------------------------------------------------------------
# Targets Vitis HLS 2024.2, matching doc/planning_기획서.md §12.1's toolchain
# choice. The core Tcl commands used below (open_project, set_top, add_files,
# csim_design, csynth_design, cosim_design) have been stable across HLS
# releases for years, so this script isn't expected to need version-specific
# changes - but if csim/csynth behaves differently than expected, that's the
# first thing to suspect.
#
# Part note: PART below (xck26-sfvc784-2LV-c) is confirmed to resolve
# correctly via `set_part` - verified against a real Vitis HLS run (2021.1,
# during early bring-up before switching to 2024.2). If it doesn't resolve
# on this install, replace PART with a generic Zynq UltraScale+ part instead
# (check `get_parts -family {zynq ultrascale+}` in a Vivado Tcl console for
# what's installed) - C synthesis DSP/LUT/BRAM counts are largely
# part-family-independent for early sizing; only exact timing/place-and-route
# would differ.
# ---------------------------------------------------------------------------

set PART      "xck26-sfvc784-2LV-c"
set CLOCK_NS  5.0    ;# 200 MHz. Tried 4.0 (250MHz) at PE_OC=20 - real
                     ;# csynth result: LUT 104,577 (89%) -> 119,320
                     ;# (**101%, over budget**), FF 44,750 (19%) -> 91,990
                     ;# (39%) - Vitis HLS added enough extra pipeline/retiming
                     ;# registers to hit the tighter target that LUT no
                     ;# longer fits, even though Estimated Fmax itself rose
                     ;# to 342 MHz. Reverted to 200MHz, keeping the
                     ;# cosim-confirmed PE_OC=20/200MHz combination - see
                     ;# RESOURCE_BUDGET.md §5, "Raising PE_OC after packing"
                     ;# / "Trying 250MHz at PE_OC=20" for the full numbers
                     ;# and the reasoning on why clock and PE_OC compete for
                     ;# the same LUT budget rather than combining for free.
set RUN_COSIM   [info exists ::env(RUN_HLS_COSIM)]
set RUN_PACKAGE [info exists ::env(RUN_HLS_PACKAGE)]
;# Optional real-layer index (0-12, see HW/real_layers_data.h's REAL_LAYERS[])
;# to cosim instead of the synthetic config-A-cosim shape - see
;# conv_engine_tb.cpp main()'s cosim_real_layer handling and
;# doc/plan_no_board_verification_and_perf.md P1. Passed through via env var,
;# same reason RUN_HLS_COSIM itself is (vitis_hls's CLI rejects extra argv
;# tokens outright - see TROUBLESHOOTING.md §13), not via `cosim_design`'s
;# own -argv, which IS a real Tcl-level passthrough to the compiled
;# testbench's argv (unrelated to vitis_hls's own CLI parsing).
set RUN_COSIM_LAYER [expr {[info exists ::env(RUN_HLS_COSIM_LAYER)] ? $::env(RUN_HLS_COSIM_LAYER) : ""}]

;# Debug-only override: capture a waveform on the targeted real-layer cosim
;# run instead of the default -trace_level none, to see which loop is
;# actually consuming cycles instead of guessing from the cycle count alone.
set RUN_COSIM_TRACE_ALL [info exists ::env(RUN_HLS_COSIM_TRACE_ALL)]

set SCRIPT_DIR [file dirname [file normalize [info script]]]
set HW_DIR     "$SCRIPT_DIR/HW"

open_project -reset conv_engine_prj
add_files      "$HW_DIR/conv_engine.cpp"
add_files -tb  "$HW_DIR/conv_engine_tb.cpp"
set_top conv_engine

open_solution -reset "solution1"
set_part $PART
create_clock -period $CLOCK_NS -name default

;# TIER B2 (2026-08-05, TIMING_CLOSURE_PLAN.md §4 Tier B / §3 원인 A).
;# Vitis defaults clock uncertainty to 27% of the period (1.35 ns at
;# CLOCK_NS=5.0), and the scheduler then packs logic right up to that
;# boundary: the csynth report read `Target 5.00 / Estimated 3.650`, and
;# 3.650 = 5.00 - 1.35 EXACTLY. That is not a comfortable margin, it is HLS
;# reporting that it spent the entire budget it was given.
;#
;# On this design that budget was wrong, because routing takes 63-80% of the
;# real datapath delay (measured on the routed 200 MHz run) while the HLS
;# estimate models only logic. The result was a routed WNS of -0.021 ns -
;# missing 200 MHz by 21 ps.
;#
;# Raising uncertainty to 2.0 ns (40%) forces the scheduler to leave real
;# slack, which in practice means inserting registers between the TREE_L1/
;# L2/L3 reduction stages instead of chaining them into one stage.
;#
;# Cost is pipeline depth, not throughput: MAC_REDUCE is II=1 with a trip
;# count up to 8x3x3 = 72, so 1-2 extra stages is under 3% on that loop.
;#
;# CHECK AFTER CHANGING THIS: conv_engine_csynth.rpt's Estimated should drop
;# to <= 3.0 ns. If it lands at 5.00 - 2.00 = 3.00 exactly, the scheduler has
;# again spent the whole budget and the number is a ceiling, not a margin.
set_clock_uncertainty 2.0

if {$RUN_COSIM_LAYER ne ""} {
    puts "\n==> C Simulation - RUN_HLS_COSIM_LAYER is set, so this ALSO only\
runs the one targeted real-layer check via --cosim-only (same argv path\
cosim_design uses below), skipping the full self-test + 7-config + 13-real-\
layer battery. That full battery isn't being skipped because it's\
untrustworthy - it's already passed and been confirmed multiple times\
(README.md's own history) - it's skipped here purely to cut real wall-clock\
time when the only goal is one cosim data point\
(doc/plan_no_board_verification_and_perf.md P1). It is NOT skipped when\
RUN_HLS_COSIM_LAYER is unset, so the normal full-verification path\
(csim_design with no -argv) is completely unaffected by this change."
    csim_design -argv "--cosim-only $RUN_COSIM_LAYER"
} else {
    puts "\n==> C Simulation - expect configs A through G to each print PASS,\
then \"ALL CONFIGS PASS\" (see HW/conv_engine_tb.cpp)"
    csim_design
}

puts "\n==> C Synthesis - check the Interface/DSP/BRAM/FF numbers in\
conv_engine_prj/solution1/syn/report/conv_engine_csynth.rpt against\
RESOURCE_BUDGET.md's PE_OC=16, TR=16 -> ~256 DSP estimate"
csynth_design

if {$RUN_COSIM} {
    if {$RUN_COSIM_LAYER ne ""} {
        puts "\n==> C/RTL Co-simulation - slow, only runs with --cosim.\
Running REAL layer index $RUN_COSIM_LAYER from HW/real_layers_data.h's\
REAL_LAYERS[] (real shape/weights/golden output), not the synthetic\
config-A-cosim shape - see conv_engine_tb.cpp's run_real_layer_cosim() and\
doc/plan_no_board_verification_and_perf.md P1. This can take dramatically\
longer than the toy shape - real layers are ~1000x its size.\
-trace_level none: xsim is single-threaded regardless of core count (this\
is NOT a knob for using more CPU), but skipping waveform/signal-dump\
generation removes real per-cycle overhead - only the PASS/FAIL + cycle\
count in conv_engine_cosim.rpt is needed here, not a waveform to debug.\
Set RUN_HLS_COSIM_TRACE_ALL=1 to override this to -trace_level all when a\
waveform IS the goal (slower, debug-only)."
        set trace_opt [expr {$RUN_COSIM_TRACE_ALL ? "all" : "none"}]
        cosim_design -trace_level $trace_opt -argv "--cosim-only $RUN_COSIM_LAYER"
    } else {
        puts "\n==> C/RTL Co-simulation - slow, only runs with --cosim.\
Confirms the achieved II for the MAC_IC/MAC_KY/MAC_KX reduction\
(see the code-review finding on conv_engine.cpp's loop-carried accumulator).\
Runs only conv_engine_tb.cpp's config-A-cosim (full m_axi-depth buffers) via\
--cosim-only - cosim RTL-simulates the FIRST conv_engine() call the compiled\
testbench makes, and configs A-H's small buffers SIGSEGV against the full\
m_axi depth if reached first (see conv_engine_tb.cpp main()). Set\
RUN_HLS_COSIM_LAYER=<0-12> to cosim a REAL layer shape instead."
        cosim_design -argv "--cosim-only"
    }
} else {
    puts "\n==> Skipping co-simulation (pass --cosim to also run it)"
}

if {$RUN_PACKAGE} {
    puts "\n==> Package IP - exporting to conv_engine_prj/solution1/impl/ip\
(see SW/conv_engine_hw_driver.h's header comment for why the real\
generated xconv_engine_hw.h from THIS export - ip_catalog format, not the\
older impl/misc/verilog-only export already on disk from before this\
project was forked/renamed - is what that driver was actually built\
from, and why vivado/create_bd.tcl's ip_repo_paths needs impl/ip\
specifically, not impl/ alone)"
    export_design -rtl verilog -format ip_catalog
} else {
    puts "\n==> Skipping Package IP (set RUN_HLS_PACKAGE=1 to also run it)"
}

close_project
exit
