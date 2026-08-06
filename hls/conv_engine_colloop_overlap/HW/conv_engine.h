#ifndef CONV_ENGINE_H
#define CONV_ENGINE_H

#include <ap_int.h>
#include <stdint.h>
#include "pack4.h"

// ---------------------------------------------------------------------------
// Shared, time-multiplexed convolution engine.
//
// Replaces the "one dedicated HLS IP per layer" approach (hls/conv_layer1/):
// instead of instantiating a separate fixed-function pipeline per layer,
// which does not scale to YOLOv2-tiny channel counts on the KR260's DSP
// budget, THIS single IP is reused for every conv layer, one at a time.
// Software reconfigures its geometry registers + DDR addresses per layer and
// re-triggers it (see SW/network_run.c) - conceptually the same idea as a
// CPU executing different instructions, except here the "instruction" is a
// layer's shape and the "data" is that layer's weights/feature maps in DDR.
//
// Consequence: every layer's input AND output feature map lives in DDR, not
// a live AXI4-Stream from a producer - even the very first layer's camera
// frame must land in a DDR buffer (via the existing V4L2/DMA capture path)
// before this engine reads it the same way it reads any other layer's input.
// This also means no AXI DMA IP is needed for the layer chain itself (see
// vivado/create_bd.tcl) - only AXI-Lite control and an HP port for the two
// m_axi bundles below.
//
// conv_layer1/ is NOT deleted or replaced by this - it stays as the
// per-layer HLS learning exercise doc/hls_study_plan.md is built around.
// This module is the production architecture; see README.md for how the two
// relate.
// ---------------------------------------------------------------------------

// ---- Compile-time upper bounds: size on-chip buffers, NOT a single layer's
// actual size. Runtime geometry (img_h/img_w/in_ch/out_ch/k, see
// conv_engine() below) must stay within these bounds.
//
// Confirmed against the real Phase 1 network (python_teammate/darknet_golden,
// a finished YOLOv3-tiny-ADAS 5-class INT8 golden model -
// artifacts/int8/model_manifest.json): width=512, height=288, and 8 of its
// 13 conv layers have in_ch > 128 (max 1024, at layer 13's 1x1 conv) - NOT a
// couple of edge cases, more than half. MAX_IMG_W raised 416->512 and input-
// channel tiling (this section's whole point) added specifically because of
// those two numbers, not as speculative headroom.
const unsigned MAX_IMG_W  = 512;   // largest spatial width across all real layers (width=512, height=288)
const unsigned MAX_IN_CH  = 128;   // on-chip tile width ONLY now (see IC_TILE, conv_engine.cpp) - window/
                                    // wtile_packed are still sized to this, unchanged from the untiled version;
                                    // it does NOT bound a layer's real total in_ch anymore - MAX_TOTAL_IN_CH does
const unsigned MAX_OUT_CH = 1024;  // OUT_CH IS tiled (by PE_OC below), so this only bounds the
                                    // bias buffer and the oc_tile loop trip count, not MAC parallelism
// Total in_ch bound - exactly analogous to MAX_OUT_CH's role for out_ch:
// bounds the ifmap/weights m_axi depth and the new IC_TILE loop's trip
// count, NOT any on-chip buffer size (MAX_IN_CH above still does that).
// 1024 matches the real network's layer 13 exactly (in=1024, its 1x1 conv
// down to 256) - the largest in_ch any real Phase 1 layer needs.
const unsigned MAX_TOTAL_IN_CH = 1024;
const unsigned MAX_K      = 3;     // 1x1 layers run as k=1 (pad=0), using only window[.][0][0]

// ---- Compute parallelism - sized against the real XCK26 DSP budget (1,248
// slices) working backward from the 30 FPS target, not picked as an
// arbitrary "safe" small number. See ../RESOURCE_BUDGET.md for the full
// calculation; summary: PE_OC*TR = 256 raw parallel INT8 MACs (~20.5% of
// the DSP budget, assuming 1 DSP/MAC - no INT8 packing attempted here).
// That figure is still SHORT of the estimated throughput needed to hit 30
// FPS on a full-scale reference network - RESOURCE_BUDGET.md §3 explains
// why and what to do about it. Do not read these numbers as "tuned to meet
// spec"; read them as "64x more parallel than the previous draft, with
// headroom to go higher once Phase 0's real network size and a real
// synthesis report exist."
// Parallelism experiment (2026-07-23): PE_OC=16,TR=16 (256 raw MACs) was
// confirmed via real cosim to be bit-exact and to use only 74% LUT / 11% DSP
// after INT8 packing (RESOURCE_BUDGET.md §5) - far under budget on DSP, and
// even 100%-efficient at 256 raw MACs/cycle @ 200MHz caps out around ~22 FPS
// on the real 2.32 GMAC/frame network (roadmap_yolov3_tiny_adas.md Phase 3),
// short of the 30 FPS gate regardless of achieved efficiency. Raising PE_OC
// (even-only requirement, no power-of-2 constraint - unlike TR below) is the
// finer-grained knob to spend the remaining LUT headroom on: TR must stay a
// divisor of MAX_IN_CH=128 for a clean cyclic partition, so TR's only next
// step (32) jumps parallelism 2x outright with no intermediate points to
// check LUT scaling against first.
//
// Tried PE_OC=24 (384 raw MACs, 1.5x) first - real csynth result: LUT
// 86,986 (74%) -> 121,566 (**103%, over the XCK26's 117,120 total** - would
// not place/route), DSP 141 (11%) -> 207 (16%), FF 37,173 (15%) -> 52,265
// (22%). Also logged "Loop Constraint Status: All loop constraints were NOT
// satisfied" at this setting, likely related. LUT scaled far faster than
// DSP/FF as parallelism rose - confirms LUT (not DSP) is still the real
// ceiling even after packing. Two real data points (16->86,986;
// 24->121,566) put the marginal cost at roughly ~270 LUT per raw MAC of
// parallelism - extrapolating back to an ~85-90% LUT target lands around
// 300-325 raw MACs, i.e. PE_OC~=20 with TR=16 unchanged. Re-synthesize
// (`run_hls.bat`, no `cosim` needed yet) and check the real numbers in
// conv_engine_csynth.rpt again before trusting this extrapolation or
// pushing further.
// LUT_REDUCTION_PLAN.md Lever 4 (2026-07-24): dropped from 20 back to 16 -
// applied TOGETHER with Lever 3's loop restructuring (conv_engine.cpp's
// scan_and_compute(), see the comment above its combined MAC_REDUCE loop),
// not as a fallback tried only after Lever 3 proved insufficient - a
// deliberate choice to guarantee the LUT target even if Lever 3's real
// synthesis result undershoots its own (unverified-by-synthesis-at-the-time)
// estimate. 20->16 is a real ~20% cut in raw parallel MACs (PE_PAIRS 10->8,
// 320->256 raw MACs) and therefore a real ~20% throughput cost on top of
// whatever Lever 3 does - not free like Levers 1/2. 16 is not a new,
// unvalidated choice: it matches the earlier "packed baseline" already
// confirmed by real csynth + cosim bit-exact (see this file's own history
// above and RESOURCE_BUDGET.md §5's "Raising PE_OC after packing" table,
// 86,986 LUT/74% at the time - a different code shape than today's, since
// requant/tiling/Lever-1-3 didn't exist yet, so that exact figure will NOT
// reproduce today, but the choice of 16 as a known-safe parallelism level
// is not a guess). Re-verify with a fresh csynth run - this is a parameter
// change, not something g++ C-sim can validate for area/throughput trade-
// offs, only for continued bit-exact correctness (which it does confirm -
// PE_OC has no effect on functional correctness, only on how many out_ch
// lanes run per oc_tile pass and thus num_oc_tiles).
// Raised 16->24 (2026-07-24, same session, immediately after Lever 4's
// 20->16 cut above was confirmed by real csynth+cosim): with Lever 3's
// loop-consolidation landing at LUT 50%/DSP 12% (RESOURCE_BUDGET.md §10),
// PE_OC=16 was quantifiably leaving real throughput on the table, not just
// "conservative" - computed directly from the real 13-layer network
// (python/layers_meta.json): total (oc_tile x ic_tile) DDR re-scan passes
// across all 13 real layers went 508 (at PE_OC=20) -> 619 (at PE_OC=16),
// +21.9%, ON TOP OF the ~20% raw-MAC parallelism cut - a real, compounding
// throughput cost Lever 4 introduced as a deliberate hedge, now worth
// paying back given the LUT headroom Lever 3 actually delivered.
// 24 is not a new guess picked in a vacuum: it was tried ONCE before, in
// the OLD (pre-Lever-3) loop structure, and came back at 121,566 LUT
// (103%, would not place/route - RESOURCE_BUDGET.md §5's "Raising PE_OC
// after packing" table). That rejection does NOT automatically still
// apply here - the whole reason it failed was per-PE_PAIR_LOOP-lane
// replicated ky/kx loop-control overhead (~130-200 LUT/lane, paid
// PE_PAIRS x TR times over), and Lever 3 specifically eliminated that
// exact cost category by sharing one MAC_REDUCE instance across all lanes
// (TROUBLESHOOTING.md §21) - re-testing a previously-rejected parallelism
// level after a structural change that directly targets the reason it was
// rejected is the right call (same reasoning TROUBLESHOOTING.md §18
// documents for BIND_STORAGE), not "assuming an old number without
// re-deriving it." A rough linear estimate from the single current data
// point (PE_PAIRS=8 -> MAC_REDUCE module itself at 29,791 LUT, ~3,724
// LUT/pair average) puts PE_PAIRS=12 (this change) around ~73,500 LUT
// (~63%) - comfortably under both the 85% target and this document's own
// 60-70% "safe" line (RESOURCE_BUDGET.md §1) - but this is an ESTIMATE
// from one data point, the same kind of extrapolation that was wrong by a
// wide margin once already in this file's own history (§3's PE_OC=16->24
// linear guess, before packing, badly undershot the real LUT jump). Not
// confirmed by real synthesis at the time of this change - re-run
// `run_hls.bat` and check the real LUT/DSP/Fmax numbers before trusting
// this estimate; g++ C-sim only confirms PE_OC has no effect on
// functional correctness (which it does - see conv_engine_tb.cpp), it
// says nothing about area/timing.
//
// 24 -> 32 (2026-08-05, team decision). Everything above this paragraph
// describes the 16 -> 24 step and is kept as-is; it is history, not a
// description of the current value. What is true of 32 today:
//
//   - Functional correctness: unaffected by PE_OC by construction, and
//     re-confirmed the usual way - C-sim on 2026-08-05 printed both
//     ALL REAL LAYERS PASS (13/13 bit-exact) and ALL CONFIGS PASS at this
//     value (hls_logs/run_hls.log). Also being verified end to end by the
//     host-side full-chain run of
//     doc/plan_full_chain_e2e_verification.md.
//   - Area at PE_OC=32: **measured, not assumed** - csynth 2026-08-05
//     13:29 (conv_engine_prj/solution1/syn/report/conv_engine_csynth.rpt),
//     on the current restructured + PACK4 + URAM code:
//
//         LUT 89,914 (76%)   FF 40,982 (17%)   DSP 287 (22%)
//         BRAM_18K 8 (2%)    URAM 64 (100%)
//         estimated clock 3.650 ns vs a 5.00 ns target (uncertainty 1.35)
//
//     BRAM is no longer the constraint (53% -> 2%, the BIND_STORAGE=URAM
//     change below did that); **URAM is, at 64/64 = 100%**, deliberately
//     (see that note). LUT at 76% is an HLS estimate, and this project has
//     measured that estimator at ~2.5x pessimistic against real Vivado
//     synthesis (doc/plan_no_board_verification_and_perf.md §0.2) - so
//     treat 76% as an upper bound, not as a near-miss.
//   - Still NOT established at PE_OC=32: a real Vivado implementation
//     (place & route, real system-level utilization and WNS/WHS), and
//     cosim cycle counts at real layer shapes. The one cosim started on
//     2026-08-05 (config-A) reached 100% RTL simulation but its log ends
//     without a PASS banner - treat that run as incomplete, not as a
//     result. Do not quote an FPS or a routed-utilization figure for this
//     value until those exist.
//
// CROSS-FILE COUPLING - the thing that actually bit (2026-08-05):
// python/export_sw_headers.py has its own hand-maintained PE_OC constant
// that sizes SW/network_layers.h's MAX_ACCUM_ELEMS, which sizes
// SW/network_run_full.c's accum_buf ([out_h][out_w][PE_OC]). It was left
// at 24 when this moved to 32, undersizing that buffer by 33%
// (3,538,944 vs the required 4,718,592) - silent DRAM corruption on real
// hardware, not a crash. Resynced and the headers regenerated on
// 2026-08-05 (see that file's own PE_OC comment for the full record).
// Nothing enforces the two staying in sync: changing PE_OC here means
// editing that constant and re-running that script, every time.
const unsigned PE_OC = 32;  // output channels computed in parallel per pass
// TR=32 tried and abandoned (2026-08-04, teammate's PL branch - recorded
// here so nobody re-tries it without new evidence): raising TR was attempted
// twice (BRAM-only and with line_buf URAM-bound) and both pushed the LUT
// csynth estimate to 112% (over budget even at the estimate stage - no
// precedent in this project of a >100%-at-csynth design succeeding in real
// Vivado), even though it did relieve BRAM (47% est, down from 53.47% real).
// Pivoted instead to binding line_buf - the dominant BRAM consumer - to
// UltraRAM (BIND_STORAGE pragma in conv_engine.cpp), which relieves BRAM
// without touching the MAC/tree logic at all. Teammate's known-good with
// that binding (PE_OC=24, TR=16, pre-restructure code): LUT 65%, BRAM 2%,
// URAM 100%, DSP 219, real Vivado WNS=+1.85ns@100MHz. PE_OC (independent
// lanes, not a reduction tree) remains the structurally cheaper axis if
// more parallelism is wanted - which is the axis the 2026-08-05 decision
// took: PE_OC=32 IS the shipped value now (see the PE_OC note above; this
// sentence previously said that experiment was "NOT merged yet", which
// stopped being true on that date). The rest of that old warning stands
// unchanged and is the open item: it has NOT been re-tested on top of the
// current (restructured + PACK4 + URAM) code via fresh csynth, so its
// area/timing is assumed, not known.
const unsigned TR    = 16;  // reduction (in_ch) terms unrolled in parallel per PE_OC lane;
                            // must divide MAX_IN_CH evenly (128/16 = 8 tiles) for a clean
                            // cyclic array partition (see conv_engine.cpp)

typedef ap_int<8>  weight_t;
typedef ap_int<8>  act_t;
typedef ap_int<32> accum_t;
// Separate from weight_t: the real target network's golden model
// (python_teammate/darknet_golden, RTL_HANDOFF_KO.md section 2) ships bias
// as signed INT32 ("bias.bin", 4 bytes/channel), not INT8 - conv_engine's
// original activation math (a flat "acc>>3 then saturate", no per-layer
// scale) never needed a wider bias, but the real per-layer requantization
// below does.
typedef ap_int<32> bias_t;

// Narrowed accumulator for PE_PAIR_LOOP's per-ic_tile-pass partial sum
// (acc_lo/acc_hi/contrib_lo/contrib_hi in conv_engine.cpp) - LUT_REDUCTION_PLAN.md
// Lever 1. Deliberately NOT the same type as `accum_t` (kept at ap_int<32>
// for the DDR-resident `accum` scratch buffer and the accumulate_or_finish()/
// apply_activation() interface, both of which carry a running cross-ic_tile
// total that must stay 32-bit) - this narrower type only covers the
// single-pass reduction inside the 160x-replicated (PE_PAIRS x TR) MAC_IC/
// MAC_KY/MAC_KX hot loop, where every bit of declared width gets paid for
// 160 times over (confirmed in a real csynth report's Instance/Multiplexer
// tables: acc_lo/acc_hi's 32-bit add and their loop-exit lcssa mux/register
// were among the largest LUT line items - see conv_engine_csynth.rpt).
//
// Width derivation (28 bits, NOT the theoretical int8 x MAX_IN_CH x K x K
// worst case, which would be far more pessimistic than any real layer needs):
// python_teammate/darknet_golden/artifacts/int8/model_manifest.json's
// per-layer "accumulator_bound" field is the golden model's own documented,
// weight-conditioned worst-case bound for the FULL (untiled) per-layer
// accumulator - the largest value across all 13 real conv layers is
// 74,576,608 (layer index 12, in_ch=512->out_ch=1024, k=3). Any ic_tile's
// PARTIAL sum (a subset of that same full sum's terms) is bounded by the
// same figure, since accumulator_bound is itself a sum-of-absolute-value
// budget, not a value achieved by cancellation. 2^27 = 134,217,728 comfortably
// exceeds that (not a knife-edge fit, ~1.8x headroom) - ap_int<28> (range
// -2^27..2^27-1) is the minimum width that fits with that margin.
// Cross-checked empirically against the real per-layer .bin data
// (python/layer00..12_*.bin via layers_meta.json): actually-measured max
// |per-ic_tile partial sum, bias included| across all 13 layers on real
// weights/activations is only 264,099 (needs 20 bits) - far under the
// documented bound, confirming 28 bits is generous, not tight, for real
// data while still tracking the model's own documented worst case rather
// than overfitting to one calibration image.
typedef ap_int<28> partial_t;

// ---------------------------------------------------------------------------
// INT8 DSP packing experiment (RESOURCE_BUDGET.md §5, TROUBLESHOOTING.md):
// pack 2 MACs that share the same activation into ONE DSP48E2 multiply.
// window[ic][ky][kx] does not depend on `pe` - it's the same value read by
// every PE_OC lane each cycle - so PE_OC lanes are grouped into PE_OC/2
// pairs (2j, 2j+1); each pair's two weights are packed into one signed
// value, multiplied once by the shared activation, then split back into two
// independent products. See conv_engine.cpp's load_weight_tile() (pack) and
// scan_and_compute()'s MAC_IC/MAC_KY/MAC_KX (unpack) for the actual math.
// This is a live experiment in an isolated copy of hls/conv_engine/ - not
// yet ported back to the original project.
static_assert(PE_OC % 2 == 0, "PE_OC must be even for DSP MAC-pairing");
const unsigned PE_PAIRS = PE_OC / 2;

// { Y_hi (signed, bits[24:16]), Y_lo_biased = Y_lo XOR 0x80 (bits[7:0]) }.
// 25 bits: exact minimum is 24 (Y_hi=-128,Y_lo_biased=0 -> exactly -2^23,
// ap_int<24>'s own minimum - a knife-edge fit); this keeps 1 bit of margin
// over that edge instead of relying on it exactly. Fits the DSP48E2's 27-bit
// A port with 2 bits spare.
typedef ap_int<25> packed_weight_t;

// ---------------------------------------------------------------------------
// Top-level function.
//
// AMBA interface mapping (see README.md "AMBA Interface Map"):
//   ifmap/weights/bias : AXI4 (m_axi, bundle RD_BUS) - read from DDR
//   ofmap/accum        : AXI4 (m_axi, bundle WR_BUS) - accum is read AND
//                         written by this IP (partial-sum scratch, see
//                         below); sharing WR_BUS with ofmap costs no new
//                         physical AXI master port since nothing here runs
//                         under DATAFLOW (the double-buffering attempt that
//                         would have made concurrent-access-per-bundle a
//                         real constraint was tried and reverted - see
//                         RESOURCE_BUDGET.md section 5)
//   img_h/img_w/in_ch/out_ch/k/stride/pad and the five base addresses,
//   plus ap_start/ap_done/ap_idle : AXI4-Lite (bundle CTRL)
//
// All four array layouts are NHWC (channel innermost), matching
// conv_layer1's convention, so a layer's ofmap can be used directly as the
// next layer's ifmap without a reformatting pass.
//
// stride is accepted but only stride=1 is implemented (see README "design
// limits") - YOLOv2-tiny's conv layers are all stride 1; downsampling is a
// separate max-pool operation this engine does not perform.
//
// requant_multiplier/requant_shift/leaky_relu_enable (added for the real
// YOLOv3-tiny-ADAS golden model, python_teammate/darknet_golden -
// conv_engine_int8pack's original engine had no per-layer scale at all,
// just a fixed acc>>3): one conv_engine() call still handles exactly one
// layer, so these are per-call scalars, not a DDR-resident per-layer table -
// software is expected to read them out of the golden model's
// model_manifest.json (requant_multiplier/requant_right_shift fields) and
// program them per layer the same way it already programs k/stride/pad.
// See conv_engine.cpp's apply_activation()/round_shift() for the exact
// fixed-point formula this implements (must match RTL_HANDOFF_KO.md section
// 5 bit-for-bit, not just approximately).
//
// `accum` (added for input-channel tiling): DDR-resident scratch, NOT a
// layer input or the layer's real output - software allocates it (see
// SW/conv_engine_hw_driver.h) but never needs to initialize or read it
// itself. When in_ch <= MAX_IN_CH (a single IC_TILE pass, true for 5 of the
// real network's 13 conv layers), it is never touched at all - same
// zero-extra-DDR-traffic behavior as before this feature existed. When
// in_ch > MAX_IN_CH, conv_engine() re-scans the image once per (oc_tile,
// ic_tile) pair instead of once per oc_tile, using `accum` to carry each
// output pixel's running (pre-activation) sum across ic_tile passes for the
// PE_OC channels the CURRENT oc_tile is working on - sized
// [out_h][out_w][PE_OC], not [out_h][out_w][out_ch], since only one
// oc_tile's worth of channels needs to survive at a time (see
// conv_engine.cpp's scan_and_compute()). LeakyReLU/requantize/saturate only
// ever applies once, on the LAST ic_tile - every earlier pass writes a raw,
// unactivated INT32 partial sum, never a finished act_t value.
// ---------------------------------------------------------------------------
void conv_engine(
    // PACK4: int8x4-packed - 4 consecutive NHWC channels per 32-bit AXI
    // beat. Addresses the SAME DDR bytes the act_t* version did (see
    // pack4.h), so no PS-side repacking; `in_ch` is still in CHANNELS.
    // Layers with in_ch % 4 != 0 (only layer 0, in_ch=3) still work - see
    // READ_CH's slow path in conv_engine.cpp.
    const pack4_t  *ifmap,     // [img_h][img_w][in_ch/4], NHWC, row-major, DDR-resident
    const weight_t *weights,   // [out_ch][in_ch][k][k], row-major, DDR-resident
    // Second pointer aliasing the SAME DRAM region/address as `weights` -
    // NOT a second copy of the data. Added so LOAD_W_IC (conv_engine.cpp)
    // can read a pair's y_lo (via `weights`) and y_hi (via `weights_hi`) on
    // two independent physical AXI4 master ports in the same cycle: one
    // m_axi bundle can only issue one read per cycle (confirmed by a real
    // csynth run - achieved II=2 against a target of II=1, see
    // TROUBLESHOOTING.md), and that read was to `weights` twice per
    // iteration. Software must program this register with the exact same
    // base address as `weights` (see SW/conv_engine_hw_driver.h) - never a
    // different buffer.
    const weight_t *weights_hi,
    const bias_t   *bias,      // [out_ch], signed INT32
    act_t          *ofmap,     // [out_h][out_w][out_ch], NHWC, row-major, DDR-resident
    accum_t        *accum,     // [out_h][out_w][PE_OC] scratch - see note above, not a real layer I/O
    uint16_t img_h, uint16_t img_w, uint16_t in_ch, uint16_t out_ch,
    uint8_t  k, uint8_t stride, uint8_t pad,
    int32_t  requant_multiplier, uint8_t requant_shift, uint8_t leaky_relu_enable
);

#endif // CONV_ENGINE_H
