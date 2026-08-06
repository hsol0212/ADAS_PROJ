#include "conv_engine.h"
#include <cassert>

// Widened to ap_int<64> (was accum_t/32-bit): apply_activation()'s
// round_shift(acc * requant_multiplier, requant_shift) product can reach
// ~62 bits (27-bit accumulator_bound x 31-bit multiplier, per
// model_manifest.json) before the shift brings it back into int8 range -
// saturating on the narrower, already-truncated value would silently
// corrupt exactly the large-magnitude cases this check exists to catch.
static act_t saturate(ap_int<64> v) {
    if (v > 127)  return (act_t)127;
    if (v < -128) return (act_t)-128;
    return (act_t)v;
}

// round(x / 2^s), ties away from zero - matches RTL_HANDOFF_KO.md section 5
// exactly: `round_shift(x, s) = sign(x) * ((abs(x) + 2^(s-1)) >> s)`. NOT the
// same as a plain arithmetic `x >> s` on a biased value: two's-complement
// right shift rounds negative numbers toward -infinity, not away from zero,
// so the sign has to be split out explicitly. Used for both the LeakyReLU
// slope (13/128) and the final per-layer requantization - one definition,
// not two, specifically so a rounding bug (RESOURCE_BUDGET.md/TROUBLESHOOTING
// #8 already has one war story about a split-arithmetic bug like this) can't
// silently diverge between the two call sites.
static ap_int<64> round_shift(ap_int<64> x, ap_uint<6> s) {
#pragma HLS INLINE
    if (s == 0) return x;
    ap_int<64> half = (ap_int<64>)1 << (s - 1);
    // Explicit if/else with an explicit ap_int<64> cast on each return -
    // NOT a ternary: ap_int's natural bit-growth widens `x + half` to
    // ap_int<65> on this branch, which the compiler cannot unify with the
    // other branch's ap_int<64> inside a single `?:` expression (a real
    // build error, caught by actually compiling this against Vitis's
    // standalone ap_int.h, not by inspection).
    if (x >= 0) {
        return (ap_int<64>)((x + half) >> s);
    } else {
        return (ap_int<64>)(-(((-x) + half) >> s));
    }
}

// LeakyReLU (13/128, applied to the raw accumulator BEFORE requantization -
// RTL_HANDOFF_KO.md section 5, not the generic "0.1" a float model would use)
// + per-layer requantization (requant_multiplier/requant_shift, from the
// golden model's model_manifest.json) + int8 saturation, in that order.
//
// INLINE (2026-08-04) - this reverses an earlier, deliberate "NOT INLINE"
// decision, and the reversal is safe only because the CALLER's structure
// changed, not because the old reasoning was wrong. History: a first
// version inlined this into PE_PAIR_LOOP's UNROLL'd body and a real csynth
// run came back at 105% LUT - round_shift()'s final call shifts by
// `requant_shift`, a genuine runtime value, needing a real variable-width
// 64-bit barrel-shift mux, and the UNROLL gave it 20 separate spatial
// copies (PE_PAIRS x lo/hi), one mux tree each. Un-inlining collapsed that
// to ONE shared block time-multiplexed across the call sites.
//
// That shared block is exactly what real-layer cosim (2026-08-03/04,
// TROUBLESHOOTING.md §23-§24 + the csynth Instance table) then measured as
// the design's dominant per-position cost once the SHIFT_WINDOW fix
// landed: `accumulate_or_finish` (this function's only caller) synthesized
// as ONE NON-PIPELINED instance with Interval 10-21 cycles, and the old
// PE_PAIR_FINISH made 2*PE_PAIRS=24 back-to-back calls into it per output
// position - 240-500 cycles/position, vs ~36-78 for the whole MAC_REDUCE.
// That serialization, not LUT, is why measured conv efficiency was 4.3%.
//
// The only caller is now FINISH_WR (see scan_and_compute's phase-split
// finish loops): a single PIPELINE II=1 loop over lanes, NOT an UNROLL -
// so inlining this creates exactly ONE spatial copy of the barrel
// shift/multiply (same spatial count as the shared-instance version),
// scheduled inside the pipelined body instead of behind a 10-21-cycle
// non-pipelined function-call interval per lane. The 20-copy LUT
// explosion cannot recur because nothing UNROLLs over this function
// anymore; if a future change puts a call to this back under an UNROLL'd
// loop, re-read the history above before keeping the INLINE.
static act_t apply_activation(
    accum_t acc, bool leaky_relu_enable,
    int32_t requant_multiplier, unsigned requant_shift
) {
#pragma HLS INLINE
    accum_t post_leaky = acc;
    if (leaky_relu_enable && acc < 0) {
        post_leaky = (accum_t)round_shift((ap_int<64>)acc * 13, 7);
    }
    ap_int<64> scaled = (ap_int<64>)post_leaky * (ap_int<64>)requant_multiplier;
    return saturate(round_shift(scaled, requant_shift));
}

// Input-channel tiling (added once the real YOLOv3-tiny-ADAS network showed
// 8 of its 13 conv layers need in_ch > MAX_IN_CH - see conv_engine.h): one
// PE_PAIR's raw (pre-activation) sum for ONE output pixel, carried across
// ic_tile passes via `accum` (DDR scratch, [out_h][out_w][PE_OC] - see
// conv_engine.h's note on the top-level `accum` parameter for why PE_OC, not
// out_ch). `num_ic_tiles == 1` (true for 5 of the 13 real conv layers, and
// every conv_engine_tb.cpp config before this feature existed) takes the
// bottom branch and never touches `accum` at all - byte-for-byte the same
// behavior this function had before tiling existed.
//
// accumulate_or_finish() used to live here - the per-lane helper carrying
// one output pixel's raw sum across ic_tile passes via the DDR `accum`
// scratch (see conv_engine.h's note on the top-level `accum` parameter).
// Removed 2026-08-04: as a NON-INLINED helper it synthesized to one shared
// non-pipelined instance (Interval 10-21) that the old UNROLL'd
// PE_PAIR_FINISH called 24x back-to-back per output position (~240-500
// cycles/position, ~half of layer 9's measured per-position cost), and
// re-INLINING it into a single pipelined lane loop only achieved II~14
// because its internal first/middle/last-ic_tile branching mixed a
// conditional accum read with accum/ofmap writes in one loop body. Its
// logic now lives flattened in scan_and_compute()'s FINISH_GATHER/
// ACCUM_RD/FINISH_WR/ACCUM_WR phase loops - see the comment there for the
// full two-step history and the measurements behind it. Behavior
// (including the num_ic_tiles==1 fast path never touching `accum`) is
// preserved exactly.

// ---------------------------------------------------------------------------
// load_weight_tile / scan_and_compute are split into separate functions -
// not just for readability, but so weight-tile double buffering (see
// RESOURCE_BUDGET.md §5, tried 2026-07-23, reverted - see that section for
// why) can later wrap these two calls in #pragma HLS DATAFLOW with ping-pong
// wtile/btile buffers, without restructuring the algorithm itself. wtile/
// btile are owned by conv_engine() (the caller) and passed in, precisely so
// a future ping-pong version only has to change the caller, not these two
// functions' internals.
// ---------------------------------------------------------------------------

static void load_weight_tile(
    const weight_t *weights, const weight_t *weights_hi, const bias_t *bias,
    unsigned oc_tile, unsigned ic_lo, unsigned ic_count, unsigned in_ch, unsigned out_ch, unsigned k,
    packed_weight_t wtile_packed[PE_PAIRS][MAX_IN_CH][MAX_K][MAX_K],
    bias_t btile[PE_OC]
) {
    // Forced INLINE, not left to the tool's default heuristics: Vitis HLS
    // has a real, documented gotcha where fully-partitioned arrays passed
    // across a NON-inlined function boundary can silently lose their
    // parallel-access partitioning (the boundary implies a port-like
    // interface that doesn't always preserve per-element parallel access).
    // wtile_packed is partitioned in conv_engine() (its declaration site) for
    // exactly the parallelism the MAC reduction in scan_and_compute()
    // depends on - relying on the tool happening to auto-inline this
    // function by default would make that parallelism dependent on
    // unstated tool-version behavior instead of an explicit pragma. A
    // DATAFLOW/double-buffered version (RESOURCE_BUDGET.md §5) would
    // deliberately remove this pragma, since DATAFLOW requires its stages
    // to NOT be inlined - tried 2026-07-23, reverted (see that section):
    // DATAFLOW itself hit a hard error unrelated to inlining (ifmap/weights/
    // bias sharing one AXI bundle), so this pragma's own correctness was
    // never actually tested in that attempt.
#pragma HLS INLINE
    // INT8 DSP-packing experiment (conv_engine.h): the pack step (bias-XOR +
    // shift + add) happens HERE, once per oc_tile, deliberately NOT inside
    // scan_and_compute()'s per-cycle MAC_IC/MAC_KY/MAC_KX reduction - hoisting
    // it out is what actually halves wtile's own instance count (256 -> 128)
    // on top of halving the multiplier instance count, since this loop runs
    // once per tile rather than being replicated TR*PE_PAIRS times.
LOAD_WTILE:
    for (unsigned j = 0; j < PE_PAIRS; j++) {
        unsigned oc_lo = oc_tile * PE_OC + 2 * j;
        unsigned oc_hi = oc_tile * PE_OC + 2 * j + 1;
        bool lo_valid = oc_lo < out_ch;
        bool hi_valid = oc_hi < out_ch;
        if (lo_valid || hi_valid) {
            if (lo_valid) btile[2 * j]     = bias[oc_lo];
            if (hi_valid) btile[2 * j + 1] = bias[oc_hi];
        LOAD_W_IC:
            // Each nesting level gets its OWN LOOP_TRIPCOUNT, bounding only
            // that loop's trip count - NOT a single combined max=IN_CH*K*K on
            // the innermost loop (tried first, and wrong): Vitis HLS checks a
            // directive's max against its own internally-computed average for
            // THAT loop level, and a product-of-all-levels value on the
            // innermost loop failed that check ("Ignored invalid trip count
            // directive (MAX (= 255) < AVE (= 576))" - the tool had already
            // clipped the requested 1,152 down to 255 before even comparing).
            // Without a per-level bound, the tool assumed the full range of
            // in_ch/k's underlying types (in_ch up to 65,535, k up to 255)
            // instead of MAX_IN_CH/MAX_K, reporting an absurd ~4.26 billion
            // cycle / 21 second worst case for this one region alone.
            // Bound `ic_count` (this ic_tile's local channel count, <=
            // MAX_IN_CH), NOT `in_ch` (the layer's real total) - `ic_count`
            // is what determines how many of wtile_packed's MAX_IN_CH slots
            // this pass actually fills; the global channel index fed into
            // weights[]'s address expression is `ic_lo + ic` instead of
            // plain `ic`, to select this tile's slice of the real weight
            // array (still laid out for the FULL `in_ch`, unaffected by
            // tiling - only which slice of it a given pass reads changes).
            for (unsigned ic = 0; ic < ic_count; ic++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_IN_CH
                for (unsigned ky = 0; ky < k; ky++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_K
                    for (unsigned kx = 0; kx < k; kx++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_K
                        // Guard each half independently - reading weights[]
                        // for an out-of-range oc_hi/oc_lo would be an OOB DDR
                        // access, same reasoning as the original per-pe
                        // "if (oc < out_ch)" guard, just split per half since
                        // the multiply that consumes both halves is shared.
                        weight_t y_lo = 0, y_hi = 0;
                        if (lo_valid) {
                            unsigned w_idx_lo = ((oc_lo * in_ch + (ic_lo + ic)) * k + ky) * k + kx;
                            y_lo = weights[w_idx_lo];
                        }
                        // Read via weights_hi, NOT weights - both alias the
                        // same DRAM data, but this is a second, independent
                        // AXI4 master port (see conv_engine.h's note on this
                        // param). Reading y_lo/y_hi both from `weights` on
                        // ONE port serialized to II=2 (confirmed by csynth);
                        // splitting the port is what gets this loop to II=1.
                        if (hi_valid) {
                            unsigned w_idx_hi = ((oc_hi * in_ch + (ic_lo + ic)) * k + ky) * k + kx;
                            y_hi = weights_hi[w_idx_hi];
                        }

                        // Offset-binary bias via sign-bit flip (exact for
                        // y_lo in [-128,127]: -128->0, -1->127, 0->128,
                        // 127->255) - a pure bit op, no add/width-growth risk.
                        ap_uint<8> y_lo_biased = (ap_uint<8>)y_lo ^ (ap_uint<8>)0x80;

                        wtile_packed[j][ic][ky][kx] =
                            (((ap_int<25>)y_hi) << 16) + (ap_int<25>)y_lo_biased;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// COL_LOOP OVERLAP PoC (read_col_into): the ifmap-read half of COL_LOOP's
// body, factored into a forced-INLINE helper so the SAME read can be issued
// from two places - once before the loop (prologue, column 0) and one column
// ahead inside it (prefetch, column c+1) - each writing into a caller-chosen
// px bank. That is what lets column c's compute run while column c+1's ifmap
// read is still in flight.
//
// Body is the ORIGINAL inline READ_CH, verbatim (both the packed FAST path and
// the layer-0 SLOW path), with only mechanical renames: writes go to the
// parameter `pxdst[]` (was `px[]`), the column index is the parameter `cc`
// (was the loop var `c`), and `in_bounds`/`in_c` are recomputed here from `cc`
// + the caller's per-row `row_in_bounds`/`in_r`. The `if (in_bounds)` guards
// are kept exactly, so an out-of-range (e.g. phantom column pad_w) prefetch
// issues NO AXI transaction and just 0-fills pxdst - harmless, never consumed.
// Forced INLINE so pxdst's cyclic(TR) partition (declared at the caller)
// survives the call boundary - same reason load_weight_tile() is INLINE.
static void read_col_into(
    const pack4_t *ifmap, act_t pxdst[MAX_IN_CH],
    bool row_in_bounds, unsigned in_r, unsigned cc,
    unsigned pad, unsigned img_w, unsigned in_ch,
    unsigned ic_lo, unsigned ic_count
) {
#pragma HLS INLINE
    bool in_bounds = row_in_bounds && (cc >= pad) && (cc < pad + img_w);
    unsigned in_c = cc - pad;
    if ((in_ch % PACK4_LANES) == 0u) {
        const unsigned ic_words  = ic_count / PACK4_LANES;
        const unsigned base_word =
            (((unsigned)in_r * img_w + in_c) * in_ch + ic_lo) / PACK4_LANES;
    READ_CH_WORDS:
        for (unsigned w = 0; w < ic_words; w++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_IN_CH/PACK4_LANES
            pack4_t v = 0;
            if (in_bounds) {
                v = ifmap[base_word + w];
            }
        READ_CH_LANES:
            for (unsigned lane = 0; lane < PACK4_LANES; lane++) {
#pragma HLS UNROLL
                pxdst[w * PACK4_LANES + lane] = pack4_get(v, lane);
            }
        }
    } else {
    READ_CH_ELEMS:
        for (unsigned ic = 0; ic < ic_count; ic++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_IN_CH
            act_t val = 0;
            if (in_bounds) {
                unsigned flat =
                    ((unsigned)in_r * img_w + in_c) * in_ch + (ic_lo + ic);
                pack4_t v = ifmap[flat / PACK4_LANES];
                val = pack4_get(v, flat % PACK4_LANES);
            }
            pxdst[ic] = val;
        }
    }
}

static void scan_and_compute(
    // PACK4: ifmap is int8x4-packed (pack4.h). accum/ofmap are unchanged -
    // accum_t is already 32 bits (full beat), and ofmap's FINISH_WR is a
    // smaller candidate left for a separate change so this one can be
    // measured on its own.
    const pack4_t *ifmap, accum_t *accum, act_t *ofmap,
    unsigned img_h, unsigned img_w, unsigned ic_lo, unsigned ic_count, unsigned in_ch, unsigned out_ch,
    unsigned k, unsigned pad, unsigned oc_tile, unsigned ic_tile, unsigned num_ic_tiles,
    const packed_weight_t wtile_packed[PE_PAIRS][MAX_IN_CH][MAX_K][MAX_K],
    const bias_t btile[PE_OC],
    bool leaky_relu_enable, int32_t requant_multiplier, unsigned requant_shift
) {
    // Forced INLINE - same reasoning as load_weight_tile() above.
#pragma HLS INLINE
    const unsigned out_w = img_w + 2 * pad - k + 1; // stride=1
    const unsigned pad_h = img_h + 2u * pad;
    const unsigned pad_w = img_w + 2u * pad;

    // Line buffers / sliding window - same raster-scan structure as
    // conv_layer1.cpp, generalized to runtime in_ch/k bounded by MAX_*.
    // Both partitioned cyclic(TR) on the in_ch dimension only - see the FIX
    // comment on the partition pragmas below for why that's the dimension
    // that actually needs it. `line_buf`'s column dimension is MAX_IMG_W
    // (416) wide and indexed by COL_LOOP's `c`, a plain sequential loop
    // variable, so it must stay a real (unpartitioned) memory; see
    // RESOURCE_BUDGET.md §4, which sizes line_buf as "32 BRAMs, 416 elements
    // each" - `complete`-partitioning that dimension too (as an earlier
    // version of this file did) silently turned each of those 32 BRAMs into
    // ~416 individual muxed registers, which is what made C synthesis hang
    // in scheduling/binding instead of just costing extra area.
    //
    // TRIED AND REJECTED (real csynth run, both configs measured - see
    // RESOURCE_BUDGET.md §4): `line_buf`'s dim=2 (its MAX_K-1=2 row
    // dimension) is `complete`-partitioned even though SHIFT_WINDOW's row
    // shift (`for (ky=0; ky<k-2; ky++) line_buf[ic][ky][c] = ...`) never
    // unrolls it - on paper the same never-unrolled-dimension anti-pattern
    // documented in FIX 2 below for `window`/`wtile`. Dropping it to
    // cyclic(TR) dim=1 only (matching `window`) does shrink line_buf's own
    // Multiplexer-table entry (1,843 LUT / 32 instances -> 432 LUT / 16
    // instances), but Vitis re-implements the row addressing it frees up
    // using DSP-based address arithmetic (32 new `ama_addmuladd` instances)
    // elsewhere instead of LUT muxes - net design LUT barely moved
    // (108,294 -> 108,100, -0.18%) while DSP went 267 -> 315 (+48, +18%),
    // an unrequested DSP-budget cost for no real LUT win. Left `complete`
    // here deliberately; unlike `window`/`wtile`'s K,K fix (FIX 2), this
    // one is a measured net loss, not a free win - don't redo it without
    // re-checking the DSP column, not just the LUT column.
    // FIX (found via a real conv_engine_tb.cpp C-sim run against the real
    // network's actual layer-0 width, not by inspection): the column
    // dimension must hold `pad_w` (`img_w + 2*pad`) entries, not just
    // `img_w` - `COL_LOOP` below already documented this via its own
    // `LOOP_TRIPCOUNT max=MAX_IMG_W+2`, but the array itself was still
    // declared `[MAX_IMG_W]`, 2 short of what COL_LOOP's own stated trip
    // count implies. Never caught before because no test prior to
    // REAL_WEIGHTS_PLAN.md's real-layer suite ever actually ran img_w at
    // the true MAX_IMG_W boundary with pad=1 at the same time - the real
    // network's layer 0 (in_ch=3, img_w=512, pad=1) is the first case that
    // does (TROUBLESHOOTING.md #19's `img_w<=MAX_IMG_W` assertion failure
    // was a *different*, later symptom of this same underlying gap, hit
    // once layer 0's pre-padding fix pushed `img_w` itself past 512).
    // `+2`, not `+2*MAX_K`: matches COL_LOOP's own existing annotation and
    // every real layer's actual pad (0 or 1) - `pad` up to `MAX_K` (3) is
    // permitted by the separate `assert(pad <= MAX_K ...)` sanity check
    // below but is not real-network usage; revisit this `+2` together with
    // that assert if a real layer ever legitimately needs pad>1.
    static act_t line_buf[MAX_IN_CH][MAX_K - 1][MAX_IMG_W + 2];
    static act_t window[MAX_IN_CH][MAX_K][MAX_K];
    // FIX (found from a real Vitis HLS 2021.1 run, not by inspection):
    // `complete dim=0` was originally written assuming SHIFT_WINDOW's `ic`
    // loop would be fully unrolled (giving compile-time-constant per-lane
    // indices into these arrays). It isn't - `in_ch` is a runtime
    // parameter, and Vitis HLS cannot fully unroll a variable-trip-count
    // loop (confirmed by an actual synthesis run: "WARNING: [HLS 200-936]
    // Cannot unroll loop 'SHIFT_WINDOW' ... cannot completely unroll a loop
    // with a variable trip count"). The unroll pragma on SHIFT_WINDOW below
    // is now `factor=TR` (partial, confirmed working - see MAC_IC further
    // down, which already used this correctly), so the matching partition
    // here is cyclic(TR) on the in_ch dimension, not complete - a complete
    // partition accessed through the resulting non-constant index is
    // exactly what the same run flagged as "may result in long runtime and
    // suboptimal QoR due to large multiplexers" for this array.
    //
    // FIX 2 (found from the csynth report after C synthesis actually
    // completed, not from a tool warning): a previous version of this file
    // also had `complete dim=2`/`dim=3` here (window's two K,K dims), on the
    // assumption that MAC_KY/MAC_KX's ky/kx loops would be unrolled and need
    // the parallel access. They aren't - MAC_KY/MAC_KX (see the compute loop
    // below) are `#pragma HLS PIPELINE II=1`, not UNROLL, same as
    // SHIFT_WINDOW's and load_weight_tile's ky/kx loops - so nothing ever
    // reads or writes window (or wtile, see its declaration in conv_engine()
    // below) through a compile-time-constant ky/kx index. That made the
    // complete partitioning pure overhead: the csynth report's Multiplexer
    // detail table showed window split into 144 separate memory instances
    // (16 cyclic in_ch banks x 9 K,K positions), each paying a fixed
    // ~3 address0/ce0/we0 arbitration signals x ~9 LUT for sharing that
    // instance's port between SHIFT_WINDOW's writer and MAC_KY/MAC_KX's
    // reader - 17,713 LUT total for window alone, on top of 60,264 LUT for
    // the analogous problem on wtile (2,304 instances there, since it also
    // has a `complete`-partitioned pe dimension) - together over 40% of the
    // whole design's LUT count and the reason total LUT utilization was 155%
    // of the xck26's budget. Dropping the K,K partitioning cuts window to 16
    // instances and wtile to 256, each now absorbing the ky/kx addressing
    // internally instead of paying per-position port-arbitration overhead.
#pragma HLS ARRAY_PARTITION variable=line_buf cyclic factor=TR dim=1
#pragma HLS ARRAY_PARTITION variable=line_buf complete dim=2
// BRAM->UltraRAM pivot (2026-08-04, merged from teammate's PL branch):
// line_buf is the dominant BRAM_18K consumer (128 of ~136 total blocks in
// the csynth report before this change) and UltraRAM sat at 0% used on this
// device - moving this specific array there relieves BRAM without touching
// any MAC/tree logic (unlike the abandoned TR=32 attempt, see conv_engine.h),
// so functional correctness is unaffected (BIND_STORAGE only changes which
// physical memory primitive implements the array, never its C-level
// behavior - csim results must be identical to before this pragma).
// Teammate's real measurements with this binding (on the pre-restructure
// code, PE_OC=24): LUT 65%, BRAM 53%->2%, URAM 100%, DSP 219, real Vivado
// WNS=+1.85ns@100MHz. RAM_1P is a starting point, NOT confirmed against
// this array's access pattern under the CURRENT loop structure - if csynth
// reports a port conflict or a new II violation on the loops that access
// line_buf (WINDOW_TAIL_FILL reads it; LINE_BUF_ROW_SHIFT reads AND writes
// it in the same II=1 iteration; LINE_BUF_TAIL_FILL writes it), that means
// the array genuinely needs simultaneous independent read+write and RAM_1P
// is the wrong choice - switch to RAM_2P (or RAM_T2P) and re-check, rather
// than assuming RAM_1P is correct just because it compiles.
// TIER C1 REVERSAL (2026-08-05, TIMING_CLOSURE_PLAN.md §4 Tier C / §3 원인 C).
// impl=URAM -> impl=BRAM. The comment above describes why URAM was chosen,
// and that reasoning was correct WHEN IT WAS MADE - BRAM was the binding
// constraint at 53% and URAM sat unused at 0%. That balance has since
// inverted completely:
//
//     URAM  64 / 64  = 100.00%   <- saturated
//     BRAM   5 / 144 =   3.47%   <- nearly empty
//
// and nothing else in the design now wants BRAM. Meanwhile the 200 MHz
// routed run (WNS -0.021) shows CLOCK SKEW alone at -0.207 ns - ten times
// the shortfall - because using every URAM column forces conv_engine_0 to
// span clock regions (SLICE_X27~X53, Y108~Y168 in the placed design). URAM
// columns are few and fixed in position; BRAM tiles are 144 and far more
// evenly distributed, so this should let the placer keep the engine local.
//
// This targets a different term than B1/B2 do: they attack the 4.728 ns data
// path, this attacks the -0.207 ns skew and the 62.6% route share.
//
// What this costs: BRAM returns to ~53%. That only matters if something
// later needs BRAM - the obvious candidate, TR=32, is already blocked twice
// over on LUT (see conv_engine.h), so the loss is theoretical today.
//
// WHAT MUST BE RE-CHECKED, NOT ASSUMED (the note above already warns about
// exactly this for RAM_1P): BRAM and URAM differ in read latency, and this
// array is read AND written inside II=1 loops (WINDOW_TAIL_FILL reads it;
// the fused FUSED_SHIFT_STEP loop both reads and writes it in one
// iteration). If csynth reports an II violation on those loops, or cosim
// cycles jump by more than the ~3% B1 cost, this change is wrong and must be
// reverted rather than worked around. Confirm II=1 in the csynth loop table
// and compare layer-3 cosim against the 1,999,021-cycle post-B1/B2 baseline.
#pragma HLS BIND_STORAGE variable=line_buf type=RAM_1P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=window cyclic factor=TR dim=1

    // COL_LOOP OVERLAP PoC: px promoted from a per-column local to a 2-bank
    // ping-pong buffer living across columns, so column c's compute reads
    // pxbuf[c&1] while column c+1's read fills pxbuf[(c+1)&1]. Same cyclic(TR)
    // channel partition the old per-iteration px had - here on dim=2, since
    // dim=1 is the 2-entry ping-pong axis.
    static act_t pxbuf[2][MAX_IN_CH];
#pragma HLS ARRAY_PARTITION variable=pxbuf cyclic factor=TR dim=2

ROW_LOOP:
    for (unsigned r = 0; r < pad_h; r++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_IMG_W+2
        // in_bounds/in_r for THIS row (constant across columns), plus the
        // prologue read of column 0 into bank 0 - so iteration c=0 finds its
        // data already staged, exactly as every later column finds the data
        // its previous iteration prefetched. (When pad>=1, column 0 is a
        // padding column: read_col_into 0-fills it, matching the old c=0.)
        const bool row_in_bounds = (r >= pad) && (r < pad + img_h);
        const unsigned in_r = r - pad;
        read_col_into(ifmap, pxbuf[0], row_in_bounds, in_r, 0u,
                      pad, img_w, in_ch, ic_lo, ic_count);
    COL_LOOP:
        for (unsigned c = 0; c < pad_w; c++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_IMG_W+2
            // Deliberately NOT pipelined at this level - see
            // RESOURCE_BUDGET.md and the MAC_IC loop below for why: the
            // parallelism knob lives on the reduction loop (UNROLL factor=TR
            // there), not here. Pipelining COL_LOOP would force the tool to
            // fully unroll everything nested inside it, defeating the
            // bounded-DSP design this engine exists to provide.

            // COL_LOOP OVERLAP PoC: this column's pixels were already staged
            // into pxbuf[c&1] by the previous iteration's prefetch (or, for
            // c=0, the per-row prologue above). Kick off the NEXT column's read
            // into the OTHER bank now - independent of the compute below
            // (different bank => no data hazard), so the scheduler may keep
            // this ifmap read in flight while window-update and MAC run.
            const unsigned cur = c & 1u;
            const unsigned nxt = (c + 1u) & 1u;
            read_col_into(ifmap, pxbuf[nxt], row_in_bounds, in_r, c + 1u,
                          pad, img_w, in_ch, ic_lo, ic_count);
#if 0  // ---- ORIGINAL inline read, now performed by read_col_into() above.
       //      Kept here (compiled out) so the before/after diff is easy to
       //      study: the helper body IS this block, with px->pxdst, c->cc. ----
            bool in_bounds = (r >= pad) && (r < pad + img_h) &&
                              (c >= pad) && (c < pad + img_w);
            unsigned in_r = r - pad;
            unsigned in_c = c - pad;

            act_t px[MAX_IN_CH];
#pragma HLS ARRAY_PARTITION variable=px cyclic factor=TR dim=1

        READ_CH:
            // Regression fix (found in self-review): conv_layer1.cpp's
            // original code uses if/else here, not a ternary, and that
            // matters in synthesized hardware even though it's equivalent
            // in C-simulation. A ternary typically synthesizes as a mux
            // applied AFTER both sides are computed - which for this read
            // means the m_axi address expression `in_r*img_w+in_c` (built
            // from in_r/in_c that underflow to huge unsigned values when
            // r/c < pad) could still be issued as an actual AXI read every
            // cycle, just with its result discarded, instead of the read
            // being skipped entirely. An if/else makes the read
            // conditionally EXECUTED, not just conditionally SELECTED -
            // required so padding positions never issue a garbage-address
            // bus transaction.
            //
            // FIX (found from a real Vitis HLS 2021.1 run): this loop was
            // `#pragma HLS UNROLL` (full), which both (a) failed outright -
            // in_ch is a runtime parameter, and Vitis HLS cannot fully
            // unroll a variable-trip-count loop - and (b) even if it had
            // worked, would have been the wrong choice for a loop reading
            // from an m_axi port: LOAD_WTILE's read of weights[] (which
            // DOES burst-coalesce, confirmed by the same run: "Multiple
            // burst reads ... has been inferred on port 'RD_BUS'") is
            // structured as a plain sequential PIPELINE, not an unroll -
            // that is the pattern the tool's burst inference is built to
            // recognize. Matching that pattern here fixes both problems
            // with one change.
            //
            // TRIED AND REJECTED (2026-08-03, real csynth + layer-9 cosim,
            // both measured): the `if (in_bounds)` test below is what stops
            // Vitis inferring a burst on this loop - `burst.xml` reports
            // ifmap/READ_CH under `AccessInCondBranchMissed` ("Access load
            // is in the conditional branch"), the same reason `bias` misses
            // in LOAD_WTILE. `in_bounds` depends only on r/c/pad/img_h/img_w
            // and never on `ic`, so hoisting it out of this loop (splitting
            // into an unconditional READ_CH plus a ZERO_CH zero-fill loop)
            // is bit-exact - verified against all 12 configs and all 13 real
            // layers - and it DOES make the burst pass
            // (`BurstInferredPassed`, bundle RD_BUS, loop READ_CH).
            //
            // It is still not worth doing. Real layer-9 cosim went
            // 1,291,969 -> 1,293,121 cycles, i.e. +1,152 - EXACTLY the
            // number of COL_LOOP scan positions in that run
            // (2 oc_tiles x 4 ic_tiles x 9 x 16), one extra cycle per
            // position for the hoisted branch, for zero measured gain.
            // The reason the burst buys nothing here: this loop already
            // achieves II=1, and the m_axi ports are 32-bit while `act_t`
            // is 8-bit, so `burst.xml` also reports
            // `GreaterOrEqualThresholdMissed` - the burst cannot be WIDENED
            // (`m_axi_max_widen_bitwidth` defaults to 0, and run_hls.tcl
            // sets no `config_interface` at all). An un-widened burst still
            // moves one int8 per beat, which is what II=1 was already doing;
            // bursting only saves AR-handshake overhead, not data beats.
            //
            // Do not redo this on the burst-inference argument alone. It
            // only becomes interesting TOGETHER with
            // `config_interface -m_axi_max_widen_bitwidth 32` (32, not more:
            // vivado/create_system_bd.tcl §3 deliberately narrows HP0/HP1 to
            // 32-bit to match every engine's m_axi ports, so a wider HLS
            // port would be a real system-level change, not a pragma). Even
            // then the ceiling is small: the csynth loop table puts READ_CH
            // at 12~139 cycles against SHIFT_WINDOW's 49~3554 in the same
            // COL_LOOP body, so the whole ifmap-read path is a minority of
            // real runtime.
            //
            // Bound by `ic_count` (this ic_tile's local channel count), not
            // `in_ch` - but the m_axi address expression still uses `in_ch`
            // (the real total) as the NHWC channel stride and `ic_lo + ic`
            // as the channel offset, since ifmap's actual DRAM layout is
            // unaffected by tiling - only which channel slice a given
            // ic_tile pass reads changes.
            //
            // PACK4 (this variant): `ifmap` is now pack4_t*, carrying 4
            // consecutive NHWC channels per AXI beat. This loop is the
            // reason the whole variant exists - see pack4.h for the csynth
            // table showing READ_CH is now the LARGEST term in COL_LOOP
            // (139 of 411 cycles), not the minority the stale comment above
            // still claims.
            //
            // Two paths, because layer 0 is a genuine exception:
            //   FAST  in_ch % 4 == 0 - one beat per 4 channels. This is
            //         every real layer except layer 0 (in_ch is
            //         16/32/64/128/256/384/512/1024).
            //   SLOW  in_ch % 4 != 0 - one channel per beat, extracted from
            //         its containing word. Only layer 0 (in_ch=3) takes
            //         this, and its READ_CH trip count is 3, so the lost
            //         speedup there is worth nothing anyway.
            //
            // Keeping the slow path is what preserves the "no PS-side data
            // change" property that makes this technique cheap: layer 0's
            // RGB input does NOT have to be re-laid-out as RGBX in DDR, and
            // its weights do not need a zero 4th input channel.
            if ((in_ch % PACK4_LANES) == 0u) {
                // ic_count is a multiple of 4 whenever in_ch is: ic_lo is a
                // multiple of IC_TILE (128) and ic_count is
                // min(IC_TILE, in_ch - ic_lo). Asserted in conv_engine().
                const unsigned ic_words  = ic_count / PACK4_LANES;
                const unsigned base_word =
                    (((unsigned)in_r * img_w + in_c) * in_ch + ic_lo) / PACK4_LANES;
            READ_CH_WORDS:
                for (unsigned w = 0; w < ic_words; w++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_IN_CH/PACK4_LANES
                    // Same "conditionally EXECUTED, not conditionally
                    // selected" property the original if/else had: an
                    // out-of-bounds position issues no AXI transaction at
                    // all. v stays 0, so all 4 lanes unpack to 0 - exactly
                    // the `px[ic] = 0` the else-branch used to write.
                    pack4_t v = 0;
                    if (in_bounds) {
                        v = ifmap[base_word + w];
                    }
                READ_CH_LANES:
                    for (unsigned lane = 0; lane < PACK4_LANES; lane++) {
#pragma HLS UNROLL
                        // 4 writes per cycle into px. Safe at II=1 because
                        // px is ARRAY_PARTITION cyclic factor=TR (TR=16),
                        // so these 4 consecutive indices always land in 4
                        // distinct banks.
                        px[w * PACK4_LANES + lane] = pack4_get(v, lane);
                    }
                }
            } else {
            READ_CH_ELEMS:
                for (unsigned ic = 0; ic < ic_count; ic++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_IN_CH
                    act_t val = 0;
                    if (in_bounds) {
                        unsigned flat =
                            ((unsigned)in_r * img_w + in_c) * in_ch + (ic_lo + ic);
                        pack4_t v = ifmap[flat / PACK4_LANES];
                        val = pack4_get(v, flat % PACK4_LANES);
                    }
                    px[ic] = val;
                }
            }
#endif  // ---- end ORIGINAL inline read (compiled out; see read_col_into) ----

        SHIFT_WINDOW:
            // FIX (found from a real Vitis HLS 2021.1 run): `#pragma HLS
            // UNROLL` (full) cannot unroll a loop with a runtime trip count
            // (in_ch), confirmed by that synthesis run's "WARNING: [HLS
            // 200-936] Cannot unroll loop 'SHIFT_WINDOW' ... variable trip
            // count" - a partial unroll (factor=TR) on the ic dimension is
            // required for the same reason it is on MAC_IC/MAC_TR below.
            // window/line_buf are cyclic(TR)-partitioned on the ic dimension
            // (not complete) to match, unchanged by the 2026-08-03
            // restructure below - only WHICH loop level carries the
            // `UNROLL factor=TR` changed (now the innermost lane loop, not
            // the outermost one), not the partitioning itself.
            //
            // Each nested loop below gets its OWN LOOP_TRIPCOUNT (same
            // per-level rule as LOAD_W_IC above, TROUBLESHOOTING.md #4):
            // without one, `hls_compile.log` showed the tool inferring its
            // own max=2 bound for the ky/kx loops (from static range
            // analysis of k-1 <= MAX_K-1) and then rejecting it against an
            // internally-computed average of 127 for that loop level -
            // "Ignored invalid trip count directive (MAX (= 2) < AVE (=
            // 127))" - falling back to an unbounded default and inflating
            // OC_TILE/ROW_LOOP/COL_LOOP's reported worst-case latency into
            // the quadrillions of cycles.
            // Bound by `ic_count`, not `in_ch` - see READ_CH above for why.
            //
            // RESTRUCTURED (2026-08-03, TROUBLESHOOTING.md §23-e, real
            // csynth loop table + ablation, not just inspection): the
            // original version of this loop had `ic` as the OUTERMOST
            // dimension (`UNROLL factor=TR`) wrapping these sequential ky/kx
            // loops - which replicates the ky/kx loop control TR-way, one
            // full copy per unrolled lane. That's the exact anti-pattern
            // §21/LUT_REDUCTION_PLAN.md Lever 3 already diagnosed and fixed
            // for MAC_KY/MAC_KX below, just never applied here: a real
            // csynth run measured this block (SHIFT_WINDOW) at 3,464 of
            // COL_LOOP's 4,328 worst-case cycles (80%), and an ablation
            // (deleting these loops, re-synthesizing) confirmed the nested
            // ky/kx structure itself - not the memory accesses - as the
            // cost: SHIFT_WINDOW's iteration latency dropped 433 -> 97 with
            // the loops gone.
            //
            // Fixed the same way Lever 3 fixed MAC_KY/MAC_KX: flip the nest
            // so the sequential dimension (ky, kx) is OUTERMOST and
            // PIPELINE'd, with the TR-wide lane dimension (`ic_step`/`t`,
            // same split MAC_REDUCE/MAC_TR below already use) UNROLLed
            // INSIDE the pipelined body. Unlike MAC_REDUCE's accumulator
            // (RESOURCE_BUDGET.md §10 - TR lanes all writing into the SAME
            // loop-carried acc_lo/acc_hi forced a real 16-deep sequential
            // chain and blew the timing budget 4x), every lane here writes
            // an independent `window[ic]`/`line_buf[ic]` slot - cyclic(TR)-
            // partitioned, `ic = ic_step*TR+t` lands each lane `t` on a
            // distinct physical partition, identical indexing to MAC_TR's
            // own `ic = ic_step*TR+t` - so there is no shared write target
            // and no reduction, hence no reason to expect that specific
            // failure mode here. Still re-check csynth's estimated clock
            // regardless (§21's own warning: a cycle/LUT win can hide a
            // timing loss) rather than assuming this reasoning is enough.
            //
            // SHIFT-LOOP FUSION (2026-08-05, this fork - see
            // doc/plan_datapath_efficiency.md §1). This block previously WAS
            // 4 separate pipelined loop nests (row-shift, tail-fill,
            // line_buf-shift, line_buf-tail). That split was deliberate and
            // its reasoning is preserved below, because it is what makes the
            // merge provably safe rather than merely plausible.
            //
            // WHY THE SPLIT COST SO MUCH. All four nests walk the SAME
            // `ic_step` axis (TR channels per cycle, because `window`/
            // `line_buf` are cyclic(TR)-partitioned on the channel dim), and
            // each paid its own pipeline entry latency. §29's fitted
            // formulas, which reproduce the csynth min~max exactly:
            //
            //   WINDOW_ROW/COL_SHIFT  3 + k(k-1)*ceil(ic/TR) = 51  (48 iters)
            //   WINDOW_TAIL_FILL      3 + k*ceil(ic/TR)      = 27  (24 iters)
            //   LINE_BUF_ROW_SHIFT    2 + (k-1)*ceil(ic/TR)  = 18  (16 iters)
            //   LINE_BUF_TAIL_FILL    2 + ceil(ic/TR)        = 10  ( 8 iters)
            //                                          total = 106 cycles
            //
            // 106 of COL_LOOP's 413-cycle worst case - 25.7%, more than
            // MAC_REDUCE's own 78. But the per-channel work is only ~12
            // register moves; `k` is at most MAX_K=3, so ky/kx can simply be
            // UNROLL'd into ONE ic_step loop: ~8 iterations + one entry
            // instead of 96 + four.
            //
            // WHY MERGING IS SAFE. The old comment justified the split by
            // noting that ALL row-shift writes must precede ANY tail-fill
            // write "across every `ic`, not just within one". Re-reading the
            // accesses, that cross-`ic` requirement does not actually exist:
            // every access here - window[ic][..], line_buf[ic][..], px[ic] -
            // is indexed by the SAME `ic` as the lane performing it. There is
            // no cross-channel read/write pair anywhere in these four pieces,
            // so per-lane ordering is sufficient, and this loop preserves it:
            //
            //   (a) snapshot line_buf's OLD column-c values into lb_old
            //   (b) window column shift, ASCENDING kx (so window[..][kx+1] is
            //       still the original value when read - same as the old
            //       nest's ascending kx loop)
            //   (c) window tail fill, reading lb_old (not line_buf), so it
            //       cannot see (d)'s writes
            //   (d) line_buf row shift + tail fill, last
            //
            // (a) exists specifically so (c)'s "reads line_buf's OLD rows"
            // guarantee survives even if Vitis schedules (d) early: the value
            // was already captured. That is strictly stronger than relying on
            // C-sim's top-to-bottom order, which is what the split version
            // depended on.
            //
            // NOT SYNTHESIS-CONFIRMED AT THE TIME THIS WAS WRITTEN - the same
            // caveat the split version carried, and for the same reason. What
            // IS confirmed: g++-against-standalone-ap_int.h C-sim, all 12
            // configs + all 13 real layers, bit-exact against the split
            // version's own golden results. Before trusting this delivered a
            // real win: re-run csynth, confirm COL_LOOP's max drops from 413
            // toward ~320 AND that the estimated clock is still under the
            // 5.00 ns target (§21's warning: a cycle/LUT win can hide a
            // timing loss - and this loop now does ~192 register moves per
            // cycle instead of ~16, which is exactly the kind of change that
            // buys cycles with routing pressure). Then re-run a real-layer
            // cosim and compare against the pre-fusion baseline.
            const unsigned ic_steps_sw = (ic_count + TR - 1) / TR;

            // The four former nests, fused. `ky`/`kx` are bounded by MAX_K=3
            // so they UNROLL into the body; only `ic_step` remains as a real
            // loop, which is the axis the arrays are partitioned on.
            //
            // Every runtime-`k` guard below is written to avoid UNSIGNED
            // UNDERFLOW rather than relying on a signed comparison - the
            // exact class of bug the old LINE_BUF_ROW_SHIFT comment records
            // (`k - 2` on unsigned k=1 wrapping to a huge value, which config
            // C in conv_engine_tb.cpp exercises for real). Hence `kx + 1 < k`
            // instead of `kx < k - 1`, and `ky + 2 < k` instead of
            // `ky < k - 2`. The `k >= 2` guard on the line_buf half is kept
            // for the same reason it was added: a k=1 layer needs zero
            // line-buffer rows, so the whole block must be skipped, not just
            // its final write.
        FUSED_SHIFT_STEP:
            for (unsigned ic_step = 0; ic_step < ic_steps_sw; ic_step++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_IN_CH/TR
            FUSED_SHIFT_LANE:
                for (unsigned t = 0; t < TR; t++) {
#pragma HLS UNROLL
                    unsigned ic = ic_step * TR + t;
                    if (ic < ic_count) {

                        // (a) Snapshot line_buf's OLD column-c values BEFORE
                        // anything writes line_buf. MAX_K-1 = 2 elements of
                        // act_t; small enough that no partition pragma is
                        // needed (constant indices under UNROLL scalarize it).
                        act_t lb_old[MAX_K - 1];
                    FUSED_LB_SNAPSHOT:
                        for (unsigned ky = 0; ky + 1 < MAX_K; ky++) {
#pragma HLS UNROLL
                            lb_old[ky] = (ky + 1 < k) ? line_buf[ic][ky][c] : (act_t)0;
                        }

                        // (b) Window column shift, ASCENDING kx so each read
                        // of [kx+1] still sees the original value.
                    FUSED_WIN_SHIFT:
                        for (unsigned ky = 0; ky < MAX_K; ky++) {
#pragma HLS UNROLL
                            for (unsigned kx = 0; kx + 1 < MAX_K; kx++) {
#pragma HLS UNROLL
                                if (ky < k && kx + 1 < k) {
                                    window[ic][ky][kx] = window[ic][ky][kx + 1];
                                }
                            }
                        }

                        // (c) Window tail fill. Reads lb_old, NOT line_buf, so
                        // it cannot observe (d)'s writes no matter how Vitis
                        // schedules them.
                    FUSED_WIN_TAIL:
                        for (unsigned ky = 0; ky < MAX_K; ky++) {
#pragma HLS UNROLL
                            if (ky < k) {
                                window[ic][ky][k - 1] =
                                    (ky + 1 < k) ? lb_old[ky] : pxbuf[cur][ic];
                            }
                        }

                        // (d) line_buf row shift + tail fill, last.
                        if (k >= 2) {
                        FUSED_LB_SHIFT:
                            for (unsigned ky = 0; ky + 2 < MAX_K; ky++) {
#pragma HLS UNROLL
                                if (ky + 2 < k) {
                                    line_buf[ic][ky][c] = line_buf[ic][ky + 1][c];
                                }
                            }
                            line_buf[ic][k - 2][c] = pxbuf[cur][ic];
                        }
                    }
                }
            }

            if (r >= k - 1 && c >= k - 1) {
                unsigned out_r = r - (k - 1);
                unsigned out_c = c - (k - 1);

            // INT8 DSP-packing experiment (conv_engine.h): PE_LOOP over
            // individual pe replaced with PE_PAIR_LOOP over PE_PAIRS - each
            // pair (2j, 2j+1) shares ONE multiply per (ic,ky,kx) step against
            // the common activation window[ic][ky][kx], since that value
            // never depended on pe in the first place.
            //
            // LUT_REDUCTION_PLAN.md Lever 3 (2026-07-24): this whole block
            // was rewritten from "PE_PAIR_LOOP(UNROLL) -> MAC_IC(UNROLL
            // factor=TR) -> MAC_KY(seq) -> MAC_KX(seq,PIPELINE II=1)" to
            // "MAC_REDUCE(seq,PIPELINE II=1, over ic_step*ky*kx) ->
            // PE_PAIR_LOOP(UNROLL) -> MAC_TR(UNROLL)". Reasoning: nesting
            // ky/kx (a genuinely sequential, pipelined dimension - MAC_KY/
            // MAC_KX were always PIPELINE, never UNROLL) INSIDE two UNROLLed
            // dimensions (PE_PAIRS x TR = 160-way at PE_OC=20) meant Vitis
            // HLS synthesized 160 separate, fully independent copies of the
            // ky/kx loop's own control logic (FSM state, trip counter,
            // address arithmetic) - a real csynth report's Instance table
            // showed each of those 160 `MAC_KY_MAC_KX` copies at 410 LUT
            // (398 after Lever 1), with roughly 130 of that being pure loop
            // bookkeeping rather than MAC/unpack arithmetic (~65,600 LUT
            // total, 61% of the whole design - LUT_REDUCTION_PLAN.md §1-a).
            // Flipping the nest so the sequential (ic_step, ky, kx) loop is
            // OUTERMOST and PIPELINE'd, with PE_PAIRS/TR's spatial
            // parallelism UNROLLed INSIDE its body instead of wrapping it,
            // gives Vitis one shared FSM/counter/address-generator for the
            // whole datapath instead of one per lane - same total MAC/
            // unpack operation count and same per-cycle spatial parallelism
            // as before, just described so the tool only pays for loop
            // control once.
            //
            // NOT SYNTHESIS-CONFIRMED AT THE TIME THIS WAS WRITTEN - only
            // g++-against-standalone-ap_int.h C-sim (conv_engine_tb.cpp, all
            // configs + all 13 real layers, bit-exact) has verified this is
            // functionally identical to the version it replaces. Loop
            // restructuring is the single riskiest class of change in this
            // file's own history (see e.g. the weight-tile-double-buffering
            // DATAFLOW attempt, RESOURCE_BUDGET.md §5, which failed outright
            // before even reaching the question it set out to answer) -
            // re-run `run_hls.bat`/`run_hls.sh` and check
            // conv_engine_csynth.rpt's Instance table for how many
            // `MAC_KY_MAC_KX`-equivalent copies actually got created (should
            // be 1, not PE_PAIRS x TR) before trusting this delivered the
            // LUT win it was written for.
            //
            // acc_lo/acc_hi are now PE_PAIRS-element arrays, not per-j
            // locals - they must survive across the ENTIRE combined
            // MAC_REDUCE loop below (all ic_step/ky/kx iterations), not just
            // one PE_PAIR_LOOP unroll instance, since PE_PAIR_LOOP is now
            // nested INSIDE MAC_REDUCE instead of wrapping it. `complete`-
            // partitioned so every PE_PAIRS-way UNROLLed access below is to
            // an independent register, matching wtile_packed's dim=1 (pair
            // index) partition in conv_engine() below.
            partial_t acc_lo[PE_PAIRS];
            partial_t acc_hi[PE_PAIRS];
#pragma HLS ARRAY_PARTITION variable=acc_lo complete dim=0
#pragma HLS ARRAY_PARTITION variable=acc_hi complete dim=0

            // Bias folded in exactly once per output pixel/channel, on the
            // FIRST ic_tile pass - same rule as before (see accumulate_or_
            // finish()'s own comment), just hoisted into its own small
            // UNROLLed init loop now that PE_PAIR_LOOP no longer wraps the
            // whole per-pair computation.
        PE_PAIR_INIT:
            for (unsigned j = 0; j < PE_PAIRS; j++) {
#pragma HLS UNROLL
                acc_lo[j] = (ic_tile == 0) ? (partial_t)btile[2 * j] : (partial_t)0;
                acc_hi[j] = (ic_tile == 0) ? (partial_t)btile[2 * j + 1] : (partial_t)0;
                // LUT_REDUCTION_PLAN.md Lever 2 (kept from the pre-Lever-3
                // version - see RESOURCE_BUDGET.md §8 for its measured
                // effect, DSP +4 only, smaller than hoped but harmless).
#pragma HLS BIND_OP variable=acc_lo op=add impl=dsp
#pragma HLS BIND_OP variable=acc_hi op=add impl=dsp
            }

            // Combined (ic_step, ky, kx) reduction - ONE shared PIPELINE
            // loop instead of PE_PAIRS x TR independent copies, see this
            // block's top comment. `ic_step` replaces the old MAC_IC loop's
            // UNROLL-factor=TR trip count: steps through ic_count in chunks
            // of TR, same total iteration count as before
            // (ceil(ic_count/TR)), just an explicit sequential variable
            // instead of being implicit in a partial-UNROLL's remainder
            // handling.
            const unsigned ic_steps = (ic_count + TR - 1) / TR;
        MAC_REDUCE:
            for (unsigned ic_step = 0; ic_step < ic_steps; ic_step++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_IN_CH/TR
            MAC_KY:
                for (unsigned ky = 0; ky < k; ky++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_K
                MAC_KX:
                    for (unsigned kx = 0; kx < k; kx++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_K
                    PE_PAIR_LOOP:
                        for (unsigned j = 0; j < PE_PAIRS; j++) {
#pragma HLS UNROLL
                            // RESOURCE_BUDGET.md §10 (2026-07-24): a first
                            // version of this block wrote `acc_lo[j] +=
                            // contrib_lo;` directly inside the UNROLLed
                            // MAC_TR loop below - functionally correct
                            // (bit-exact in g++ C-sim) but a real csynth run
                            // came back with the estimated clock period at
                            // 21.231 ns against a 5.00 ns target (~4x over,
                            // Fmax ~47 MHz vs the ~263 MHz this design ran
                            // at before Lever 3). `vitis_hls.log`'s own
                            // critical-path dump showed why: TR (16)
                            // UNROLLed lanes all writing `+=` into the SAME
                            // loop-carried `acc_hi[j]`, each ALSO gated by
                            // `if (ic < ic_count)`, forced a genuine 16-deep
                            // SEQUENTIAL chain of (add, select) pairs, all
                            // inside one PIPELINE II=1 iteration's
                            // combinational logic - not the balanced,
                            // multi-cycle-friendly tree HLS's own automatic
                            // reduction-variable handling would have built
                            // if TR had stayed a genuinely UNROLLed OUTER
                            // loop (the pre-Lever-3 structure) instead of an
                            // UNROLLed loop feeding a single per-cycle
                            // accumulate. Fix: compute all TR contributions
                            // into independent temporaries first (each
                            // lane's own `if` selects between its computed
                            // value and 0, in PARALLEL, not chained), then
                            // combine them with an explicit balanced binary
                            // tree (log2(TR)=4 levels of adds), and only
                            // THEN do ONE add into the loop-carried
                            // `acc_lo[j]`/`acc_hi[j]` per cycle - matching
                            // the "one add per cycle into the persistent
                            // accumulator" cadence the pre-Lever-3 design
                            // always had, just now combining TR spatial
                            // contributions instead of 1.
                            partial_t lo0[TR], hi0[TR];
#pragma HLS ARRAY_PARTITION variable=lo0 complete dim=0
#pragma HLS ARRAY_PARTITION variable=hi0 complete dim=0
                        MAC_TR:
                            // t is the TR-way spatial lane within this
                            // ic_step - ic = ic_step*TR+t. Guarding
                            // `ic < ic_count` here (rather than relying on a
                            // partial-UNROLL loop bound like the pre-Lever-3
                            // version did) is required for the same reason
                            // it always was: window[]/wtile_packed[] beyond
                            // ic_count hold STALE data left over from a
                            // previous ic_tile pass (SHIFT_WINDOW/
                            // load_weight_tile only ever write up to
                            // ic_count), not zeros - reading them
                            // unconditionally would silently corrupt the sum
                            // for any layer where ic_count isn't a multiple
                            // of TR (conv_engine_tb.cpp's config-F exercises
                            // exactly this). `ic` itself never exceeds
                            // MAX_IN_CH-1 even without this guard (TR
                            // divides MAX_IN_CH evenly - conv_engine.h - so
                            // ic_steps*TR <= MAX_IN_CH whenever ic_count <=
                            // MAX_IN_CH), so this guard is purely a
                            // correctness requirement, not an out-of-bounds-
                            // access one. Now gates which VALUE this lane
                            // contributes (0 when out of range), not whether
                            // the accumulate happens - see this loop's outer
                            // comment for why that distinction is the whole
                            // point of this rewrite.
                            for (unsigned t = 0; t < TR; t++) {
#pragma HLS UNROLL
                                unsigned ic = ic_step * TR + t;
                                if (ic < ic_count) {
                                    act_t x = window[ic][ky][kx];
                                    packed_weight_t packed = wtile_packed[j][ic][ky][kx];

                                    // ONE DSP48E2-mappable signed multiply
                                    // (8b x 25b, both within the 27x18 port
                                    // limits) computing X*Y_lo and X*Y_hi
                                    // simultaneously. Natural ap_int
                                    // bit-growth (8+25=33) - do NOT pre-cast
                                    // operands to a wider type before
                                    // multiplying (that would force a
                                    // wider-than-33-bit multiply, spilling
                                    // across several DSP48E2s instead of one).
                                    ap_int<33> product = x * packed;
// TIER B1 (2026-08-05, TIMING_CLOSURE_PLAN.md §4 Tier B / §3 원인 B).
// Without this the DSP48E2 this maps to is PURELY COMBINATIONAL - the
// MAC_REDUCE csynth report shows `mul_24s_8s_32_1_1_*` instances with
// **FF = 0**, i.e. AREG/BREG/MREG/PREG all unused. That leaves the
// multiplier electrically continuous with the fabric logic on both sides,
// and the 200 MHz routed run's worst path (WNS -0.021) runs straight
// through it: 16-lane guard mux -> TREE_L1 -> TREE_L2 -> TREE_L3 with four
// CARRY8 stages, all in one pipeline stage.
//
// latency=2 turns on the DSP's internal registers, isolating the multiply
// from fabric on both sides. Start at 2; raise to 3 if still short.
//
// Costs pipeline DEPTH, not throughput: the enclosing loop is PIPELINE
// II=1, so extra multiplier latency is absorbed by the pipeline rather
// than multiplying the trip count. Arithmetic is untouched - this is a
// binding directive, so C-sim results must be bit-identical (that is the
// first thing to check after applying it, precisely because a pragma that
// silently does nothing looks the same as one that works).
#pragma HLS BIND_OP variable=product op=mul impl=dsp latency=2

                                    // Split, WITH a +1 carry correction on the
                                    // high half - caught wrong by
                                    // pack_unpack_selftest (conv_engine_tb.cpp)
                                    // before this fix. `product >> 16` (H_raw)
                                    // and "low 16 bits taken UNSIGNED" satisfy
                                    // product = H_raw*65536 + L_unsigned
                                    // unconditionally (that's just what an
                                    // arithmetic shift + low-bits split means).
                                    // But we want L reinterpreted as SIGNED
                                    // (l_signed = X*y_lo_biased, which is
                                    // negative whenever X<0, since
                                    // y_lo_biased is always >=0) - and
                                    // whenever L_unsigned's top bit is set
                                    // (i.e. l_signed < 0), that reinterpretation
                                    // silently "steals" 65536 from the high
                                    // half that a plain arithmetic shift never
                                    // gives back. H_raw is short by exactly 1
                                    // in exactly that case; nothing else needs
                                    // adjusting (l_signed/contrib_lo are
                                    // already exact on their own).
                                    ap_int<16> l_signed = product.range(15, 0);
                                    // partial_t, NOT accum_t - completes
                                    // Lever 1 (contrib_hi/contrib_lo were
                                    // left at accum_t/32-bit in Lever 1's
                                    // first pass, an oversight caught while
                                    // rewriting this block for Lever 3; both
                                    // contrib_hi/contrib_lo's own natural
                                    // range - an 8b x 8b product, |.| <=
                                    // 16384 - and acc_lo/acc_hi's declared
                                    // width already bound this comfortably).
                                    partial_t contrib_hi = (partial_t)(product >> 16);
                                    if (l_signed < 0) {
                                        contrib_hi += 1;
                                    }
                                    partial_t contrib_lo = (partial_t)l_signed - ((partial_t)x << 7);

                                    lo0[t] = contrib_lo;
                                    hi0[t] = contrib_hi;
                                } else {
                                    lo0[t] = 0;
                                    hi0[t] = 0;
                                }
                            }

                            // Balanced binary-tree reduce of the TR
                            // contributions - log2(TR) levels of PARALLEL
                            // adds, not a serial chain (see this loop's top
                            // comment for why that distinction is the point).
                            // Hardcoded to TR==16 (4 explicit levels: 16->8
                            // ->4->2->1) rather than a generic runtime-
                            // bounded loop - this file's own established
                            // preference (see e.g. the if/else-not-ternary
                            // and per-level-LOOP_TRIPCOUNT fixes elsewhere)
                            // is explicit code over a clever construct whose
                            // UNROLL/scheduling behavior would need its own
                            // separate synthesis-confirmation before trust.
                            // conv_engine.h constrains TR to a power-of-2
                            // divisor of MAX_IN_CH - if TR ever changes from
                            // 16, this tree must be re-written by hand to
                            // match (the static_assert below only catches
                            // silently building this against the wrong TR,
                            // it doesn't generalize the tree itself).
                            static_assert(TR == 16, "MAC_TR's explicit 4-level "
                                          "tree reduction is hardcoded for TR==16 "
                                          "- update it by hand if TR changes "
                                          "(conv_engine.h)");
                            partial_t lo1[8], hi1[8];
#pragma HLS ARRAY_PARTITION variable=lo1 complete dim=0
#pragma HLS ARRAY_PARTITION variable=hi1 complete dim=0
                        TREE_L1:
                            for (unsigned i = 0; i < 8; i++) {
#pragma HLS UNROLL
                                lo1[i] = lo0[2 * i] + lo0[2 * i + 1];
                                hi1[i] = hi0[2 * i] + hi0[2 * i + 1];
                            }
                            partial_t lo2[4], hi2[4];
#pragma HLS ARRAY_PARTITION variable=lo2 complete dim=0
#pragma HLS ARRAY_PARTITION variable=hi2 complete dim=0
                        TREE_L2:
                            for (unsigned i = 0; i < 4; i++) {
#pragma HLS UNROLL
                                lo2[i] = lo1[2 * i] + lo1[2 * i + 1];
                                hi2[i] = hi1[2 * i] + hi1[2 * i + 1];
                            }
                            partial_t lo3[2], hi3[2];
#pragma HLS ARRAY_PARTITION variable=lo3 complete dim=0
#pragma HLS ARRAY_PARTITION variable=hi3 complete dim=0
                        TREE_L3:
                            for (unsigned i = 0; i < 2; i++) {
#pragma HLS UNROLL
                                lo3[i] = lo2[2 * i] + lo2[2 * i + 1];
                                hi3[i] = hi2[2 * i] + hi2[2 * i + 1];
                            }

                            acc_lo[j] += lo3[0] + lo3[1];
                            acc_hi[j] += hi3[0] + hi3[1];
                        }
                    }
                }
            }

            // On the LAST ic_tile (or the only one, when num_ic_tiles==1):
            // real per-layer LeakyReLU(13/128) + requantize + saturate,
            // matching RTL_HANDOFF_KO.md section 5. On any earlier ic_tile,
            // writes a raw partial sum to `accum` instead - see
            // accumulate_or_finish() above. Runs after MAC_REDUCE since
            // acc_lo[j]/acc_hi[j] don't finish accumulating until every
            // MAC_REDUCE iteration above has run for every j.
            //
            // RESTRUCTURED TWICE (2026-08-04, both steps against real
            // measurements - TROUBLESHOOTING.md has the full story):
            //
            // Original: `PE_PAIR_FINISH`, an UNROLL'd loop making 24 calls
            // (2*PE_PAIRS at PE_OC=24) into the then-un-inlined
            // accumulate_or_finish() - ONE shared NON-PIPELINED instance
            // ("Interval 10-21 ... Pipeline: no" in the csynth Instance
            // table). An UNROLL'd loop calling one shared sequential
            // instance serializes: ~240-500 cycles per output position,
            // roughly half of layer 9's measured 742 cycles/COL_LOOP-body.
            //
            // First fix attempt (single `FINISH_LANES` loop, PIPELINE II=1,
            // callee inlined) DID NOT WORK - measured: layer-9 cosim
            // 855,307 -> 861,811 (+0.76%, i.e. nothing), and the csynth
            // Instance table showed why: `Pipeline_FINISH_LANES` came back
            // with fixed latency 340 for 24 iterations, i.e. achieved
            // II~14, not 1. One loop body mixing a CONDITIONAL m_axi accum
            // read, an accum write, and an ofmap write - all on the same
            // WR_BUS bundle, with the read inside a branch (the exact
            // `AccessInCondBranchMissed` pattern §23-d documented for
            // READ_CH) and a potential read-after-write alias on `accum`
            // within one body - left the scheduler no room to pipeline the
            // requests; II collapsed to roughly the old call interval.
            //
            // Current fix: PHASE-SPLIT into separate, unconditional,
            // consecutive-address pipelined loops - the same shape that
            // already gets II=1 for READ_CH's m_axi reads:
            //   gather (registers only) -> optional ACCUM_RD (read-only
            //   loop) -> ONE of FINISH_WR/ACCUM_WR (write-only loop).
            // The per-lane conditions (`ic_tile > 0`, "is this the last
            // ic_tile") never depended on `lane`, so they hoist out of the
            // loops entirely; the lane-validity guard (`global_oc <
            // out_ch`) becomes the loop BOUND (`lanes`) instead of an
            // in-body branch. No loop both reads and writes `accum`, so
            // there is no intra-loop alias to serialize around. Write
            // order across lanes is identical to the old 2j/2j+1 call
            // order, so the ofmap/accum contents are byte-identical - only
            // the timing changes.
            //
            // accumulate_or_finish() (the helper both earlier versions
            // called) is gone - its 4-way (first/middle/last/only ic_tile)
            // branch structure is exactly what's been flattened into these
            // phase loops; keeping a per-lane helper would just re-create
            // the in-body branching this fix exists to remove.
            unsigned pixel_idx = out_r * out_w + out_c;
            unsigned oc_base = oc_tile * PE_OC;
            unsigned lanes = out_ch - oc_base;
            if (lanes > 2 * PE_PAIRS) lanes = 2 * PE_PAIRS;

            accum_t acc_fin[2 * PE_PAIRS];
#pragma HLS ARRAY_PARTITION variable=acc_fin complete dim=0

        FINISH_GATHER:
            for (unsigned lane = 0; lane < 2 * PE_PAIRS; lane++) {
#pragma HLS UNROLL
                unsigned j = lane / 2;
                acc_fin[lane] = (lane % 2 == 0) ? (accum_t)acc_lo[j] : (accum_t)acc_hi[j];
            }

            if (num_ic_tiles > 1 && ic_tile > 0) {
            ACCUM_RD:
                for (unsigned lane = 0; lane < lanes; lane++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=24
                    acc_fin[lane] += accum[pixel_idx * PE_OC + lane];
                }
            }

            if (ic_tile == num_ic_tiles - 1) {
            FINISH_WR:
                for (unsigned lane = 0; lane < lanes; lane++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=24
                    ofmap[pixel_idx * out_ch + oc_base + lane] =
                        apply_activation(acc_fin[lane], leaky_relu_enable,
                                         requant_multiplier, requant_shift);
                }
            } else {
            ACCUM_WR:
                for (unsigned lane = 0; lane < lanes; lane++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=24
                    accum[pixel_idx * PE_OC + lane] = acc_fin[lane];
                }
            }
            }
        }
    }
}

void conv_engine(
    // PACK4: int8x4-packed ifmap - see pack4.h. Same DDR bytes, same
    // register map, `in_ch` still in channels.
    const pack4_t  *ifmap,
    const weight_t *weights,
    const weight_t *weights_hi,
    const bias_t   *bias,
    act_t          *ofmap,
    accum_t        *accum,
    uint16_t img_h, uint16_t img_w, uint16_t in_ch, uint16_t out_ch,
    uint8_t  k, uint8_t stride, uint8_t pad,
    int32_t  requant_multiplier, uint8_t requant_shift, uint8_t leaky_relu_enable
) {
    // ---- AMBA interface pragmas -------------------------------------------
    // ifmap/weights depth now scales with MAX_TOTAL_IN_CH (1024), not
    // MAX_IN_CH (128) - a real 8x depth increase, since these ports must be
    // addressable across a layer's FULL in_ch even though on-chip buffers
    // (window/wtile_packed) only ever hold one MAX_IN_CH-wide tile at a
    // time. `depth=` is a cosim/IP-XACT sizing hint only (synthesis ignores
    // it - confirmed in TROUBLESHOOTING.md), so this costs nothing in
    // synthesized hardware, but DOES make full-depth cosim buffers (see
    // conv_engine_tb.cpp's IFMAP_DEPTH/WEIGHTS_DEPTH) bigger - the existing
    // "only the one dedicated config-A-cosim call pays full depth" pattern
    // is what keeps that from slowing down cosim further.
// PACK4: ifmap's depth= is now in 32-bit WORDS, not int8 elements. The
// total addressable byte footprint is unchanged - only the pointer's unit
// changed. Every other port keeps its original element type and depth.
#pragma HLS INTERFACE m_axi port=ifmap   offset=slave bundle=RD_BUS  depth=MAX_IMG_W*MAX_IMG_W*MAX_TOTAL_IN_CH/PACK4_LANES
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=RD_BUS  depth=MAX_OUT_CH*MAX_TOTAL_IN_CH*MAX_K*MAX_K
    // Second, independent AXI4 master port aliasing the same DRAM region as
    // `weights` (own bundle, NOT RD_BUS) - see conv_engine.h's note on this
    // param and LOAD_W_IC's read of it below. This is the fix for the real
    // csynth-measured LOAD_W_IC II=2 (target II=1): two reads to `weights`
    // in one iteration could not both issue on RD_BUS in the same cycle, no
    // matter how the rest of RD_BUS's contention (ifmap/bias) was managed -
    // splitting `weights` onto its own physical port is what buys the second
    // read/cycle, not a bundle-sharing tweak. Costs one more physical AXI4
    // master port at the Vivado level - see vivado/create_bd.tcl.
#pragma HLS INTERFACE m_axi port=weights_hi offset=slave bundle=RD_BUS2 depth=MAX_OUT_CH*MAX_TOTAL_IN_CH*MAX_K*MAX_K
#pragma HLS INTERFACE m_axi port=bias    offset=slave bundle=RD_BUS  depth=MAX_OUT_CH
#pragma HLS INTERFACE m_axi port=ofmap   offset=slave bundle=WR_BUS  depth=MAX_IMG_W*MAX_IMG_W*MAX_OUT_CH
    // accum is only ever [out_h][out_w][PE_OC] - see conv_engine.h's note on
    // this parameter for why PE_OC and not out_ch.
#pragma HLS INTERFACE m_axi port=accum   offset=slave bundle=WR_BUS  depth=MAX_IMG_W*MAX_IMG_W*PE_OC

#pragma HLS INTERFACE s_axilite port=ifmap   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=weights bundle=CTRL
#pragma HLS INTERFACE s_axilite port=weights_hi bundle=CTRL
#pragma HLS INTERFACE s_axilite port=bias    bundle=CTRL
#pragma HLS INTERFACE s_axilite port=ofmap   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=accum   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=img_h   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=img_w   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=in_ch   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=out_ch  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=k       bundle=CTRL
#pragma HLS INTERFACE s_axilite port=stride  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=pad     bundle=CTRL
#pragma HLS INTERFACE s_axilite port=requant_multiplier bundle=CTRL
#pragma HLS INTERFACE s_axilite port=requant_shift       bundle=CTRL
#pragma HLS INTERFACE s_axilite port=leaky_relu_enable   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return  bundle=CTRL

    // ---- Defensive bounds checks (simulation only - __SYNTHESIS__ is a
    // Vitis HLS built-in macro defined during C/RTL synthesis and NOT
    // during C-simulation, the standard idiom for sim-only sanity checks
    // that cost zero hardware). These exist because MAX_IN_CH/MAX_K are
    // placeholders (see conv_engine.h) and stride>1 is unimplemented -
    // misconfiguration from SW would otherwise silently index out of bounds
    // or silently compute a stride-1 result for a stride>1 request instead
    // of failing loudly in simulation where it's cheap to catch. ------------
#ifndef __SYNTHESIS__
    // Checks img_h/img_w PLUS padding against line_buf's real capacity
    // (MAX_IMG_W+2, scan_and_compute()'s FIX comment on line_buf's
    // declaration) - not just img_h/img_w alone against MAX_IMG_W. A
    // plain `img_w <= MAX_IMG_W` check (this file's earlier version, see
    // TROUBLESHOOTING.md #19) passes for the real network's actual layer 0
    // (img_w=512, pad=1) while `pad_w = img_w+2*pad = 514` still overflows
    // line_buf's column dimension by 2 - the padded size, not the raw
    // size, is what COL_LOOP actually indexes with.
    assert(img_h + 2u * pad <= MAX_IMG_W + 2u &&
           "img_h+2*pad exceeds line_buf's capacity (MAX_IMG_W+2) - see conv_engine.h");
    assert(img_w + 2u * pad <= MAX_IMG_W + 2u &&
           "img_w+2*pad exceeds line_buf's capacity (MAX_IMG_W+2) - see conv_engine.h");
    // Was `in_ch <= MAX_IN_CH` - MAX_IN_CH is now just the on-chip tile
    // width (see conv_engine.h); a layer's real total in_ch is tiled across
    // multiple passes (IC_TILE below) up to MAX_TOTAL_IN_CH instead.
    assert(in_ch <= MAX_TOTAL_IN_CH && "in_ch exceeds MAX_TOTAL_IN_CH - see README design limits");
    // PACK4 contract. Deliberately NOT "in_ch % 4 == 0" - READ_CH has a
    // slow path for the remainder case precisely so layer 0 (in_ch=3) keeps
    // working without re-laying-out its RGB input in DDR. What IS required
    // is that the fast path's assumption holds whenever it is taken:
    // ic_lo must be word-aligned so each ic_tile pass starts on a word
    // boundary, and ic_count must then be a whole number of words. The
    // ic_tile width is MAX_IN_CH (128) - see `ic_lo = ic_tile * MAX_IN_CH`
    // below - so both hold by construction. Asserted rather than assumed,
    // since a future MAX_IN_CH change could silently break the fast path.
    assert((MAX_IN_CH % PACK4_LANES) == 0 &&
           "MAX_IN_CH (the ic_tile width) must be a multiple of 4 for READ_CH's "
           "packed fast path - see pack4.h");
    assert(out_ch <= MAX_OUT_CH && "out_ch exceeds MAX_OUT_CH - see conv_engine.h");
    assert(k <= MAX_K && k >= 1 && "k out of [1, MAX_K] range");
    assert(stride == 1 && "only stride=1 implemented - see README design limits");
    assert(pad <= MAX_K && "implausible pad value");
    // out_w/pad_h/pad_w (scan_and_compute) are computed as img_w+2*pad-k+1
    // etc. on unsigned types - a layer where the padded image is smaller
    // than the kernel would underflow that subtraction instead of just
    // being an invalid config. Not a realistic network layer (feature maps
    // are always far larger than a 1x1/3x3 kernel), but cheap to make an
    // explicit assertion rather than an implicit assumption.
    assert((unsigned)img_h + 2u * pad >= k && "img_h+2*pad must be >= k");
    assert((unsigned)img_w + 2u * pad >= k && "img_w+2*pad must be >= k");
    // round_shift()'s ap_int<64> intermediate has room to spare, but a
    // shift this large would mean the caller mis-programmed the register
    // (model_manifest.json's real values top out at 42) rather than any
    // legitimate layer needing it.
    assert(requant_shift <= 48 && "requant_shift implausibly large for round_shift's ap_int<64>");
#else
    (void)0; // no-op in synthesis - assertions above compile out entirely
#endif
    (void)stride; // enforced by the assertion above, not by branching hardware

    const unsigned num_oc_tiles = ((unsigned)out_ch + PE_OC - 1) / PE_OC;
    const unsigned num_ic_tiles = ((unsigned)in_ch + MAX_IN_CH - 1) / MAX_IN_CH;

    // wtile/btile are owned here (not inside load_weight_tile/scan_and_compute)
    // specifically so a future double-buffered version only needs to
    // duplicate these two arrays and add a DATAFLOW pragma around the loop
    // body below - see RESOURCE_BUDGET.md §5. Tried 2026-07-23 (moving these
    // inside the loop, non-static, plus DATAFLOW on OC_TILE): hit a hard
    // Vitis HLS error before ever reaching the ping-pong-buffering question -
    // `ifmap`/`weights`/`bias` all share one AXI bundle (RD_BUS below), and
    // DATAFLOW requires each AXI bundle be read by only one process; with
    // load_weight_tile() (weights/bias) and scan_and_compute() (ifmap) now
    // meant to run concurrently, sharing RD_BUS became a real conflict, not
    // just a style issue: "[HLS 200-1013] Bundled bus interface 'RD_BUS' ...
    // failed dataflow checking: it cannot read data in multiple processes."
    // Reverted rather than also splitting `ifmap` onto its own AXI bundle -
    // that's a real interface change (one more physical AXI master port),
    // not a pragma-only fix, and would also need README.md's AMBA interface
    // map and the eventual Vivado `create_bd.tcl` wiring updated to match.
    // See RESOURCE_BUDGET.md §5 for the full finding and what doing this for
    // real would require.
    //
    // wtile_packed is complete on dim=1 (pair index j) - PE_PAIR_LOOP in
    // scan_and_compute UNROLLs j, so each of the PE_PAIRS lanes needs a
    // compile-time-constant-indexed, simultaneously-readable slice - and
    // cyclic(TR) on dim=2 (ic), matching MAC_IC's UNROLL factor=TR for the
    // same reason. It is NOT partitioned on dim=3/dim=4 (ky/kx) - see the
    // "FIX 2" comment on window's declaration in scan_and_compute() for why:
    // nothing accesses it through a ky/kx index that's ever actually
    // unrolled (load_weight_tile's and MAC_KY/MAC_KX's ky/kx loops are
    // PIPELINE, not UNROLL).
    //
    // INT8 DSP-packing experiment (conv_engine.h): this replaces the
    // original wtile[PE_OC][...] with wtile_packed[PE_PAIRS][...] - half as
    // many instances (128 vs 256) for the same reason the multiplier count
    // halves, since each pair now shares one packed weight value instead of
    // two independent weight_t values.
    static packed_weight_t wtile_packed[PE_PAIRS][MAX_IN_CH][MAX_K][MAX_K];
    static bias_t btile[PE_OC];
#pragma HLS ARRAY_PARTITION variable=wtile_packed complete dim=1
#pragma HLS ARRAY_PARTITION variable=wtile_packed cyclic factor=TR dim=2
#pragma HLS ARRAY_PARTITION variable=btile complete dim=0
    // History (PE_OC=16, real csynth run): packed_weight_t (25 bits) is ~3x
    // wider per element than the original weight_t (8 bits), and even though
    // instance count halved (256 -> 128, cyclic(TR) x PE_PAIRS), each
    // surviving instance's data volume (72 elements x 25 bits = 1,800 bits)
    // crossed Vitis's automatic BRAM-vs-LUTRAM inference threshold - the
    // original 576-bit instances stayed off BRAM entirely, but all 128 of
    // these got mapped to individual BRAM_18K blocks, pushing total design
    // BRAM_18K from 23% to 67%. Forced back into distributed/LUT-based
    // storage (impl=LUTRAM) at the time to trade that back for LUT headroom.
    //
    // TRIED AND REJECTED (2026-07-24, real csynth run at the current
    // PE_OC=20/weights_hi config): re-tested impl=BRAM on the theory that
    // BRAM_18K's headroom (72/288, 25%, unlike the PE_OC=16 measurement
    // above where BRAM was the one under pressure) could be spent to pull
    // LUT down from its 92% baseline (107,732/117,120, the tightest
    // resource in the design). Real result was a regression on every
    // metric, not a trade: LUT 107,732 -> 111,412 (92% -> 95%, WORSE, not
    // better), FF 47,335 -> 59,181 (+25%, from BRAM's synchronous
    // read-latency output registers on ~80-128 partitioned instances), and
    // BRAM_18K itself 72 -> 232 (25% -> 81%, most of the "headroom" this was
    // supposed to spend). DSP and Fmax were unaffected either way. Per-
    // instance BRAM control/address/output-register overhead across this
    // many small partitioned instances apparently costs more than LUTRAM's
    // direct implementation saves - impl=LUTRAM was already the correct
    // choice at this config. Do not retry BRAM here without a real
    // architecture change (e.g. fewer, larger partitions) first.
#pragma HLS BIND_STORAGE variable=wtile_packed type=RAM_1P impl=LUTRAM

OC_TILE:
    for (unsigned oc_tile = 0; oc_tile < num_oc_tiles; oc_tile++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
    IC_TILE:
        // Known, expected trade-off - now on TWO axes instead of one: ifmap
        // is re-read from DDR once per (oc_tile, ic_tile) pair
        // (num_oc_tiles*num_ic_tiles total passes) instead of once per
        // oc_tile. See RESOURCE_BUDGET.md §2 - the "engine efficiency"
        // derating factor there already accounted for the oc_tile axis
        // without a real cosim trace to measure it from; this multiplies
        // that same unmeasured factor by num_ic_tiles too, for any layer
        // where it's >1 (8 of the real network's 13 conv layers - see
        // conv_engine.h). `ic_tile` is the loop that actually needs
        // `accum` (see accumulate_or_finish() above) - `oc_tile` staying
        // the OUTER loop is what lets `accum` be sized [out_h][out_w][PE_OC]
        // instead of [out_h][out_w][out_ch]: one oc_tile fully finishes
        // (across all its ic_tiles) before the next one starts, so the
        // same small accum region is safely reused/overwritten fresh each
        // time, never needing to hold more than one oc_tile's channels.
        for (unsigned ic_tile = 0; ic_tile < num_ic_tiles; ic_tile++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=8
            unsigned ic_lo = ic_tile * MAX_IN_CH;
            unsigned ic_count = ((unsigned)in_ch - ic_lo < MAX_IN_CH)
                                     ? ((unsigned)in_ch - ic_lo) : MAX_IN_CH;
            load_weight_tile(weights, weights_hi, bias, oc_tile, ic_lo, ic_count, in_ch, out_ch, k,
                              wtile_packed, btile);
            scan_and_compute(ifmap, accum, ofmap, img_h, img_w, ic_lo, ic_count, in_ch, out_ch,
                              k, pad, oc_tile, ic_tile, num_ic_tiles, wtile_packed, btile,
                              leaky_relu_enable != 0, requant_multiplier, requant_shift);
        }
    }
}
