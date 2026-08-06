#ifndef PACK4_H
#define PACK4_H

#include <ap_int.h>

// ---------------------------------------------------------------------------
// int8x4 packing for conv_engine's ifmap m_axi port ("pack4" variant)
//
// Same technique already merged into hls/pool_upsample_route (see that
// project's PACK4_NOTES.md), applied here to the ONE port in conv_engine
// that the current csynth report says is worth it.
//
// WHY HERE, AND WHY ONLY `ifmap`
// ------------------------------
// After the finish-path fix (TROUBLESHOOTING §25, conv efficiency 4.3% ->
// 10.6%), the csynth loop table for one COL_LOOP scan position reads:
//
//     READ_CH (ifmap read)          12 ~ 139   <-- LARGEST single term
//     MAC_REDUCE / MAC_KY / MAC_KX   7 ~  78
//     WINDOW_ROW/COL_SHIFT           2 ~  51
//     FINISH_WR (ofmap write)       13 ~  36
//     WINDOW_TAIL_FILL               4 ~  27
//     ACCUM_RD                       4 ~  27
//     ACCUM_WR                       3 ~  26
//     LINE_BUF_ROW_SHIFT             2 ~  18
//     LINE_BUF_TAIL_FILL             3 ~  10
//     ------------------------------------------
//     COL_LOOP total                19 ~ 411
//
// (The maxima sum to 412 against a reported COL_LOOP max of 411, so they do
// essentially co-occur - this is not unrelated bounds being added up.)
//
// READ_CH moves `ic_count` int8 values one per AXI beat. Packing 4 channels
// per beat should take its 139 worst case to roughly 128/4 + 11 = 43,
// i.e. ~96 cycles off a 411-cycle COL_LOOP.
//
// NOT applied to the other ports, deliberately:
//   - `accum` is accum_t (ap_int<32>) already - the beat is already full.
//   - `weights`/`bias` are read in LOAD_WTILE, which totals 14,256 cycles
//     inside an IC_TILE whose max is 108,600,361 - about 0.013%. Not worth
//     the disturbance to the weights/weights_hi port split that exists
//     specifically to hold II=1 there.
//   - `ofmap` (FINISH_WR, 8.8%) is a real but smaller candidate; left for a
//     separate change so this one can be measured on its own.
//
// A NOTE ON THE STALE COMMENT THIS CONTRADICTS
// --------------------------------------------
// conv_engine.cpp's comment above READ_CH still says READ_CH is "12~139
// against SHIFT_WINDOW's 49~3554" and therefore a minority of runtime. That
// was true when written; SHIFT_WINDOW has since been optimized ~70x to
// 2~51, which is what makes READ_CH the top term today. That comment is
// corrected in this variant.
//
// WHAT THIS DOES NOT CHANGE
// -------------------------
//  1. The physical AXI port width. `m_axi_..._WDATA` is already 32 bits
//     wide with a 4-bit WSTRB; `ap_int<8>*` never narrowed the bus, it just
//     left 3 of 4 byte lanes unused per beat. `ap_uint<32>*` produces the
//     identical port geometry - no block-design edit.
//  2. The DDR byte layout / PS-side data contract. On this little-endian
//     platform 4 consecutive NHWC channels ARE the 4 bytes of one 32-bit
//     word, lowest channel in the lowest byte. Packing is a pure
//     reinterpretation of the same bytes, so a buffer written by the pool
//     engines is a valid conv input and vice versa.
//  3. The s_axilite register map. `in_ch` is still counted in CHANNELS.
//
// WHAT IT REQUIRES (asserted in conv_engine())
// --------------------------------------------
//  1. `in_ch % 4 == 0`. Every real YOLOv3-tiny-ADAS conv layer complies
//     (3 is the sole exception - layer 0's RGB input - see conv_engine.cpp's
//     assert for how that case is handled).
//  2. `ifmap` 4-byte aligned. Any ordinary allocator result already is.
//  3. `ic_lo % 4 == 0`, so each ic_tile pass starts on a word boundary.
//     IC_TILE is 128, so this holds by construction; asserted anyway.
// ---------------------------------------------------------------------------

// One AXI beat = 4 consecutive NHWC channels. Lane L occupies bits
// [8L+7 : 8L], i.e. lane 0 is the LOW byte = the LOWEST channel index,
// matching little-endian DDR byte order.
typedef ap_uint<32> pack4_t;

const unsigned PACK4_LANES = 4;

// Bit-copy accessor, written as an explicit .range() move through a
// same-width temporary rather than a cast between ap_uint<8> and ap_int<8>.
// A cast would rely on ap_int's modular-truncation rule to turn 0x80 into
// -128; this states the bit copy directly.
inline ap_int<8> pack4_get(pack4_t w, unsigned lane) {
#pragma HLS INLINE
    ap_int<8> v;
    v.range(7, 0) = w.range(8 * lane + 7, 8 * lane);
    return v;
}

inline void pack4_set(pack4_t &w, unsigned lane, ap_int<8> v) {
#pragma HLS INLINE
    w.range(8 * lane + 7, 8 * lane) = v.range(7, 0);
}

#endif // PACK4_H
