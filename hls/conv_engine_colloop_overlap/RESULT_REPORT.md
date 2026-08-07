# COL_LOOP 오버랩 레버 — 실험 결과 보고 (결론: 원본 유지)

- 작성일: 2026-08-07
- 대상: generic `conv_engine`의 `COL_LOOP` — ifmap read ↔ compute 오버랩
- 측정 환경: KR260(xck26), 200MHz, **real layer 3**, Vitis HLS 2024.2, `--cosim-only 3`

## TL;DR
COL_LOOP 오버랩을 **3가지 방식**으로 시도했으나 **전부 baseline(원본)을 못 넘음.**
**원본 conv_engine이 최선** (layer3 = **1,999,021 cycles**). 이 레버는 **접는 걸 권고.**

## 기준선 (baseline)
- 원본, layer 3 cosim 실측: **1,999,021 cycles** (기존 문서값과 일치, bit-exact)

## 시도 1 — 수동 ping-pong prefetch (v1)
- 방법: `px` 2뱅크 핑퐁 + `read_col_into()` 헬퍼, 컬럼 c 계산 중 c+1을 미리 read (prologue + prefetch).
- 결과: csim **bit-exact ✓**, cosim **2,003,497 (+4,476, +0.22%) → 느려짐.**
- 원인: HLS는 `DATAFLOW` 없이 read/compute 서브루프를 **순차 스케줄** → 오버랩 안 됨. 매 행 phantom-column prefetch만 잔업으로 추가.

## 시도 2 — phantom-read 가드 (v1b)
- 방법: 마지막 컬럼 prefetch를 `if (c+1 < pad_w)`로 스킵 (phantom read 제거 목적).
- 결과: csim **bit-exact ✓**, cosim **2,019,301 (+20,280, +1.01%) → 더 느려짐.**
- 원인: phantom read ~228회(-4.5k)는 제거했으나, 매 COL_LOOP 반복(~15,000회)에 분기 비용 ~1cyc/iter(+15.8k) 추가 → 순손해. (COL_LOOP 비파이프라인이라 분기가 그대로 사이클로 누적.)

## 시도 3 — DATAFLOW producer/consumer 분리 (v2)
- 방법: `scan_and_compute`를 `read_stream`(ifmap→`hls::stream`) / `compute_stream`(window+MAC, WR_BUS)로 분리, `#pragma HLS DATAFLOW`. RD_BUS/WR_BUS가 프로세스별로 분리돼 §14 번들공유 문제는 회피.
- 결과:
  - csim **bit-exact ✓** (첫 시도 통과)
  - csynth: **DATAFLOW 적용 확인** (`scan_and_compute` = dataflow type). LUT 78% / BRAM 47%→**67%**(FIFO 비용) / DSP 22% — 전부 fit.
  - cosim: **실패** — `SIGSEGV @ ENTER_WRAPC` (apatb "insufficient depth"). **m_axi 포트가 dataflow 프로세스 내부에 있어 cosim 자동 테스트벤치(apatb)가 깨짐** — 로직 버그 아님(csim이 정확성 증명), **cosim 툴 한계.**
- 판정: **기능적으로는 유효하나 성능 실측 경로가 막힘.** 뚫으려면 m_axi를 dataflow 밖으로 빼야 하는데, 그러면 오버랩 취지가 사라져 앞뒤가 안 맞음.

## 결론
| 버전 | layer3 cycles | vs baseline |
|---|---|---|
| **baseline (원본)** | **1,999,021** | — (최선) |
| v1 ping-pong | 2,003,497 | +0.22% (느림) |
| v1b + 가드 | 2,019,301 | +1.01% (더 느림) |
| v2 DATAFLOW | 측정불가 | cosim 툴 한계 |

- 근본 원인: 이 엔진에선 COL_LOOP당 **read(~19cyc)가 compute(~190cyc) 대비 소수** → 오버랩 이득 상한이 애초에 작고(~8%), 그마저 HLS 스케줄링/cosim 제약으로 실현 안 됨.
- **권고: COL_LOOP 오버랩 레버는 종료. 원본 conv_engine 유지.** 프레임 사이클 개선은 다른 레버(OC-hoist 등)에서 찾는 것을 제안.

## 재현 / 코드 위치
- v1: `ADAS_PROJ/hls/conv_engine_colloop_overlap` (csim/csynth/cosim 리포트 포함)
- v2: 로컬 `conv_engine_dataflow_v2` (csim/csynth 통과, cosim SIGSEGV 로그 보존)
- 측정 전부 real layer 3, `--cosim-only 3`.
