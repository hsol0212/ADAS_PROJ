# HLS Conv Engine — COL_LOOP Overlap 최적화 (PoC)

KR260(Zynq UltraScale+) YOLOv3-tiny-ADAS CNN 가속기의 **convolution 엔진(Vitis HLS)** 에서,
`COL_LOOP`의 **입력 읽기(ifmap read)** 와 **연산(window 갱신 + MAC)** 을 소프트웨어
파이프라인으로 **겹쳐서(overlap)** 프레임 사이클을 줄이려는 최적화 실험 저장소입니다.

> 이 저장소는 팀 KR260 ADAS 프로젝트의 일부이며, **COL_LOOP overlap 기여분**에 초점을
> 둡니다. `conv_engine.cpp`의 대부분 로직/튜닝은 팀 공동 작업물이고, 이 PoC는 그 위에
> 얹은 변경입니다.

## 무엇을 했나 — ping-pong prefetch

기존 `COL_LOOP`은 한 컬럼마다 `READ_CH → window 갱신 → MAC` 을 **순차** 실행해서,
메모리를 읽는 동안 연산기가 놀고 연산하는 동안 메모리가 놉니다.

```
[before] c: [READ][SHIFT+MAC]   c+1: [READ][SHIFT+MAC]     (읽기와 계산이 안 겹침)

[after ] c=0: [READ]
         c=1:      [READ ][SHIFT+MAC(0)]                   (c+1 읽기 ∥ c 계산)
         c=2:             [READ ][SHIFT+MAC(1)]
```

핵심 변경(`HW/conv_engine.cpp`):
1. **`read_col_into()`** — 인라인 READ_CH를 forced-INLINE 헬퍼로 분리 (prologue/prefetch 재사용용)
2. **`pxbuf[2][MAX_IN_CH]`** — px를 2뱅크 ping-pong 버퍼로 승격 (cyclic(TR) 파티션)
3. **prologue** — 각 행에서 컬럼 0을 미리 read
4. **prefetch** — 반복 `c`에서 컬럼 `c+1`을 `pxbuf[nxt]`로 미리 read, 현재 컬럼은 `pxbuf[cur]`로 계산
5. 원본 인라인 read는 `#if 0`으로 남겨 before/after 비교가 쉽게 되어 있음

**bit-exact 보장**: 각 컬럼은 여전히 정확히 한 번씩 읽히고 window 갱신 순서도 동일 →
출력 픽셀 값 불변. (마지막 phantom 컬럼 prefetch는 `in_bounds=false`라 AXI 접근 없이 0-fill.)

## 검증 (설계 + 검증)

| 단계 | 목적 | 도구 |
|---|---|---|
| C-sim | golden 대비 **bit-exact** (7 config + 13 real layer) | Vitis HLS `csim_design` |
| C-synth | 자원(LUT/DSP/BRAM), achieved II, 예상 클럭 | `csynth_design` |
| Co-sim | 실제 사이클 (before/after 비교) | `cosim_design` |

### 측정 결과 (real layer 3, 2026-08-06)

| 항목 | Before | After (v1) | 판정 |
|---|---|---|---|
| 정확성 (csim) | 기준 | 일치 | **bit-exact PASS** ✓ |
| 사이클 (cosim) | 1,999,021 | 2,003,497 | +4,476 (**+0.22%**) — 이득 없음 |
| LUT | 76.9% | 78.3% | 여유 (한도 100%) |
| DSP / BRAM | 22% / 47% | 22% / 47% | 동일 |

**결론**: 구현은 bit-exact로 정확하고 회로 자원도 여유가 있지만, **사이클은 줄지 않았다**(오히려 +0.22%).
원인은 HLS 스케줄링 — `DATAFLOW` 지시 없이는 도구가 prefetch 읽기와 계산 서브루프를 **순차 실행**하므로
오버랩이 일어나지 않고, 매 행의 phantom-column prefetch만 ~4.5k 사이클의 순수 오버헤드로 남는다.
즉 **"수동 코드 재배치만으로는 오버랩 불가"가 실측으로 확인**됨. 진짜 오버랩은 ifmap read를 별도
`DATAFLOW` 프로세스로 분리해야 하는데, `window`/`line_buf`의 컬럼 간 의존성 때문에 단순하지 않다(→ v2 후보).

## 빌드 / 실행

Vitis HLS **2024.2** 필요. Windows에서 Vitis 환경 로드 후:

```bat
call "<VITIS>\2024.2\settings64.bat"
run_hls.bat            :: C-sim + C-synthesis
run_hls.bat cosim      :: + co-simulation (느림)
```

> `HW/real_layers_data.h`(생성된 골든 데이터, 대용량)는 저장소에서 제외되어 있습니다.
> 실제 레이어 테스트를 돌리려면 canonical 프로젝트의 `python/real_layers.py`로 재생성하세요.
> 합성 산출물(`conv_engine_prj/`)도 `.gitignore` 대상이며 `run_hls`가 재생성합니다.

## 상태
- [x] baseline 확보 — layer 3 = 1,999,021 cycles
- [x] ping-pong prefetch 구현 (v1)
- [x] csim bit-exact 검증 — **PASS** (ALL CONFIGS PASS, 0 errors)
- [x] csynth 자원 확인 — LUT 78.3% (여유)
- [x] cosim 사이클 비교 — 2,003,497 (이득 없음, +0.22%)
- [ ] (다음 후보) v2: ifmap read를 `DATAFLOW` 별도 프로세스로 분리 — sliding-window 의존성 때문에 난이도 높음

> **요약**: v1(수동 ping-pong 재배치)은 정확성·자원은 OK지만 사이클 이득 없음. HLS는 `DATAFLOW` 없이
> 서브루프를 겹치지 않는다는 것을 실측으로 확인한 PoC. (가설 → 구현 → 검증 → 측정 → 결론)
