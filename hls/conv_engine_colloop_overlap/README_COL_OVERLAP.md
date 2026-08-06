# conv_engine_col_overlap — COL_LOOP 오버랩 PoC (격리 사본)

## 출처 / 목적
`conv_engine_requant_merged/`의 **격리 사본**(HW 소스 + run_hls 스크립트만; 빌드
산출물 `conv_engine_prj/`는 `run_hls`가 재생성). SW 최적화 레버 중
**COL_LOOP 오버랩** 담당 트랙 작업공간이다. canonical은 직접 편집하지 않는다는
팀 규칙(git 부재 → 각자 사본 후 수동 병합)에 따라 분리했다.

## 무엇을 하려는가
`scan_and_compute()`의 `COL_LOOP`은 한 컬럼 `c`마다
`READ_CH(ifmap 읽기) → FUSED_SHIFT_STEP(window 갱신) → MAC_REDUCE(계산)`을
**순차** 실행한다. 메모리 읽기(READ_CH)와 연산(MAC)이 번갈아 놀기 때문에,
컬럼 `c` 계산 중 `c+1`의 입력을 미리 읽는 **2단 수동 파이프라인**으로 겹쳐
프레임 사이클을 줄이는 것이 목표.

- 금지: `COL_LOOP` 바깥에 단순 `#pragma HLS PIPELINE` (내부 완전 언롤 → 자원/타이밍 폭발)
- 주의: `window`/`line_buf`는 컬럼 간 의존성 있음 → prefetch 대상은 **px 읽기만**

## 검증 계획 (portfolio: 설계 + 검증)
1. baseline: 사본 무수정 상태로 `run_hls.bat cosim` → before 사이클 기록
2. 구현 후 csim: golden 대비 bit-exact (12 config + 13 real layer)
3. csynth: achieved II, 내부 언롤 폭발 없음, LUT/DSP/BRAM
4. cosim: after 사이클 → 절감률 산출

## 실행 (이 PC)
Vitis가 `D:\Vitis\2024.2`에 있으므로 `run_hls.bat` 자동탐지(C:\Xilinx)는 실패한다.
cmd에서 먼저 `call "D:\Vitis\2024.2\settings64.bat"` 후 실행할 것.
