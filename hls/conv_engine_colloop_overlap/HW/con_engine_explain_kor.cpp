// =============================================================================
// conv_engine.cpp 한글 라인별 설명 버전
// (con_engine_explain_kor.cpp)
//
// 원본 파일: conv_engine.cpp / conv_engine.h
// 목적: YOLOv2-tiny 등 CNN의 컨볼루션 레이어를 "레이어마다 하나씩" HW로
//       찍어내는 대신, 하나의 IP를 시간분할(time-multiplexed)로 재사용하는
//       Vitis HLS 컨볼루션 엔진. SW가 매 레이어마다 크기(geometry) 레지스터와
//       DDR 주소를 다시 세팅하고 재기동시키는 구조 (CPU가 매 클럭 다른
//       명령어를 실행하는 것과 유사한 컨셉).
//
// 이 파일은 컴파일용이 아니라 "설명용"입니다. 원본 코드/프라그마를 그대로
// 두고, 각 줄/블록 위에 한국어 설명 주석을 추가했습니다.
// =============================================================================

#include "conv_engine.h"
#include <cassert>

// -----------------------------------------------------------------------------
// [헤더 conv_engine.h 요약 - 이 파일에서 쓰는 상수/타입]
//
//   MAX_IMG_W  = 416   : 온칩 버퍼가 감당할 수 있는 최대 가로/세로 크기
//                        (YOLOv2-tiny 입력단 크기 기준. 실제 레이어 크기가
//                        아니라 "버퍼를 얼마나 크게 잡아둘지"의 상한선)
//   MAX_IN_CH  = 128   : 최대 입력 채널 수 (이 버전에서는 in_ch를 타일링하지
//                        않음 - in_ch가 이보다 크면 이 엔진으로 못 돌림)
//   MAX_OUT_CH = 1024  : 최대 출력 채널 수 (out_ch는 PE_OC 단위로 타일링되므로
//                        이 값은 bias 버퍼 크기 + oc_tile 루프 횟수만 결정,
//                        MAC 병렬성과는 무관)
//   MAX_K      = 3     : 최대 커널 크기 (1x1 레이어는 k=1로 window[.][0][0]만 사용)
//   PE_OC      = 16    : 한 번에 병렬로 계산하는 출력 채널 개수
//                        (Processing Element per Output Channel)
//   TR         = 16    : PE_OC 레인 하나당, in_ch 축으로 동시에 언롤되는
//                        리덕션(reduction) 항 개수. PE_OC*TR = 256개의 병렬
//                        INT8 MAC (KR260 DSP 예산 대비 약 20.5%)
//
//   weight_t = ap_int<8>   : 8비트 부호있는 가중치 타입
//   act_t    = ap_int<8>   : 8비트 부호있는 activation(특성맵) 타입
//   accum_t  = ap_int<32>  : 32비트 부호있는 누산기(accumulator) 타입
//                            (8bit x 8bit 곱을 여러 번 더해도 오버플로 안 나게
//                            충분히 넓게 잡음)
// -----------------------------------------------------------------------------

// saturate(): 32비트 누산값(accum_t)을 8비트 activation 범위(-128~127)로
// "클램프(clamp)"하는 함수. 곱셈-누적 결과가 8비트 표현 범위를 넘으면 자르고,
// 안 넘으면 그대로 8비트로 캐스팅해서 반환한다.
static act_t saturate(accum_t v) {
    if (v > 127)  return (act_t)127;   // 127보다 크면 최댓값 127로 자름 (양의 포화)
    if (v < -128) return (act_t)-128;  // -128보다 작으면 최솟값 -128로 자름 (음의 포화)
    return (act_t)v;                   // 범위 안이면 그대로 8비트로 downcast
}

// ---------------------------------------------------------------------------
// [설계 노트 원문 요약]
// load_weight_tile()와 scan_and_compute() 두 함수로 쪼갠 이유는 단순 가독성이
// 아니라, 나중에 "가중치 타일 더블 버퍼링"(RESOURCE_BUDGET.md §5, 지금은
// 보류)을 도입할 때 이 두 함수 호출을 #pragma HLS DATAFLOW로 감싸고
// wtile/btile을 핑퐁(ping-pong) 버퍼로 바꾸기만 하면 되도록, 알고리즘 자체는
// 안 건드리고 구조만 재배치할 수 있게 하기 위함이다. wtile/btile은 호출자인
// conv_engine()이 소유하고 두 함수에 "전달"만 하는 방식 - 나중에 더블 버퍼링
// 버전을 만들 때 호출자만 고치면 되고, 이 두 함수 내부 로직은 그대로 두면 됨.
// ---------------------------------------------------------------------------

// load_weight_tile(): 현재 처리할 출력채널 타일(oc_tile) 하나 분량의
// 가중치(weight)와 편향(bias)을 DDR(m_axi 포트)에서 온칩 버퍼 wtile/btile로
// 읽어오는 함수.
static void load_weight_tile(
    const weight_t *weights, const weight_t *bias,
    unsigned oc_tile, unsigned in_ch, unsigned out_ch, unsigned k,
    weight_t wtile[PE_OC][MAX_IN_CH][MAX_K][MAX_K], weight_t btile[PE_OC]
) {
    // [핵심 설명] 왜 강제로 INLINE인가?
    // Vitis HLS에는 실제로 문서화된 함정이 있다: "완전 분할(complete
    // partition)된 배열"이 INLINE되지 않은 함수 경계를 넘어 전달되면, 그
    // 배열의 병렬 접근(partitioning) 속성이 조용히 사라질 수 있다 (함수
    // 경계가 일종의 포트(port)처럼 취급되어, 항상 원소별 병렬 접근을
    // 보존해주지는 않기 때문). wtile은 conv_engine()에서 선언될 때
    // 파티션되어 있고, scan_and_compute()의 MAC 리덕션이 바로 그 파티셔닝이
    // 주는 병렬성에 의존한다. 툴이 "기본값으로 알아서 인라인해주겠지"에
    // 기대는 대신, 명시적 pragma로 강제해서 이 병렬성이 "툴 버전에 따라
    // 달라질 수도 있는 암묵적 동작"이 아니라 "명시된 동작"이 되게 한 것.
    // (나중에 DATAFLOW/더블버퍼 버전에서는 이 INLINE 프라그마를 일부러
    // 빼야 함 - DATAFLOW는 스테이지들이 인라인되지 "않아야" 동작하므로.)
#pragma HLS INLINE
LOAD_WTILE:
    // pe: 0..PE_OC-1, 이번 oc_tile 안에서 병렬 처리할 출력채널 레인 인덱스
    for (unsigned pe = 0; pe < PE_OC; pe++) {
        // oc: 이 pe 레인이 실제로 담당하는 "전체 출력채널 중 몇 번째"인지
        // (oc_tile * PE_OC 만큼 오프셋을 더해서 절대 채널 인덱스로 변환)
        unsigned oc = oc_tile * PE_OC + pe;
        // out_ch가 PE_OC의 배수가 아닐 수 있으므로, 실제 존재하는 채널
        // 범위를 넘어가는 pe 레인은 아무 것도 로드하지 않고 건너뜀
        if (oc < out_ch) {
            // 이 출력채널의 편향(bias) 값을 온칩 btile에 저장
            btile[pe] = bias[oc];
        LOAD_W_IC:
            // 입력채널(ic) 전체를 순회하며 이 출력채널의 커널 가중치를 로드
            for (unsigned ic = 0; ic < in_ch; ic++) {
                for (unsigned ky = 0; ky < k; ky++) {      // 커널 세로 방향
                    for (unsigned kx = 0; kx < k; kx++) {  // 커널 가로 방향
                        // 이 3중 루프(ic,ky,kx)를 파이프라인 II=1로 실행:
                        // 매 사이클 하나의 가중치를 순차적으로 읽어옴.
                        // (이 순차 접근 패턴이 m_axi 버스트 리드로 합쳐질 수
                        // 있게 해주는 핵심 - 아래 conv_engine() 쪽 주석 참고)
#pragma HLS PIPELINE II=1
                        // weights[]는 [out_ch][in_ch][k][k] 형태의 1차원
                        // row-major 배열이므로, 4차원 인덱스를 아래처럼 직접
                        // 계산해서 선형 오프셋(w_idx)을 구함
                        unsigned w_idx = ((oc * in_ch + ic) * k + ky) * k + kx;
                        wtile[pe][ic][ky][kx] = weights[w_idx];
                    }
                }
            }
        }
    }
}

// scan_and_compute(): 입력 특성맵(ifmap)을 래스터 스캔(raster scan) 방식으로
// 한 번 훑으면서, 현재 oc_tile에 해당하는 출력채널들의 컨볼루션 결과를
// 계산해 ofmap에 쓰는 함수. line buffer + sliding window 구조로, 전체 이미지를
// 온칩에 다 올리지 않고도 컨볼루션을 수행한다 (conv_layer1.cpp와 동일한
// 구조를 런타임 in_ch/k에 맞게 일반화한 버전).
static void scan_and_compute(
    const act_t *ifmap, act_t *ofmap,
    unsigned img_h, unsigned img_w, unsigned in_ch, unsigned out_ch,
    unsigned k, unsigned pad, unsigned oc_tile,
    const weight_t wtile[PE_OC][MAX_IN_CH][MAX_K][MAX_K], const weight_t btile[PE_OC]
) {
    // load_weight_tile()과 동일한 이유로 강제 INLINE (위 설명 참고)
#pragma HLS INLINE

    // 출력 가로 크기: stride=1 기준 표준 컨볼루션 출력 크기 공식
    // out_w = (img_w + 2*pad - k) / stride + 1, stride=1이므로 /1 생략
    const unsigned out_w = img_w + 2 * pad - k + 1; // stride=1
    // 패딩까지 포함한 "가상의" 세로/가로 크기 (스캔은 이 패딩된 좌표계 기준으로 돎)
    const unsigned pad_h = img_h + 2u * pad;
    const unsigned pad_w = img_w + 2u * pad;

    // -------------------------------------------------------------------
    // line_buf: 슬라이딩 윈도우를 만들기 위해, 현재 행(row) 기준으로 위쪽
    //           (k-1)개 행의 픽셀들을 채널별로 저장해두는 라인 버퍼.
    //           크기: [최대입력채널][최대 (k-1)][최대 가로폭]
    // window:   실제로 이번 사이클에 컨볼루션 곱셈에 쓰이는 k x k 크기의
    //           슬라이딩 윈도우 (채널별로 하나씩).
    //
    // 두 배열 모두 "static"으로 선언되어 함수 호출 사이에 값이 유지됨
    // (다음 oc_tile 스캔 때도 같은 온칩 메모리를 재사용).
    // -------------------------------------------------------------------
    static act_t line_buf[MAX_IN_CH][MAX_K - 1][MAX_IMG_W];
    static act_t window[MAX_IN_CH][MAX_K][MAX_K];

    // ---- 배열 파티셔닝(ARRAY_PARTITION) 설명 -----------------------------
    // [실제 버그 수정 이력 - 코드 주석에 남아있는 실전 교훈]
    // 원래 "dim=0 전체 complete 파티션"으로 작성되어 있었는데, 이는
    // SHIFT_WINDOW 루프의 ic 인덱스가 "완전히 언롤(unroll)되어 컴파일타임
    // 상수가 될 것"이라는 잘못된 가정 하에 쓰인 것이었다. 하지만 in_ch는
    // 런타임 파라미터라서 Vitis HLS는 가변 반복 횟수 루프를 완전히 언롤할 수
    // 없다 (실제 2021.1 합성 로그에서 "cannot completely unroll a loop with
    // a variable trip count" 경고로 확인됨).
    // 그래서 SHIFT_WINDOW의 언롤 방식을 "factor=TR" 부분 언롤로 바꿨고, 그에
    // 맞춰 아래 파티션도 in_ch(dim=1) 축은 "complete"이 아니라
    // "cyclic factor=TR"로 맞춘 것. (complete로 파티션해놓고 런타임 인덱스로
    // 접근하면 거대한 멀티플렉서가 생겨 "long runtime / suboptimal QoR"
    // 경고가 나는데, 바로 그 문제를 피하기 위함)
#pragma HLS ARRAY_PARTITION variable=line_buf cyclic factor=TR dim=1  // 입력채널 축: TR개씩 묶어 순환 분할
#pragma HLS ARRAY_PARTITION variable=line_buf complete dim=2          // 커널 세로(ky) 축: 완전 분할 (병렬 접근)
#pragma HLS ARRAY_PARTITION variable=line_buf complete dim=3          // 가로 위치(col) 축: 완전 분할
#pragma HLS ARRAY_PARTITION variable=window cyclic factor=TR dim=1    // window도 line_buf와 동일한 이유로 동일하게 분할
#pragma HLS ARRAY_PARTITION variable=window complete dim=2
#pragma HLS ARRAY_PARTITION variable=window complete dim=3

ROW_LOOP:
    // 패딩 포함 세로(r) 방향 래스터 스캔. r=0..pad_h-1
    for (unsigned r = 0; r < pad_h; r++) {
        // 합성 시 루프 반복 횟수를 툴에게 힌트로 알려줌 (실제 실리콘 동작과
        // 무관, 리포트/추정용 트립카운트)
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_IMG_W+2
COL_LOOP:
        // 패딩 포함 가로(c) 방향 스캔. r,c 조합으로 패딩된 좌표계의 모든
        // 픽셀 위치를 한 번씩 방문
        for (unsigned c = 0; c < pad_w; c++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_IMG_W+2
            // [설계 의도] 이 COL_LOOP 레벨에는 일부러 PIPELINE을 걸지 않음.
            // 병렬성을 담당하는 지점은 이 레벨이 아니라, 아래 MAC_IC 리덕션
            // 루프(UNROLL factor=TR)이다. 만약 COL_LOOP에 파이프라인을 걸면
            // 툴이 그 안에 중첩된 모든 것을 완전 언롤해야 하는데, 이는 이
            // 엔진이 지향하는 "DSP 사용량 상한이 정해진(bounded)" 설계를
            // 깨뜨린다.

            // 현재 (r,c)가 패딩이 아니라 "진짜 이미지 안쪽" 좌표인지 판별
            bool in_bounds = (r >= pad) && (r < pad + img_h) &&
                              (c >= pad) && (c < pad + img_w);
            // 패딩 오프셋을 뺀 실제 이미지 좌표 (in_bounds가 false일 때는
            // 아래에서 이 값을 실제로 사용하지 않으므로 언더플로되어도 무방)
            unsigned in_r = r - pad;
            unsigned in_c = c - pad;

            // 이번 사이클에 읽어올 모든 입력채널의 픽셀값을 담을 임시 배열
            act_t px[MAX_IN_CH];
#pragma HLS ARRAY_PARTITION variable=px cyclic factor=TR dim=1  // 아래 READ_CH/MAC_IC와 동일한 TR 단위로 병렬 접근되게 분할

READ_CH:
            // [중요 - 실제 회귀(regression) 버그 수정 이력]
            // conv_layer1.cpp 원본은 여기서 삼항연산자(?:)가 아니라
            // if/else를 쓴다 - C 시뮬레이션에서는 둘이 동일한 결과를 내지만,
            // "합성된 하드웨어"에서는 다르다. 삼항연산자는 보통 "양쪽 값을
            // 모두 계산한 뒤" 마지막에 mux로 고르는 형태로 합성되는데, 이
            // 코드에서 "양쪽 다 계산한다"는 것은 곧 in_r/in_c가 언더플로되어
            // 거대한 값이 된 상태로도 ifmap[...] 주소 계산식이 매 사이클
            // "실제 AXI 읽기 트랜잭션"으로 나갈 수 있다는 뜻 (결과만 버려질
            // 뿐, 버스 트랜잭션 자체는 발생). if/else는 "조건부 선택"이
            // 아니라 "조건부 실행"이 되므로, 패딩 위치에서는 애초에
            // 말도 안 되는 주소로 버스 트랜잭션이 나가지 않도록 보장한다.
            //
            // [또 다른 실제 합성 로그 기반 수정] 이 루프는 원래
            // `#pragma HLS UNROLL`(완전 언롤)이었는데, (a) in_ch가 런타임
            // 값이라 애초에 완전 언롤이 불가능했고 (b) 설령 됐더라도, m_axi
            // 포트에서 읽는 루프에는 잘못된 선택이었다 - LOAD_WTILE의
            // weights[] 읽기(버스트 결합이 확인된 패턴)는 UNROLL이 아니라
            // 순차적인 PIPELINE 구조이고, 툴의 버스트 추론(burst inference)이
            // 인식하도록 설계된 패턴이 바로 그것이다. 여기도 같은 패턴으로
            // 맞춰서 두 문제를 동시에 해결함.
            for (unsigned ic = 0; ic < in_ch; ic++) {
#pragma HLS PIPELINE II=1
                if (in_bounds) {
                    // NHWC 레이아웃: ((row*width + col) * channels) + ic
                    px[ic] = ifmap[((unsigned)in_r * img_w + in_c) * in_ch + ic];
                } else {
                    // 이미지 바깥(패딩 영역)은 0으로 채움 (zero-padding)
                    px[ic] = 0;
                }
            }

SHIFT_WINDOW:
            // [실제 합성 로그 기반 수정 - READ_CH와 동일한 사유]
            // 원래 `#pragma HLS UNROLL`(완전)이었는데 in_ch가 런타임 값이라
            // 완전 언롤 불가 (같은 "variable trip count" 경고). 이 루프는
            // READ_CH와 달리 m_axi 포트를 건드리지 않는 순수 온칩
            // window/line_buf 시프트 연산이라, PIPELINE으로 낮추는 대신
            // MAC_IC에서 이미 검증된 것과 동일한 "factor=TR 부분 언롤"을
            // 적용 - 시프트 연산의 TR-way 병렬성을 유지하면서, 위에서
            // window/line_buf를 complete가 아닌 cyclic(TR)로 바꾼 것과
            // 정확히 짝을 맞춘 것.
            for (unsigned ic = 0; ic < in_ch; ic++) {
#pragma HLS UNROLL factor=TR
                // --- 이 채널(ic)의 k x k 윈도우를 "왼쪽으로 한 칸씩 밀어서"
                //     갱신하는 과정 (새 열이 오른쪽 끝에 들어오는 구조) ---
                for (unsigned ky = 0; ky < k - 1; ky++) {
                    // 아래쪽 (k-1)개 행: 각 행 안에서 가로로 한 칸씩 시프트
                    for (unsigned kx = 0; kx < k - 1; kx++) {
                        window[ic][ky][kx] = window[ic][ky][kx + 1];
                    }
                    // 그 행의 맨 오른쪽 칸은 line_buf에 저장돼 있던 값으로 채움
                    window[ic][ky][k - 1] = line_buf[ic][ky][c];
                }
                // 마지막(가장 아래) 행도 마찬가지로 가로 시프트
                for (unsigned kx = 0; kx < k - 1; kx++) {
                    window[ic][k - 1][kx] = window[ic][k - 1][kx + 1];
                }
                // 마지막 행의 맨 오른쪽 칸은 방금 READ_CH에서 읽은 "현재
                // 픽셀"(px[ic])로 채움 - 이게 곧 새로 스캔에 들어온 값
                window[ic][k - 1][k - 1] = px[ic];

                // [self-review에서 발견한 버그 수정 - 툴 실행이 아니라 코드
                //  검토로 찾아낸 것]
                // conv_layer1.cpp 원본의 `for (ky=0; ky<K-2; ky++)`는 K=3이
                // 컴파일타임 상수인 그 파일에서만 안전하다 (K-2=1, 항상
                // 음수가 아님). 여기서는 k가 런타임 값이고, config C
                // (conv_engine_tb.cpp)가 일부러 k=1로 설정하는 테스트가
                // 있다 - k=1일 때 "k-2"를 부호 없는(unsigned) 타입에서
                // 계산하면 음수로 안 내려가고 거대한 값으로 언더플로되어,
                // 사실상 무한에 가까운 루프가 되어 line_buf 범위를 한참
                // 벗어나 인덱싱하게 된다. 그래서 이 블록 전체(루프 + 그
                // 다음에 이어지는 write)를 "k>=2일 때만" 실행하도록
                // 가드했다 - 단순히 write만 막는 게 아니라 블록 전체를 막는
                // 게 맞는 이유는, k=1인 레이어는 애초에 라인버퍼 행이 0개
                // 필요하기 때문 (그게 "우회책"이 아니라 "정답"인 이유).
                if (k >= 2) {
                    // 라인버퍼도 한 칸씩 위로 밀어올림 (다음 행을 위한 준비)
                    for (unsigned ky = 0; ky < k - 2; ky++) {
                        line_buf[ic][ky][c] = line_buf[ic][ky + 1][c];
                    }
                    // 라인버퍼의 마지막 슬롯은 현재 픽셀로 채움
                    line_buf[ic][k - 2][c] = px[ic];
                }
            }

            // 윈도우가 "가득 찬" 시점부터만 실제 컨볼루션 출력을 계산할 수
            // 있음 (스캔 시작 직후 (k-1)만큼은 아직 윈도우가 다 안 채워짐)
            if (r >= k - 1 && c >= k - 1) {
                // 패딩 좌표계(r,c)를 실제 출력 좌표계(out_r,out_c)로 변환
                unsigned out_r = r - (k - 1);
                unsigned out_c = c - (k - 1);

PE_LOOP:
                // pe: 이번 oc_tile 안에서 병렬로 계산할 출력채널 레인
                // (PE_OC개를 전부 UNROLL - 완전 병렬)
                for (unsigned pe = 0; pe < PE_OC; pe++) {
#pragma HLS UNROLL
                    // 이 레인이 실제로 담당하는 절대 출력채널 인덱스
                    unsigned oc = oc_tile * PE_OC + pe;
                    // out_ch가 PE_OC의 배수가 아닐 때, 범위 밖 레인은 계산
                    // 자체를 스킵 (하드웨어는 여전히 존재하지만 결과를 안 씀)
                    if (oc < out_ch) {
                        // 누산기를 이 채널의 편향값으로 초기화 (bias add)
                        accum_t acc = btile[pe];

                        // [TR-way 병렬 리덕션 설명]
                        // 바깥쪽 ic 루프에 UNROLL factor=TR을 걸어서, 안쪽의
                        // ky/kx 파이프라인 리덕션 회로를 TR개 복제하고, 각
                        // 복제본이 서로 다른 입력채널을 담당하게 한다. 그
                        // 결과들은 loop-carried reduction 변수인 acc로
                        // 합쳐지는데, 이건 HLS가 자동으로 만들어주는
                        // "덧셈 트리(adder tree)"를 통해 이루어진다.
                        // 이 레인 하나의 MAC 연산에 쓰이는 총 DSP 개수: TR개
                        // (RESOURCE_BUDGET.md §2-3 참고).
                        // in_ch는 런타임에 MAX_IN_CH 이하로 결정되므로, HLS는
                        // "TR폭짜리 하드웨어"를 한 번만 만들어두고,
                        // ceil(in_ch/TR)번 반복 실행하는 식으로 처리한다
                        // (TR로 안 나누어떨어지는 나머지(remainder)까지
                        // 자동으로 처리 - conv_engine_tb.cpp의 엣지케이스
                        // 설정들이 바로 이 나머지 처리를 검증하기 위한 것).
MAC_IC:
                        for (unsigned ic = 0; ic < in_ch; ic++) {
#pragma HLS UNROLL factor=TR
MAC_KY:
                            for (unsigned ky = 0; ky < k; ky++) {
MAC_KX:
                                for (unsigned kx = 0; kx < k; kx++) {
                                    // 이 가장 안쪽 (ky,kx) 곱셈-누적 자체는
                                    // 파이프라인 II=1로 - 매 사이클 하나의
                                    // MAC(곱셈+누적)을 수행
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_K*MAX_K
                                    // 핵심 MAC 연산: window(활성값) x
                                    // wtile(가중치)을 32비트로 확장해 곱하고
                                    // acc에 누적. accum_t로 캐스팅해서
                                    // 8bit*8bit 곱셈 오버플로를 방지.
                                    acc += (accum_t)window[ic][ky][kx] *
                                           (accum_t)wtile[pe][ic][ky][kx];
                                }
                            }
                        }

                        // [양자화/스케일링 설명]
                        // 누산값이 음수가 아니면 그냥 saturate()로 8비트에
                        // 맞게 자르고, 음수면 오른쪽으로 3비트 시프트(>>3,
                        // 즉 8로 나눔)한 뒤 saturate() - 이는 이 컨볼루션
                        // 엔진이 사용하는 (비대칭) 고정 스케일/활성화 방식으로
                        // 보이며, 음수 쪽에 대해 별도의 스케일 다운을 적용하는
                        // 단순화된 활성화 처리를 하드웨어 안에서 수행하는
                        // 부분이다.
                        act_t result = (acc >= 0) ? saturate(acc)
                                                   : saturate(acc >> 3);
                        // 계산 결과를 NHWC 레이아웃으로 ofmap DDR 버퍼에 기록
                        ofmap[(out_r * out_w + out_c) * out_ch + oc] = result;
                    }
                }
            }
        }
    }
}

// =============================================================================
// conv_engine(): 최상위(top-level) HLS 함수. Vivado/Vitis에서 IP로 패키징되는
// 실제 진입점(entry point). SW(network_run.c 등)가 이 함수의 인자들을
// AXI-Lite로 세팅하고 ap_start를 올려서 매 레이어마다 재기동시킨다.
// =============================================================================
void conv_engine(
    const act_t    *ifmap,    // [img_h][img_w][in_ch], NHWC, DDR에 있는 입력 특성맵
    const weight_t *weights,  // [out_ch][in_ch][k][k], DDR에 있는 가중치
    const weight_t *bias,     // [out_ch], DDR에 있는 편향
    act_t          *ofmap,    // [out_h][out_w][out_ch], NHWC, DDR에 쓸 출력 특성맵
    uint16_t img_h, uint16_t img_w, uint16_t in_ch, uint16_t out_ch,
    uint8_t  k, uint8_t stride, uint8_t pad
) {
    // ---- AMBA 인터페이스 프라그마 -------------------------------------------
    // 4개의 DDR 포인터를 m_axi(AXI4 마스터, 버스트 가능) 인터페이스로 노출.
    // ifmap/weights/bias는 읽기 전용이라 같은 번들(RD_BUS)로 묶어 같이
    // 스케줄링되게 하고, ofmap은 쓰기 전용이라 별도 번들(WR_BUS)로 분리
    // (읽기/쓰기가 서로 다른 AXI 채널이라 독립적으로 동시에 동작 가능).
    // depth=... 는 각 포트가 최대 얼마나 접근할 수 있는지(주소 범위) 툴에게
    // 알려주는 값 - 위 헤더의 MAX_* 상수들로부터 계산됨.
#pragma HLS INTERFACE m_axi port=ifmap   offset=slave bundle=RD_BUS depth=MAX_IMG_W*MAX_IMG_W*MAX_IN_CH
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=RD_BUS depth=MAX_OUT_CH*MAX_IN_CH*MAX_K*MAX_K
#pragma HLS INTERFACE m_axi port=bias    offset=slave bundle=RD_BUS depth=MAX_OUT_CH
#pragma HLS INTERFACE m_axi port=ofmap   offset=slave bundle=WR_BUS depth=MAX_IMG_W*MAX_IMG_W*MAX_OUT_CH

    // ---- 나머지 스칼라 인자들 + 4개 포인터의 "베이스 주소"는 전부
    //      AXI4-Lite(s_axilite)로, 같은 CTRL 번들에 묶어서 노출.
    //      SW가 이 레지스터들에 값을 써넣고 ap_start를 트리거하는 방식으로
    //      매 레이어마다 이 IP를 재구성/재기동시킨다.
#pragma HLS INTERFACE s_axilite port=ifmap   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=weights bundle=CTRL
#pragma HLS INTERFACE s_axilite port=bias    bundle=CTRL
#pragma HLS INTERFACE s_axilite port=ofmap   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=img_h   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=img_w   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=in_ch   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=out_ch  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=k       bundle=CTRL
#pragma HLS INTERFACE s_axilite port=stride  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=pad     bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return  bundle=CTRL
    // (return도 CTRL 번들에 - ap_start/ap_done/ap_idle 등 제어 신호가
    // 여기서 함께 노출됨)

    // ---- 방어적 범위 체크 (시뮬레이션에서만 동작) --------------------------
    // __SYNTHESIS__는 Vitis HLS가 "C/RTL 합성 중"일 때만 정의해주는 내장
    // 매크로이고, C-시뮬레이션 때는 정의되지 않는다 - 그래서 아래 assert들은
    // "시뮬레이션에서만 도는 안전장치, 합성 시엔 하드웨어 비용 0"이라는
    // 표준 관용구다.
    // 이 체크들이 존재하는 이유: MAX_IN_CH/MAX_K는 아직 확정 전 placeholder
    // 값이고(conv_engine.h 참고), stride>1은 아직 미구현이라, SW 쪽에서
    // 설정을 잘못 넣으면 "조용히 범위를 벗어나 인덱싱"하거나 "stride>1을
    // 요청했는데 조용히 stride=1 결과를 계산"해버릴 수 있다 - 그런 상황을
    // 시뮬레이션 단계에서 "값싸게, 시끄럽게" 잡아내기 위함.
#ifndef __SYNTHESIS__
    assert(img_h <= MAX_IMG_W && "img_h exceeds MAX_IMG_W - see conv_engine.h");
    assert(img_w <= MAX_IMG_W && "img_w exceeds MAX_IMG_W - see conv_engine.h");
    assert(in_ch <= MAX_IN_CH && "in_ch exceeds MAX_IN_CH - see README design limits");
    assert(out_ch <= MAX_OUT_CH && "out_ch exceeds MAX_OUT_CH - see conv_engine.h");
    assert(k <= MAX_K && k >= 1 && "k out of [1, MAX_K] range");
    assert(stride == 1 && "only stride=1 implemented - see README design limits");
    assert(pad <= MAX_K && "implausible pad value");
    // scan_and_compute()의 out_w/pad_h/pad_w는 img_w+2*pad-k+1 같은 식으로
    // "부호 없는(unsigned)" 타입으로 계산된다 - 만약 패딩을 더한 이미지가
    // 커널보다 작다면 이 뺄셈이 음수로 내려가는 대신 언더플로된다. 실제
    // 네트워크 레이어에서는 있을 수 없는 상황(특성맵은 항상 1x1/3x3
    // 커널보다 훨씬 크다)이지만, "암묵적 가정"으로 남겨두는 것보다 명시적
    // assert로 만드는 게 비용이 거의 안 드니 이렇게 처리.
    assert((unsigned)img_h + 2u * pad >= k && "img_h+2*pad must be >= k");
    assert((unsigned)img_w + 2u * pad >= k && "img_w+2*pad must be >= k");
#else
    (void)0; // 합성 시에는 완전히 사라지는 no-op (위 assert 블록 전체가 컴파일에서 빠짐)
#endif
    (void)stride; // stride는 위 assert로만 강제되고, 실제로 분기 하드웨어를 만들진 않음 (사용 안 함 경고 방지용 캐스트)

    // 출력채널을 PE_OC(=16)개씩 묶었을 때 몇 개의 "타일"이 필요한지 계산
    // (올림 나눗셈: out_ch가 PE_OC의 배수가 아니어도 마지막 타일까지 커버)
    const unsigned num_oc_tiles = ((unsigned)out_ch + PE_OC - 1) / PE_OC;

    // wtile/btile을 load_weight_tile/scan_and_compute 내부가 아니라 여기
    // conv_engine()에서 "소유"하는 이유: 나중에 더블 버퍼링 버전을 만들 때
    // 이 두 배열만 복제하고 아래 루프 바디에 DATAFLOW 프라그마 하나만
    // 추가하면 되도록 하기 위함 (RESOURCE_BUDGET.md §5).
    static weight_t wtile[PE_OC][MAX_IN_CH][MAX_K][MAX_K]; // 현재 oc_tile의 가중치 캐시
    static weight_t btile[PE_OC];                          // 현재 oc_tile의 편향 캐시

    // wtile 파티셔닝: dim=1(pe, 출력채널 레인)은 PE_OC개를 완전 분할 -
    // PE_LOOP에서 pe를 완전 UNROLL하므로 각 레인이 자기 몫의 wtile 슬라이스에
    // 동시에 접근해야 함. dim=2(in_ch)는 TR 단위 순환 분할 - MAC_IC가
    // factor=TR로 부분 언롤되는 것과 짝을 맞춤. dim=3,4(ky,kx)는 커널 크기가
    // 작아(최대 3x3) 완전 분할해도 비용이 적고, MAC_KY/MAC_KX 안쪽 파이프라인이
    // 이 축들에 매 사이클 접근해야 하므로 완전 분할.
#pragma HLS ARRAY_PARTITION variable=wtile complete dim=1
#pragma HLS ARRAY_PARTITION variable=wtile cyclic factor=TR dim=2
#pragma HLS ARRAY_PARTITION variable=wtile complete dim=3
#pragma HLS ARRAY_PARTITION variable=wtile complete dim=4
#pragma HLS ARRAY_PARTITION variable=btile complete dim=0 // btile은 PE_OC개뿐이라 전체를 완전 분할

OC_TILE:
    // 출력채널을 PE_OC개씩 묶은 "타일" 단위로 전체 계산을 나눠서 반복.
    // 한 타일 = "이번에 병렬로 계산할 PE_OC개의 출력채널" 하나의 묶음.
    for (unsigned oc_tile = 0; oc_tile < num_oc_tiles; oc_tile++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
        // [알려진, 의도된 트레이드오프]
        // 이 구조에서는 ifmap을 "oc_tile마다 한 번씩" DDR에서 다시 읽는다
        // (즉, 전체 이미지를 num_oc_tiles번 다시 스캔). 이는 "DDR 대역폭을
        // 더 쓰는 대신, 온칩 面적을 bounded하게 유지"하는 명시적 트레이드오프
        // (RESOURCE_BUDGET.md §2 참고). 이 재스캔 오버헤드가 바로
        // "엔진 효율(engine efficiency)" 감쇠 계수가 (실측 cosim 트레이스
        // 없이) 대략적으로 반영하려는 부분이다.

        // 1) 이번 oc_tile에 해당하는 가중치/편향을 DDR에서 온칩으로 로드
        load_weight_tile(weights, bias, oc_tile, in_ch, out_ch, k, wtile, btile);
        // 2) 로드된 wtile/btile을 가지고 입력 특성맵 전체를 한 번 스캔하며
        //    이번 oc_tile 몫의 출력채널들을 계산해 ofmap에 씀
        scan_and_compute(ifmap, ofmap, img_h, img_w, in_ch, out_ch, k, pad,
                          oc_tile, wtile, btile);
    }
    // 모든 oc_tile을 다 돌면 이 레이어의 컨볼루션 계산이 끝남 - ap_done이
    // 올라가고, SW는 다음 레이어의 파라미터로 다시 이 함수를 재기동시킨다.
}

// =============================================================================
// [전체 흐름 요약]
//
// conv_engine()
//   for oc_tile in 0..num_oc_tiles-1:      // 출력채널을 PE_OC개씩 묶어서 반복
//       load_weight_tile()                  // 이 타일의 가중치/편향을 DDR->온칩
//       scan_and_compute()                  // 입력 특성맵 전체를 한 번 스캔:
//           for r in 0..pad_h-1:                //   패딩 포함 세로 스캔
//               for c in 0..pad_w-1:             //   패딩 포함 가로 스캔
//                   px[] <- ifmap 읽기 (또는 0)      // 현재 픽셀 로드 (패딩 처리)
//                   window[]/line_buf[] 시프트 갱신    // 슬라이딩 윈도우 갱신
//                   if 윈도우가 다 찼으면:
//                       for pe in 0..PE_OC-1 (완전 병렬):
//                           acc = bias
//                           for ic (TR-way 병렬), ky, kx:
//                               acc += window * weight   // MAC
//                           ofmap <- saturate(acc)        // 8비트로 양자화 후 기록
//
// 병렬성 요약: 한 번에 PE_OC(16)개 출력채널 x TR(16)개 입력채널 = 256개의
// INT8 MAC이 동시에 동작 (KR260 DSP 예산의 약 20.5%).
// =============================================================================
