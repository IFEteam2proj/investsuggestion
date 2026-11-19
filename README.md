# Quantitative Investment & Asset Allocation Pipeline

미국 시총 상위 80% 대형주 유니버스(319개 종목)를 대상으로 한 수익률 데이터베이스 구축, **Fama-French 3-Factor (FF3) 모형 기반 종목 선정**, 그리고 **XGBoost 머신러닝 견해(Views)를 결합한 Black-Litterman (BL) 포트폴리오 최적화 자산배분 시스템**입니다.

---

## 1. 프로젝트 전체 디렉터리 구조

```
myinvestment/
├── data/
│   ├── raw/                              # 원천 수익률 및 유니버스 데이터
│   │   ├── daily_returns_1y.csv          # 최근 1년 일별 수익률 (Long 패널, 78,893행)
│   │   ├── monthly_returns_10y.csv        # 최근 10년 월별 수익률 (Long 패널, 36,210행)
│   │   ├── annual_returns_20y.csv         # 최근 20년 연간 수익률 (Long 패널, 5,636행)
│   │   ├── daily_returns_1y_wide.csv     # 일별 수익률 매트릭스 (Wide)
│   │   ├── monthly_returns_10y_wide.csv   # 월별 수익률 매트릭스 (Wide)
│   │   ├── annual_returns_20y_wide.csv    # 연간 수익률 매트릭스 (Wide)
│   │   ├── stock_universe_top80.csv       # 시총 상위 80% 종목 메타정보 (319종목)
│   │   └── .cache/                       # 주가/거래량 Parquet 캐시 및 PERMNO 매핑 캐시
│   ├── ff3/                              # FF3 팩터 회귀 및 종목 선정 산출물
│   │   ├── ff3_selected.csv              # 게이트 통과 최종 선정 9개 종목 (t(α) 내림차순)
│   │   ├── ff3_regression_all.csv        # Balanced 유니버스 전체(284개) 팩터 회귀 및 랭킹 (1위~284위)
│   │   ├── ff3_insample_performance.csv  # 선정 포트폴리오 vs 균등가중 유니버스 vs 시장 성과 비교
│   │   └── F-F_Research_Data_Factors.csv # Kenneth French 공식 3팩터 데이터셋
│   └── bl/                               # Black-Litterman 중간 데이터
│       ├── bl_input/                     # P.csv, Q.csv, Omega.csv, assets.csv, constraints.json
│       ├── diagnostics/                  # views_timeseries.csv, predicted_returns.csv, model_fits.csv, summary.json
│       └── views_readable.csv            # 월별 롱/숏 견해 및 스프레드 예측/실현 내역 (가독형)
├── results/                              # Black-Litterman 자산배분 최종 결과
│   ├── bl_weights_latest.csv             # 최신 리밸런싱 시점 종목별 최적 자산비중 및 틸트
│   ├── bl_weights_history.csv            # 월별 자산비중 시계열 히스토리 (Walk-Forward)
│   ├── bl_portfolio_summary.csv          # 포트폴리오 성과 및 리스크 요약 지표 (연환산 수익, 변동성, 샤프, MDD)
│   └── bl_backtest_performance.csv       # 구간 내 월별 및 누적 수익률 백테스트 성과
├── scripts/                              # 실행 파이썬 스크립트 모음
│   ├── update_returns.py                 # 주가 데이터 수집 및 증분 업데이트
│   ├── run_ff3.py                        # FF3 팩터 회귀, 게이트 검증 및 종목 랭킹
│   ├── run_bl_portfolio.py               # XGBoost 견해 생성 + Black-Litterman 자산배분 통합 실행
│   └── view_module_xgboost.py            # XGBoost 견해 생성 모듈 (P, Q, Omega)
└── README.md
```

---

## 2. 모듈별 실행 방법

모든 스크립트는 `scripts/` 폴더에 있으며 커맨드라인에서 바로 실행할 수 있습니다.

### ① 주가 및 수익률 데이터 업데이트
```bash
python scripts/update_returns.py
```
- 기존 캐시 이후의 신규 거래일만 자동으로 fetch하여 약 15초 내에 `data/raw/`의 모든 CSV를 갱신합니다.

### ② FF3 팩터 회귀 및 종목 선정(9개 종목 추출)
```bash
python scripts/run_ff3.py
```
- Kenneth French 팩터를 자동 동기화하고, FF3 회귀를 수행하여 게이트를 통과한 9개 종목을 `data/ff3/ff3_selected.csv`로 저장합니다.

### ③ XGBoost 기반 Black-Litterman 자산배분 실행
```bash
python scripts/run_bl_portfolio.py
```
- **기본 파라미터**:
  - `tau` ($\tau$): `0.1` (기본값)
  - `delta` ($\delta$ / 위험회피계수): `3.0` (기본값)
  - `max-weight`: `0.20` (종목당 최대 비중 20% 상한 제약)
  - `n-long-short`: `2` (견해당 상위 2종목 롱 / 하위 2종목 숏)
- **파라미터 변경 예시**:
  ```bash
  python scripts/run_bl_portfolio.py --tau 0.05 --delta 2.5 --max-weight 0.25
  ```

---

## 3. Black-Litterman (BL) 자산배분 모델링 설계

### [1] 자산 유니버스
FF3 게이트(알파 양수, $t(\alpha) \ge 2.0$, 유동성 하위 20% 제외, 시그마 가드 통과)를 만족한 **최상위 9개 주식만으로 100% 포트폴리오를 구성**:
- `NVDA` (엔비디아), `LLY` (일라이릴리), `PWR` (콴타서비스), `PGR` (프로그레시브), `AVGO` (브로드컴), `ANET` (아리스타네트웍스), `KLAC` (KLA), `LRCX` (램리서치), `FTNT` (포티넷)

### [2] 균형 사전 기대수익률 (Prior Equilibrium Returns)
역최적화(Reverse Optimization)를 통해 시장 균형 수익률 $\Pi$ 산출:
$$\Pi = \delta \Sigma w_{mkt}$$
- $w_{mkt}$: 9개 자산의 시가총액 상대 비중 ($\sum w_{mkt, i} = 1$)
- $\Sigma$: 9개 자산의 과거 36개월 롤링 공분산 행렬
- $\delta = 3.0$: 투자자 위험회피계수

### [3] XGBoost 기반 머신러닝 견해 (Views: $P, Q, \Omega$)
1. **특성 공학 (8대 횡단면 순위 팩터)**:
   - `mom1m` (1개월 모멘텀), `mom6m` (6개월 모멘텀), `mom12m` (12개월 모멘텀), `chmom` (모멘텀 가속도)
   - `retvol` (12개월 변동성), `dolvol` (거래대금), `beta` (시장 베타), `idiovol` (고유 변동성)
2. **모델 학습 및 예측**:
   - 시간 순서 기반 검증 세트(마지막 12개월)로 최적 트리 깊이, 학습률, 트리 수 조기종료(Early Stopping) 튜닝
   - 12개월마다 재학습(Walk-Forward Refit)하며 9개 자산의 다음 달 수익률 예측값 $\mu_t$ 생성
3. **견해 행렬 구조**:
   - $P$: 예측 상위 2종목 롱($+0.5, +0.5$), 하위 2종목 숏($-0.5, -0.5$)
   - $Q$: 예측 스프레드 ($Q = P \cdot \mu$)
   - $\Omega$: 과거 24개월 견해 예측 오차($Q_{real} - Q$)의 평균제곱(MSE) 기반 불확실성 산출

### [4] 사후 결합분포 (Posterior Estimates)
마스터 블랙-리터만 공식을 통해 결합 기대수익률 $E[R]$ 및 공분산 $\Sigma_{BL}$ 산출:
$$E[R] = \Pi + \tau \Sigma P^T \left( P \tau \Sigma P^T + \Omega \right)^{-1} \left( Q - P \Pi \right)$$
$$\Sigma_{BL} = \Sigma + M, \quad M = \left[ (\tau \Sigma)^{-1} + P^T \Omega^{-1} P \right]^{-1}$$

### [5] 최적 자산비중 산정 (Quadratic Programming)
$$\max_w \quad w^T E[R] - \frac{\delta}{2} w^T \Sigma_{BL} w$$
$$\text{subject to} \quad \sum_{i=1}^9 w_i = 1, \quad 0 \le w_i \le w_{max} \ (20\%)$$

---

## 4. 최종 결과 및 성과 요약

### ① 최신 시점 종목별 최적 자산비중 (`results/bl_weights_latest.csv`)

| Ticker | 기업명 | 시가총액 비중($w_{mkt}$) | BL 제약 최적비중($w_{BL}$) | 틸트($w_{BL} - w_{mkt}$) | 사전 기대수익률($\Pi$) | BL 사후 기대수익률($E[R]$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **NVDA** | NVIDIA Corp. | 57.19% | **20.00%** | -37.19% | 31.49% | 30.69% |
| **AVGO** | Broadcom Inc. | 17.86% | **18.35%** | +0.48% | 24.28% | 23.04% |
| **ANET** | Arista Networks | 2.83% | **16.97%** | +14.15% | 21.99% | 21.31% |
| **LLY** | Eli Lilly & Co. | 11.49% | **13.96%** | +2.48% | 7.88% | 8.33% |
| **PGR** | Progressive Corp. | 1.31% | **9.59%** | +8.28% | -3.07% | -2.64% |
| **LRCX** | Lam Research | 4.13% | **9.15%** | +5.02% | 20.86% | 21.67% |
| **PWR** | Quanta Services | 1.08% | **6.54%** | +5.47% | 15.40% | 15.17% |
| **FTNT** | Fortinet Inc. | 1.48% | **5.43%** | +3.96% | 7.52% | 8.66% |
| **KLAC** | KLA Corp. | 2.64% | **0.00%** | -2.64% | 13.92% | 15.09% |

> **분석 포인트**: 시가총액 가중 시 엔비디아(NVDA) 1종목에 57.2%가 쏠리는 극단적 집중 위험을 **최대 비중 20% 제약 및 ML 견해를 통해 완화**하고, 높은 위험조정 수익률을 지닌 브로드컴(AVGO), 아리스타네트웍스(ANET), 일라이릴리(LLY), 프로그레시브(PGR) 등으로 효과적인 분산 배분을 달성했습니다.

### ② 구간 외 실증 백테스트 성과 비교 (`results/bl_portfolio_summary.csv`)
2021년 1월 ~ 2026년 8월 (총 67개월 Walk-Forward 성과):

| 포트폴리오 | 연환산 수익률 | 연환산 변동성 | 샤프지수 (Sharpe) | 최대 낙폭 (MDD) | 누적 수익률 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Black-Litterman 제약 (Cap 20%)** | **48.02%** | **27.35%** | **1.638** | **-26.56%** | **1,040.9% (10.4배)** |
| Black-Litterman 무제약 | 53.46% | 35.19% | 1.428 | -44.21% | 1,245.5% |
| 시가총액 가중 벤치마크 | 54.68% | 35.33% | 1.457 | -43.64% | 1,333.8% |
| 동일가중 벤치마크 (1/9) | 43.54% | 23.57% | 1.711 | -21.91% | 841.2% |

> **위험관리 효과**: 시가총액 가중 포트폴리오 대비 **변동성을 35.3%에서 27.4%로 낮추고**, **최대 낙폭(MDD)을 -43.6%에서 -26.6%로 대폭 방어**하면서도 연환산 48.0%의 우수한 복리 수익률을 기록했습니다.
