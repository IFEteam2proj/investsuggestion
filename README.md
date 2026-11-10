# Black-Litterman Portfolio Optimization with Factor-Alpha Screening and Machine Learning Views

**한양대학교 금융공학개론 (Introduction to Financial Engineering) · Team #2 연구 프로젝트 아카이브**

이 리포지토리는 **Fama-French 3-Factor 알파 스크리닝**, **Expanding-Window 머신러닝 수익률 예측(5개 모델)**, 그리고 **블랙-리터만 뷰 통합(8개 뷰 구성법)**을 결합한 4단계 포트폴리오 최적화 프로젝트의 전체 과정, 실증 백테스트(2011–2020), 동료 평가 Q&A 21문항 검증 및 미관측 표본 외(2021–2025) 테스트 결과를 정리한 GitHub Pages 정적 웹사이트입니다.

---

## 📂 웹사이트 페이지 구성 (`pagedeploy/`)

| 파일명 | 페이지 제목 | 주요 수록 내용 |
| :--- | :--- | :--- |
| [`index.html`](./index.html) | **메인 대시보드 (Executive Summary)** | 핵심 성과 KPI, 누적 수익률 및 낙폭 인터랙티브 차트, 4단계 파이프라인 요약, 초과수익 3단계 분해, 연구진 역할 분담 |
| [`methodology.html`](./methodology.html) | **방법론 & 4단계 파이프라인** | CRSP/FF3/FRED 전처리 규약, FF3 알파 스크리닝 수식 및 Appraisal Ratio 동치성 증명, 선정된 20개 종목 검색 테이블, 5개 ML 모델 및 8개 특성 정의, 블랙-리터만 베이지안 유도 및 Idzorek $\tau$-불변성 증명 |
| [`results.html`](./results.html) | **백테스트 성과 분석 (2011–2020)** | 40개(5×8) 전략 조합 인터랙티브 히트맵(CAGR·Sharpe·Turnover 전환), 연도별 수익률, 4대 시장 국면 및 7대 역사적 에피소드 분석, 120개월 종목별 비중 동학, 강건성·민감도 분석 |
| [`qa-analysis.html`](./qa-analysis.html) | **Q&A 심층 검증 & 2021–2025 OOS** | 17개 대안 설정 마스터 비교표, $\delta=8.84$ 편향 규명 및 시장 내재 $\delta$ 추정, 40개 조합 다중검정 보정(Bonferroni, Holm, BH, White Reality Check), 연간 재선정 실험, 2021–2025 미관측 OOS 백테스트, **21개 동료 질문 전수 아코디언** |
| [`meetings.html`](./meetings.html) | **회의록 & 프로젝트 히스토리** | 교수님 1·2·3차 지도 면담에 따른 5대 설계 피벗 요약, 9월 7일~11월 7일 총 9차례 회의록 아카이브, 타 팀(Team 1,3,4,5,6,7,8) 동료 평가 요약 |
| [`code-papers.html`](./code-papers.html) | **코드 아키텍처 & 참고문헌** | 국·영문 최종 보고서 및 Q&A 답안지 PDF 다운로드 센터, 전체 코드베이스 디렉토리 구조 및 CLI 재현 명령어, 29편의 학술 참고문헌 라이브러리 |

---

## 🚀 GitHub Pages 배포 방법

1. 본 `pagedeploy` 폴더 내부의 모든 파일(`index.html`, `methodology.html`, `results.html`, `qa-analysis.html`, `meetings.html`, `code-papers.html`, `.nojekyll`, `assets/`)을 `<username>.github.io` 리포지토리 루트(또는 프로젝트 리포지토리의 `main` 브랜치)에 업로드(Push)합니다.
2. 리포지토리 **Settings → Pages → Build and deployment**에서 Branch를 `main` / `/(root)`로 설정하면 즉시 웹사이트가 활성화됩니다.
