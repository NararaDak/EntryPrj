# Best Team One - 알약 객체 탐지 프로젝트

> AI 6기 1팀 - 컴퓨터 비전 기반 약물 자동 인식 시스템

[![Streamlit App](https://img.shields.io/badge/Streamlit-Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://entryprj-rbrmekfkjfm5ldqvcqofgc.streamlit.app/)
[![Project Report](https://img.shields.io/badge/Project-Report-blue?style=for-the-badge&logo=adobe-acrobat-reader&logoColor=white)](https://github.com/NararaDak/EntryPrj/blob/main/files/Best_One_Team_project_Total.pdf)

---

## 프로젝트 소개

### 배경
컴퓨터 비전 학습 과정에서 딥러닝 기반 이미지 분류 및 객체 인식 기술을 습득하고, 실제 문제에 적용하여 최적의 모델을 선정하고 파이프라인을 개발하는 것을 목표로 합니다.

### 목표
- 미션에 부합하는 최적의 모델 선정 및 파이프라인 개발 경험 축적
- 약물 이미지 자동 탐지 시스템 구축
- 스트림릿을 통한 약물 정보 제공

### 기대 효과
- 실무 경험 축적 및 수업 이외 지식 습득
- 분석 기술 및 협업 기술 향상

---

## 설치 및 실행

### 1. 저장소 클론

```bash
git clone https://github.com/NararaDak/EntryPrj.git
cd EntryPrj
```

### 2. 가상환경 생성 및 활성화

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. Streamlit 앱 실행

```bash
streamlit run src/streamitService.py
```

---

## 프로젝트 구조

```
EntryPrj/
├── src/
│   ├── streamitService.py          # Streamlit 메인 앱
│   ├── A04.py                      # 핵심 파이프라인 (학습 실행)
│   ├── run_evaluation.py           # 평가 시스템
│   ├── streamit_*.py               # Streamlit 탭별 모듈
│   ├── eda*.py                     # 데이터 분석 스크립트
│   └── yolo_drug_info.py           # 약물 정보 처리
├── data/
│   ├── traindata/                  # 학습 데이터
│   │   └── train_annotations/      # 어노테이션 파일
│   └── test_images/                # 테스트 데이터
├── files/
│   └── Best_One_Team_project_Total.pdf  # 프로젝트 보고서
├── requirements.txt                # 패키지 의존성
└── README.md                       # 프로젝트 문서
```

---

## 결과 시연

### 웹 애플리케이션 데모

실시간 약물 탐지 및 정보 제공 서비스를 아래 링크에서 확인하실 수 있습니다.

[Streamlit 웹 애플리케이션 바로가기](https://entryprj-rbrmekfkjfm5ldqvcqofgc.streamlit.app/)

### 주요 기능

1. **EDA Tool 시연**: 이미지와 어노테이션 매핑 분석
2. **모델 시연**: 실시간 약물 탐지 및 분류
3. **GPT 약물 분석**: 탐지된 약물의 상세 정보 및 복용 가이드 제공
4. **테스트 결과 분석**: Submission 데이터 검증 및 성능 평가

### 성능 지표

- **mAP@[0.75:0.95]**: 주요 평가 지표
- **다양한 약물 형태 지원**: 정제, 캡슐, 필름제 등
- **신뢰도 기반 필터링**: 0.5 이상의 신뢰도를 가진 객체만 표시

---

## 팀원 및 협업

### Best Team One (AI 6기 1팀)

- 김영욱
- 김효중
- 장우정 
- 최무영
- 최지영

### 협업 일지

- **김영욱**: 
- **김효중**: [협업 일지](https://github.com/NararaDak/EntryPrj/blob/main/CollaborationLog/AI6기김영욱_초급프로젝트.pdf)
- **장우정**: [협업 일지](https://www.notion.so/AI-6-1-2c32cc364d718029ae0bd7645c6e1fcf)
- **최무영**:
- **최지영**:

---

## 보고서 자료

- [프로젝트 보고서 (PDF)](https://github.com/NararaDak/EntryPrj/blob/main/files/Best_One_Team_project_Total.pdf)
- [Streamlit 데모 사이트](https://entryprj-rbrmekfkjfm5ldqvcqofgc.streamlit.app/)

