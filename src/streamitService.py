import streamlit as st
import streamit_summary
import streamit_modelStudy
import streamit_edastudy
import streamit_edaexecute
import streamit_modelexecute
import streamit_submission

# 페이지 전체 레이아웃 및 폰트/탭 크기 스타일 적용
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        font-size: 1.5rem;
        min-height: 3.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1.2rem 2.5rem 1.2rem 2.5rem;
        font-size: 1.3rem;
    }
    html, body, [class*="css"]  {
        font-size: 1.15rem;
    }
    .stApp {
        padding: 0rem 2rem 2rem 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ════════════════════════════════════════
# ▣ 01. 디렉토리 및 유틸 함수 설정
# ════════════════════════════════════════
# .venv\Scripts\Activate
# Streamlit 탭 인터페이스 설정
# streamlit run D:\01.project\EntryPrj\src\streamitService.py

# 1. 탭 제목 리스트를 정의합니다.

tab_titles = [
    "프로젝트개요",
    "모델 연구",
    "데이터 분석",
    "EDA tool 시연",
    "모델 시연",
    "테스트 결과 분석",
]

# 2. st.tabs() 함수를 사용하여 탭을 생성합니다.
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_titles)

# 3. 각 탭 객체(tab1, tab2, tab3, tab4, tab5) 내부에 콘텐츠를 배치합니다.
with tab1:
    streamit_summary.display_summary()

with tab2:
    st.header("모델 연구 탭")
    try:
        streamit_modelStudy.display_model_study()
    except Exception:
        st.info("streamit_modelStudy.py의 display_model_study()를 구현하세요.")

with tab3:
    st.header("데이터 분석 탭")
    try:
        streamit_edastudy.display_eda_study()
    except Exception:
        st.info("streamit_edastudy.py의 display_eda_study()를 구현하세요.")

with tab4:
    st.header("EDA tool 시연 탭")
    try:
        # streamit_eda.display_eda()
        streamit_edaexecute.display_eda()
    except Exception:
        st.info("streamit_edaexecute.py의 display_eda()를 구현하세요.")

with tab5:
    st.header("모델 시연 탭")
    try:
        streamit_modelexecute.display_model()
    except Exception:
        st.info("streamit_modelexecute.py의 display_model()를 구현하세요.")

with tab6:
    st.header("테스트 결과 분석 탭")
    try:
        streamit_submission.display_submission_study()
    except Exception:
        st.info("streamit_submission.py의 display_submission_study()를 구현하세요.")
