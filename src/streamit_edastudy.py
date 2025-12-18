import streamlit as st

def display_eda_study():
    st.subheader("데이터 분석")
    st.write("이미지-어노테이션 매핑, 클래스 분포, 누락/중복 데이터 등 EDA 결과를 시각화합니다.")
    st.markdown("""
    - **이미지-어노테이션 매핑 통계**
    - **클래스별 분포 그래프**
    - **누락/중복 데이터 현황**
    - **시각화 예시**: bar, pie, scatter plot 등
    """)
    st.info("EDA 결과 그래프, 표, 통계 요약 등을 추가하세요.")
