import streamlit as st
import time

# 페이지 설정
st.set_page_config(page_title="Sidebar Fragment 제어", layout="wide")

st.title("🛰️ 사이드바 제어 및 Fragment 렌더링")
st.write(f"🏠 메인 페이지 전체 렌더링 시간: **{time.strftime('%H:%M:%S')}**")

# --- 1. 상태(State) 초기화 ---
if 'frag1_counter' not in st.session_state:
    st.session_state.frag1_counter = 0

# --- 2. Fragment 정의 ---
@st.fragment
def simple_fragment1():
    st.subheader("📍 구간 1: 대시보드 모드")
    st.info(f"외부/내부 업데이트 합계: {st.session_state.frag1_counter}")
    st.write(f"⏱️ 구간 1 내부 시간: {time.strftime('%H:%M:%S')}")
    
    if st.button("⚡ 구간 1 내부만 새로고침"):
        # 이 버튼은 simple_fragment1 함수만 다시 실행시킵니다.
        pass

@st.fragment
def simple_fragment2():
    st.subheader("📍 구간 2: 리포트 모드")
    st.warning("이 구간은 외부 카운터의 영향을 직접 받지 않습니다.")
    st.write(f"⏱️ 구간 2 내부 시간: {time.strftime('%H:%M:%S')}")
    
    if st.button("⚡ 구간 2 내부만 새로고침"):
        # 이 버튼은 simple_fragment2 함수만 다시 실행시킵니다.
        pass

# --- 3. 사이드바 제어 영역 ---
with st.sidebar:
    st.header("🎮 컨트롤 패널")
    
    # 셀렉트박스를 이용한 메뉴 선택 (session_state를 자동으로 관리)
    selected_view = st.selectbox(
        "표시할 구간을 선택하세요",
        ["선택 안 함", "구간 1 보이기", "구간 2 보이기"]
    )
    
    st.divider()
    
    # 사이드바 버튼을 통한 데이터 조작
    if st.button("구간 1 카운트 올리기"):
        st.session_state.frag1_counter += 1
        # 사이드바 버튼 클릭 시 전체 페이지가 리런되므로, 
        # 아래 '뷰' 영역에서 변경된 값이 반영됩니다.

# --- 4. 메인 화면 그리기 (뷰) ---
st.divider()

if selected_view == "구간 1 보이기":
    simple_fragment1()
elif selected_view == "구간 2 보이기":
    simple_fragment2()
else:
    st.info("👈 왼쪽 사이드바에서 메뉴를 선택해 주세요.")

# 페이지 하단 고정 영역
st.divider()
st.caption("사이드바 위젯을 조작하면 메인 페이지 시간이 갱신되지만, Fragment 내부 버튼을 누르면 내부 시간만 갱신됩니다.")