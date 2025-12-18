import streamlit as st
import base64
import os

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

def display_summary_org():
    """프로젝트 개요를 표시하는 함수"""
    
    st.write("## 📋 프로젝트 개요")
    st.write("이 프로젝트는 약물 이미지 분류를 위한 딥러닝 모델을 개발하는 것을 목표로 합니다.")
    
    st.write("---")
    
    # 프로젝트 목적
    st.write("### 🎯 프로젝트 목적")
    st.write("""
    - 약물 이미지를 자동으로 분류하여 의약품 식별의 정확성과 효율성 향상
    - 객체 탐지(Object Detection) 기술을 활용한 실시간 약물 인식
    - YOLOv8 및 Faster R-CNN 모델을 활용한 성능 비교 및 최적화
    """)
    
    st.write("---")
    
    # 사용 기술
    st.write("### 🔧 사용 기술")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**모델**")
        st.write("- YOLOv8 (Ultralytics)")
        st.write("- Faster R-CNN (torchvision)")
        
        st.write("**프레임워크**")
        st.write("- PyTorch")
        st.write("- Streamlit")
    
    with col2:
        st.write("**데이터 처리**")
        st.write("- OpenCV, PIL")
        st.write("- Pandas, NumPy")
        
        st.write("**시각화**")
        st.write("- Matplotlib")
        st.write("- Streamlit Charts")
    
    st.write("---")
    
    # 주요 기능
    st.write("### 📊 주요 기능")
    st.write("""
    1. **데이터 분석**: 이미지와 어노테이션 매핑 분석, 클래스 분포 확인
    2. **모델 학습**: YOLOv8 및 Faster R-CNN 모델 학습 및 평가
    3. **실시간 예측**: 업로드된 이미지에 대한 실시간 약물 탐지
    4. **성능 비교**: 여러 모델의 성능 지표 비교 및 시각화
    5. **Submission 생성**: 대회 제출용 CSV 파일 및 시각화 이미지 자동 생성
    """)
    
    st.write("---")
    
    # 프로젝트 구조
    st.write("### 📁 프로젝트 구조")
    st.code("""
    EntryPrj/
    ├── data/                   # 데이터 디렉토리
    │   ├── oraldrug/
    │   │   ├── train_images/   # 학습 이미지
    │   │   ├── train_annotations/  # 어노테이션
    │   │   └── test_images/    # 테스트 이미지
    │   ├── modelfiles/         # 저장된 모델
    │   └── submission/         # 제출 파일
    ├── src/                    # 소스 코드
    │   ├── A04.py             # 메인 학습 코드
    │   ├── eda.py             # 데이터 분석
    │   └── streamitService.py # Streamlit 앱
    └── doc/                   # 문서
    """, language="text")
    
    st.write("---")
    
    # 성과
    st.write("### 🏆 기대 효과")
    st.write("""
    - 의약품 식별의 자동화 및 효율성 증대
    - 약물 오인 방지를 통한 의료 안전성 향상
    - 실시간 처리가 가능한 경량화된 모델 개발
    - 대규모 약물 데이터베이스 구축 및 관리 용이성
    """)

def display_summary():
    """PDF 리포트를 표시하는 함수"""
    file = Best_One_Team_project.pdf"
    
    st.write("## 📄 프로젝트 리포트")
    
    # PDF 파일 존재 확인
    if not os.path.exists(file):
        st.error(f"❌ PDF 파일을 찾을 수 없습니다: {file}")
        st.info("파일 경로를 확인해주세요.")
        return
    
    try:
        if HAS_PYMUPDF:
            # PyMuPDF를 사용하여 PDF를 이미지로 변환
            pdf_document = fitz.open(file)
            
            # 각 페이지를 이미지로 변환하여 표시
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # 페이지를 이미지로 렌더링 (해상도 2배)
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat)
                
                # PIL Image로 변환
                import io
                from PIL import Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Streamlit에 표시
                st.image(img, use_container_width=True)
                
                # 페이지 구분선 (마지막 페이지 제외)
                if page_num < len(pdf_document) - 1:
                    st.divider()
            
            pdf_document.close()
        else:
            # PyMuPDF가 없으면 base64 방식 시도
            with open(file, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
            
            st.info("💡 PDF가 표시되지 않으면 PyMuPDF를 설치하세요: pip install PyMuPDF")
        
    except Exception as e:
        st.error(f"❌ PDF 로드 중 오류 발생: {e}")
        st.code(f"Error details: {str(e)}")
        
        # 파일 정보 표시
        st.write("### 파일 정보")
        st.write(f"- 파일 경로: {file}")
        st.write(f"- 파일 존재: {os.path.exists(file)}")
        if os.path.exists(file):

            st.write(f"- 파일 크기: {os.path.getsize(file) / 1024:.2f} KB")
