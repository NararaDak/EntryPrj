import streamlit as st
import base64
import os

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

def display_eda_study():
    file = "./files/eda_study.pdf"
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