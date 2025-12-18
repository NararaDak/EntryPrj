import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
import requests
import json
import io
import yaml
from openai import OpenAI


# 매핑 함수 로드
@st.cache_data  # 캐싱을 통해 한번만 로드하도록 설정
def load_class_mapping(mapping_file="./files/yolo_class_mapping.json"):
    """yolo_class_mapping.json 파일을 로드하여 클래스 ID 와 약물 정보를 매핑합니다."""
    try:
        with open(mapping_file, "r", encoding="utf-8") as f:
            # JSON의 키는 문자열이므로, class_id 조회를 위해 정수형 키로 변환
            mapping_data = {int(k): v for k, v in json.load(f).items()}
        st.sidebar.success(
            f"'{mapping_file}' 에서 {len(mapping_data)}개 클래스 매핑 로드 완료!"
        )
        return mapping_data

    except FileNotFoundError:
        st.sidebar.error(
            f"오류: '{mapping_file}'을 찾을 수 없습니다. 스크립터를 확인하세요."
        )
        return None
    except Exception as e:
        st.sidebar.error(f"매핑 파일 로드 중 오류 발생: {e}")
        return None


# 약물 API 데이터 로드 함수
@st.cache_data  # 캐싱을 통해 한번만 로드하도록 설정
def load_drug_api_data(api_data_file="./files/drug_API_info.json"):
    """drug_API_info.json 파일을 로드하여 약물 상세 정보를 가져옵니다.

    Returns:
        dict: 약물명을 키로 하는 딕셔너리 형태로 반환 (빠른 검색을 위해)
              예: {"뮤테란캡슐100밀리그램(아세틸시스테인)": {...약물정보...}, ...}
    """
    try:
        with open(api_data_file, "r", encoding="utf-8") as f:
            api_data_list = json.load(f)

        # 배열을 딕셔너리로 변환 (약물명을 키로 사용)
        # 약물명으로 빠르게 검색할 수 있도록 최적화
        api_data_dict = {}
        for item in api_data_list:
            item_name = item.get('itemName', '')
            if item_name:
                # 약물명을 키로 저장
                api_data_dict[item_name] = item

                # 약물명에서 괄호 앞 부분만 추출하여 추가 키로 등록 (더 유연한 검색)
                # 예: "뮤테란캡슐100밀리그램(아세틸시스테인)" -> "뮤테란캡슐100밀리그램"도 키로 등록
                if '(' in item_name:
                    short_name = item_name.split('(')[0].strip()
                    if short_name and short_name not in api_data_dict:
                        api_data_dict[short_name] = item

        st.sidebar.success(
            f"식품의약품안전처 의약품개요정보(e약은요) 데이터 로드 완료\n\n"
            f"{len(api_data_list)}개 약물 상세 정보 ({api_data_file})"
        )
        return api_data_dict

    except FileNotFoundError:
        st.sidebar.warning(
            f"'{api_data_file}' 파일이 없습니다. API 데이터 없이 진행됩니다."
        )
        return None
    except Exception as e:
        st.sidebar.warning(f"API 데이터 로드 중 오류 발생: {e}. API 데이터 없이 진행됩니다.")
        return None


# 페이지 설정 (함수 밖에서 먼저 실행)
st.set_page_config(
    page_title="약 검출 및 상호작용 분석 시스템", page_icon="💊", layout="wide"
)


def display_model():
    """YOLO 모델 실행 및 약물 분석 메인 함수"""
    # 제목
    st.title("약 검출 및 상호작용 분석 시스템")
    st.markdown("YOLO 모델로 약을 검출하고, GPT를 통해 약물 상호작용을 분석합니다.")

    # 매핑 파일 로드
    class_mapping = load_class_mapping()

    # 약물 API 데이터 로드
    drug_api_data = load_drug_api_data()

    # 사이드바
    st.sidebar.header("설정")


    # 신뢰도 임계값 설정
    confidence_threshold = st.sidebar.slider(
        "검출 신뢰도 임계값", min_value=0.0, max_value=1.0, value=0.25, step=0.05
    )

    # 모델 선택 방식
    model_option = st.sidebar.radio(
        "모델 선택 방식",
        ["기본 모델 사용", "커스텀 모델 업로드"],
        help="기본 제공 모델을 사용하거나 직접 업로드할 수 있습니다",
    )


    # 모델 로드
    @st.cache_resource
    def load_model(model_path):
        try:
            model = YOLO(model_path)
            return model
        except Exception as e:
            st.error(f"모델 로드 실패: {e}")
            return None


    model = None
    model_path = None

    if model_option == "기본 모델 사용":
        # model 폴더에서 .pt 파일 찾기
        model_dir = (
            "./model"
            # "/content/drive/MyDrive/Project-Team-1/data/yolo_results/yolov8n_train/weights" # Colab 부분
        )

        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            # model 폴더의 모든 .pt 파일 찾기
            pt_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]

            if pt_files:
                # 발견된 모델 파일 선택
                selected_model = st.sidebar.selectbox(
                    "모델 선택", pt_files, help="model 폴더에서 사용할 모델을 선택하세요"
                )

                model_path = os.path.join(model_dir, selected_model)

                # 모델 로드
                model = load_model(model_path)
                if model:
                    st.sidebar.success(f"모델 로드 완료: {selected_model}")
                else:
                    st.sidebar.error("모델 로드 실패")
            else:
                st.sidebar.error(f"'{model_dir}' 폴더에 .pt 파일이 없습니다")
        else:
            st.sidebar.error(f"'{model_dir}' 폴더를 찾을 수 없습니다")

    else:  # 커스텀 모델 업로드
        uploaded_model = st.sidebar.file_uploader(
            "YOLO 모델 업로드 (.pt)",
            type=["pt"],
            help="학습된 YOLO 모델 파일을 업로드하세요",
        )

        if uploaded_model is not None:
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
                tmp_file.write(uploaded_model.read())
                model_path = tmp_file.name

            # 모델 로드
            model = load_model(model_path)
            if model:
                st.sidebar.success(f"업로드된 모델 로드 완료: {uploaded_model.name}")
            else:
                st.sidebar.error("모델 로드 실패")
        else:
            st.sidebar.info("모델 파일을 업로드해주세요")


    # 약물 상호작용 분석 프롬프트 생성
    def create_drug_interaction_prompt_step1(client, detected_drug_names):
        """검출된 약물에 대한 분석 프롬프트 생성"""
        drug_list_str = ", ".join(detected_drug_names)

        # 새로운 프롬프트
        prompt_content = f"""
        내가 제공하는 약물 목록을 보고, 각 약물에 대한 핵심 정보를 담은 마크다운 테이블을 생성해주길 바라.
    
        # 분석 대상 약물
        {drug_list_str}
    
        # 분석 요청 사항
        1. 위 약물 목록을 바탕으로 다음 column 을 가진 마크다운 테이블을 작성해줘.
        - "약물명"
        - "주요성분"
        - "핵심 효능/ 효과"
        - "대표적인 부작용"
    
        만약 특정 약물의 주요 성분을 모른다면 해당칸에 "정보 없음"이라고 적어줘
        내용은 간결하고 핵심적으로 작성
    
        # 출력 형식 예시
        | 약물명 | 주요 성분 | 핵심 효능/효과 | 대표적인 부작용 |
        |---|---|---|---|
        | 타이레놀정 500mg | 아세트아미노펜 | 해열, 진통 | 소화불량, 구역 |
        """

        # API 호출 결과를 response 변수에 저장
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "너는 약물 정보를 정확하고 구조화된 형식으로 제공하는 AI 약사야.",
                },
                {"role": "user", "content": prompt_content},
            ],
        )

        return response.choices[0].message.content


    # 약물 상호작용 분석 프롬프트 생성
    def create_drug_interaction_prompt_step2(
        client, symptoms, detected_drug_names, user_profile, drug_api_data=None
    ):
        """사용자 개인정보와 약물의 상호 작용 분석하는 맞춤형 프롬프트 생성

        Args:
            client: OpenAI client
            symptoms: 사용자 증상
            detected_drug_names: 검출된 약물명 리스트
            user_profile: 사용자 프로필 정보
            drug_api_data: drug_API_info.json에서 로드한 상세 약물 정보 (선택사항)
        """
        drug_list_str = ", ".join(detected_drug_names)

        # 사용자 프로필 정보를 문자열로 변환 (입력 안된 값은 '정보 없음' 으로 처리)
        profile_str = f"""
        - 나이: {user_profile.get('age', '정보 없음')}세
        - 성별: {user_profile.get('gender', '정보 없음')}
        - 기저질환: {user_profile.get('conditions') or '정보 없음'}
        - 알러지: {user_profile.get('allergies') or '정보 없음'}
        """

        # 증상 정보 처리 (없으면 '정보 없음' 으로 표시)
        symptoms_str = symptoms if symptoms and symptoms.strip() else "정보 없음"

        # 증상 유무에 따라 프롬프트 조정
        symptom_context = ""
        if symptoms_str == "정보 없음":
            symptom_context = "현재 특정 증상은 없지만, 복용하려는 약물들에 대한 일반적인 안전성과 사용자 맞춤 정보를 분석해야 합니다."
        else:
            symptom_context = f"현재 증상({symptoms_str})을 고려하여 약물 복용의 적합성을 분석해야 합니다."

        # API 데이터가 있는 경우 상세 약물 정보 추가
        detailed_drug_info = ""
        if drug_api_data:
            detailed_drug_info = "\n## 4. 약물 상세 정보 (식품의약품안전처 의약품개요정보):\n"
            for drug_name in detected_drug_names:
                # 딕셔너리에서 약물명으로 직접 검색 (O(1) 시간복잡도)
                drug_info = None

                # 1차 시도: 정확한 약물명으로 검색
                if drug_name in drug_api_data:
                    drug_info = drug_api_data[drug_name]
                else:
                    # 2차 시도: 부분 매칭 (약물명에 검출된 이름이 포함된 경우)
                    for api_drug_name, api_drug_info in drug_api_data.items():
                        if drug_name in api_drug_name or api_drug_name in drug_name:
                            drug_info = api_drug_info
                            break

                if drug_info:
                    detailed_drug_info += f"\n### {drug_info.get('itemName', drug_name)}\n"
                    detailed_drug_info += f"- **제조사**: {drug_info.get('entpName', '정보 없음')}\n"

                    if drug_info.get('efcyQesitm'):
                        detailed_drug_info += f"- **효능/효과**: {drug_info['efcyQesitm'].strip()}\n"

                    if drug_info.get('useMethodQesitm'):
                        detailed_drug_info += f"- **사용방법**: {drug_info['useMethodQesitm'].strip()}\n"

                    if drug_info.get('atpnQesitm'):
                        detailed_drug_info += f"- **주의사항**: {drug_info['atpnQesitm'].strip()}\n"

                    if drug_info.get('intrcQesitm'):
                        detailed_drug_info += f"- **상호작용**: {drug_info['intrcQesitm'].strip()}\n"

                    if drug_info.get('seQesitm'):
                        detailed_drug_info += f"- **부작용**: {drug_info['seQesitm'].strip()}\n"
                else:
                    # 약물 정보를 찾지 못한 경우
                    detailed_drug_info += f"\n### {drug_name}\n"
                    detailed_drug_info += f"- **정보**: 데이터베이스에서 상세 정보를 찾을 수 없습니다.\n"

        # 새로운 프롬프트
        prompt_content = f"""
        너는 환자의 개인 정보를 바탕으로 맞춤형 복약 지도를 제공하는 매우 유능한 약사 AI야. 제공된 정보를 바탕으로 명확하고
          구조화된 답변을 한국어로 작성해줘.

        # 분석 정보
        ## 1. 사용자 정보 : {profile_str}
        ## 2. 현재 증상 : {symptoms_str}
        ## 3. 분석 대상 약물 : {drug_list_str}
        {detailed_drug_info}

        # 분석 상황
        {symptom_context}

        # 분석 요청 사항
        위 정보를 바탕으로 다음 항목들을 순서대로, 이해하기 쉽게 분석해줘.
        {"특히 위에 제공된 약물 상세 정보(효능, 주의사항, 상호작용, 부작용)를 반드시 참고하여 분석해줘." if detailed_drug_info else ""}

        1. **종합 평가**:
        - 사용자 정보를 고려했을 때, 이 약들을 함께 복용하는 것에 대한 전반적인 [안전, 주의 필요, 위험] 중 하나로 평가해줘.
        - 증상이 있는 경우: 현재 증상에 대한 약물의 적합성도 함께 평가해줘.
        - 증상이 없는 경우: 약물 간 상호작용과 사용자 특성에 따른 일반적인 안전성을 평가해줘.
        {"- 제공된 약물 상세 정보의 '상호작용' 항목을 반드시 검토하여 병용금기 약물이 있는지 확인해줘." if detailed_drug_info else ""}

        2. **사용자 맞춤 분석**:
        - **나이/성별** : 사용자의 나이와 성별에 따라 특별히 주의해야 할 약이나 부작용이 있는지 설명해줘. (예. 소아/고령자 용량 조절)
        - **기저질환** : 사용자의 기저질환과 약물 간의 잠재적 충돌(부작용 악화, 질병 악화 등) 을 분석해줘.
          {"약물 상세 정보의 '주의사항' 항목에서 기저질환 관련 경고사항을 확인해줘." if detailed_drug_info else ""}
        - **알러지**: 사용자의 알러지 정보와 약물 성분 간의 위험성을 확인하고 경고해줘.
        - **부작용 위험**: {"약물 상세 정보의 '부작용' 항목을 참고하여 사용자에게 발생 가능한 부작용을 구체적으로 설명해줘." if detailed_drug_info else "각 약물의 일반적인 부작용을 설명해줘."}

        3. 최종 권장 사항 :
        - 가장 안전하게 약을 복용할 수 있는 방법에 대해 구체적으로 조언해줘. (예 : 복용 순서, 시간 간격, 식전/식후 등)
          {"약물 상세 정보의 '사용방법'을 참고하여 권장해줘." if detailed_drug_info else ""}
        - 어떤 부작용이 나타나면 즉시 복용을 중단하고 전문가와 상담해야 하는지 알려줘.
        - 병용 시 위험한 약물이나 음식이 있다면 명확히 경고해줘.

        이모지는 사용하지 말고, 각 항목을 명확한 함께 구조적으로 설명해줘.
        """

        # API 호출 결과를 response 변수에 저장
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "너는 환자의 개인 정보를 바탕으로 맞춤형 복약 지도를 제공하는 유능한 약사 AI 이야. 제공된 약물 상세 정보를 반드시 참고하여 정확한 복약 지도를 제공해야 해.",
                },
                {"role": "user", "content": prompt_content},
            ],
        )

        return response.choices[0].message.content


    # 메인 영역
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("이미지 업로드")
        uploaded_file = st.file_uploader(
            "이미지를 선택하세요",
            type=["jpg", "jpeg", "png"],
            help="JPG, JPEG, PNG 형식의 이미지를 업로드하세요",
        )

    # 이미지가 업로드되고 모델이 로드되었을 때
    if uploaded_file is not None and model is not None:
        # 이미지 읽기
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # 원본 이미지 표시
        with col1:
            st.image(image, caption="원본 이미지", use_container_width=True)

        # RGB를 BGR로 변환 (YOLO는 OpenCV 기반이므로 BGR 형식 사용)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # 검출 실행
        with st.spinner("약 검출 중..."):
            results = model(image_bgr, conf=confidence_threshold)

            # 결과 이미지 생성 (BGR 형식)
            result_image = results[0].plot()
            # BGR을 RGB로 변환 (Streamlit 표시용)
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        # 결과 이미지 표시
        with col2:
            st.subheader("검출 결과")
            st.image(result_image, caption="검출된 약", use_container_width=True)

        # 검출된 객체 정보
        st.subheader("검출 상세 정보")

        detections = results[0].boxes
        if len(detections) > 0:
            st.success(f"총 {len(detections)}개의 약이 검출되었습니다.")

            # 검출 정보 테이블
            detection_data = []
            detected_drug_names = []

            # API 사용 여부에 따라 약물명 조회
            use_api = "drug_api_key" in st.session_state

      
            # API 미사용 시 기본 매핑만 사용
            for i, box in enumerate(detections):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # class_mapping (로드된 JSON) 에서 약물 정보 조회
                if class_mapping and class_id in class_mapping:
                    drug_info = class_mapping[class_id]
                    class_name = drug_info.get("item_name", f"이름 없음 (ID : {class_id})")
                else:
                    # 모델 기본 이름이 있으면 사용 , 없으면 ID 표시 (Fallback)
                    class_name = model.names.get(
                        class_id, f"알 수 없는 약물 (ID : {class_id})"
                    )

                detection_data.append(
                    {
                        "번호": i + 1,
                        "약물명": class_name,
                        "클래스 ID": class_id,
                        "신뢰도": f"{confidence:.2%}",
                    }
                )
                detected_drug_names.append(class_name)

            st.table(detection_data)

            # GPT 분석 부분
            if detected_drug_names:
                # GPT API 키 입력창
                API = st.text_input(
                    "GPT API 키를 입력하세요",
                    value="",
                    type="password",
                    help="OpenAI API 키를 입력해주세요",
                )

                if API:  # API 키가 입력된 경우에만 진행
                    client = OpenAI(api_key=API)

                    # GPT step 1 : 검출 분석
                    st.subheader("GPT 약물 분석")

                    if st.button("1단계: 검출된 약물 분석 시작", type="primary"):
                        with st.spinner("GPT가 약물을 분석하는 중..."):
                            # GPT Prompt step 1 : 검출 이미지 분석
                            step1_result = create_drug_interaction_prompt_step1(
                                client, detected_drug_names
                            )
                            # 세션에 저장 (초기화 방지)
                            st.session_state["step1_result"] = step1_result

                    # Step1 결과 표시
                    if "step1_result" in st.session_state:
                        st.markdown("### GPT 1단계 분석 결과")
                        st.markdown(st.session_state["step1_result"])

                        # GPT Prompt step 2 : 증상 기반 분석
                        st.markdown("---")
                        st.subheader("사용자 맞춤 분석")

                        st.markdown("##### 개인 정보 입력")
                        # 사용자 정보를 담을 딕셔너리 생성
                        user_profile = {}

                        # 나이와 성별을 한줄에 배치
                        col_age, col_gender = st.columns(2)
                        with col_age:
                            user_profile["age"] = st.number_input(
                                "나이", min_value=0, max_value=130, value=40, step=1
                            )

                        with col_gender:
                            user_profile["gender"] = st.selectbox(
                                "성별", ["남성", "여성"], index=1
                            )

                        # 기저 질환 및 알러지 정보 입력
                        user_profile["conditions"] = st.text_input(
                            "앓고 있는 기저질환 (예: 고혈압, 당뇨)",
                            help="여러 개일 경우 쉼표(,)로 구분해주세요.",
                        )
                        user_profile["allergies"] = st.text_input(
                            "약물 / 음식 알러지 (예 : 아스피린)",
                            help="여러 개일 경우 쉼표(,)로 구분해주세요",
                        )

                        # 증상 입력 (선택사항)
                        symptoms = st.text_input(
                            "증상을 입력하세요 (선택사항)",
                            key="symptoms_input",
                            help="현재 겪고 있는 증상이 있다면 입력해주세요. 증상이 없어도 분석이 가능합니다.",
                            placeholder="예: 두통, 복통, 발열 등",
                        )

                        # 증상 입력 여부와 관계없이 분석 버튼 표시
                        if st.button("2단계: 맞춤형 복약 분석 시작", type="secondary"):
                            with st.spinner("사용자 정보와 약물을 분석하는 중..."):
                                step2_result = create_drug_interaction_prompt_step2(
                                    client, symptoms, detected_drug_names, user_profile, drug_api_data
                                )
                                # 세션에 저장
                                st.session_state["step2_result"] = step2_result
                                st.session_state["step2_symptoms"] = symptoms  # 증상도 저장

                        # Step2 결과 표시
                        if "step2_result" in st.session_state:
                            st.markdown("### GPT 2단계 분석 결과")
                            st.markdown(st.session_state["step2_result"])

                            st.markdown(
                                "**이 분석은 AI가 제공하는 참고 정보이며, 의사의 처방이나 약사의 복약 지도를 대체할 수 없습니다. 약물 복용 전 반드시 전문가와 상담하세요.**"
                            )

                            # 입력된 정보 표시
                            with st.expander("입력한 정보 확인"):
                                if st.session_state.get("step2_symptoms"):
                                    st.info(
                                        f"**증상:** {st.session_state['step2_symptoms']}"
                                    )
                                else:
                                    st.info("**증상:** 입력 안 됨 (일반 복약 안전성 분석)")
                else:
                    st.warning("GPT API 키를 입력해야 분석을 시작할 수 있습니다.")

    elif uploaded_file is None:
        st.info("이미지를 업로드하여 약 검출을 시작하세요.")
    elif model is None:
        st.error("모델을 먼저 로드해주세요.")


    class_name_option = st.sidebar.radio(
        "클래스명 설정 방식",
        ["모델 기본값", "YAML/JSON 파일 업로드", "직접 입력"],
        help="약물명을 설정하는 방법을 선택하세요",
    )

    class_names_dict = {}

    if class_name_option == "YAML/JSON 파일 업로드":
        uploaded_class_file = st.sidebar.file_uploader(
            "클래스 파일 업로드",
            type=["yaml", "yml", "json"],
            help="YOLO YAML 파일 또는 클래스 매핑 JSON 파일을 업로드하세요",
        )

        if uploaded_class_file is not None:
            try:
                file_extension = uploaded_class_file.name.split(".")[-1].lower()

                if file_extension in ["yaml", "yml"]:
                    # YAML 파일 파싱
                    yaml_content = yaml.safe_load(uploaded_class_file)
                    if "names" in yaml_content:
                        class_names_list = yaml_content["names"]
                        class_names_dict = {
                            i: str(name) for i, name in enumerate(class_names_list)
                        }
                        st.sidebar.success(
                            f"YAML에서 {len(class_names_dict)}개 클래스 로드 완료"
                        )
                    else:
                        st.sidebar.error("YAML 파일에 'names' 필드가 없습니다")

                elif file_extension == "json":
                    # JSON 파일 파싱
                    json_content = json.load(uploaded_class_file)

                    # class_mapping.json 형식 처리 (약물코드: {index: N})
                    if all(
                        isinstance(v, dict) and "index" in v for v in json_content.values()
                    ):
                        for drug_code, info in json_content.items():
                            class_names_dict[info["index"]] = drug_code
                        st.sidebar.success(
                            f"JSON에서 {len(class_names_dict)}개 클래스 로드 완료"
                        )
                    # 일반 매핑 형식 처리 {index: name}
                    else:
                        class_names_dict = {int(k): str(v) for k, v in json_content.items()}
                        st.sidebar.success(
                            f"JSON에서 {len(class_names_dict)}개 클래스 로드 완료"
                        )

                # 매핑 확인
                if class_names_dict:
                    with st.sidebar.expander("매핑 확인 (처음 10개)"):
                        for idx in sorted(list(class_names_dict.keys())[:10]):
                            st.text(f"Class {idx} → {class_names_dict[idx]}")
                        if len(class_names_dict) > 10:
                            st.text(f"... 외 {len(class_names_dict) - 10}개")

            except Exception as e:
                st.sidebar.error(f"파일 로드 실패: {e}")

    elif class_name_option == "직접 입력":
        st.sidebar.info("쉼표로 구분하여 약물명을 입력하세요")

        class_names_input = st.sidebar.text_area(
            "약물명 입력 (순서대로)",
            value="약물1, 약물2, 약물3",
            help="클래스 0부터 순서대로 약물명을 입력하세요. 쉼표로 구분합니다.",
            height=100,
        )

        # 입력된 약물명 파싱
        if class_names_input:
            class_names_list = [name.strip() for name in class_names_input.split(",")]
            class_names_dict = {i: name for i, name in enumerate(class_names_list)}

            with st.sidebar.expander("매핑 확인"):
                for idx, name in class_names_dict.items():
                    st.text(f"Class {idx} → {name}")


# 메인 실행
if __name__ == "__main__":
    display_model()
