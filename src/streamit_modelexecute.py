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


def display_model():
    # 한국의약품안전나라 API 연동 함수
    @st.cache_data(ttl=3600)  # 1시간 캐싱
    def get_drug_name_from_api(drug_code, api_key):
        """
        약물 식별번호를 실제 약물명으로 변환

        Args:
            drug_code: 약물 식별번호 (예: '200003092')
            api_key: 의약품안전나라 API 키

        Returns:
            약물명 또는 None
        """
        try:
            # 의약품 제품정보 조회 API - 여러 API 엔드포인트 시도
            from urllib.parse import unquote

            decoded_key = unquote(api_key)

            # API 엔드포인트 목록 (우선순위 순)
            api_endpoints = [
                # 1. 의약품개요정보(e약은요) - 가장 일반적
                {
                    "url": "http://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList",
                    "params": {
                        "serviceKey": decoded_key,
                        "itemSeq": drug_code,
                        "type": "json",
                        "pageNo": "1",
                        "numOfRows": "1",
                    },
                    "item_name_key": "itemName",
                },
                # 2. 의약품 낱알식별 정보 - item_seq로 조회
                {
                    "url": "http://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService03/getMdcinGrnIdntfcInfoList03",
                    "params": {
                        "serviceKey": decoded_key,
                        "item_seq": drug_code,
                        "type": "json",
                        "pageNo": "1",
                        "numOfRows": "1",
                    },
                    "item_name_key": "ITEM_NAME",
                },
                # 3. 의약품제품정보 (대체 API)
                {
                    "url": "http://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService05/getDrugPrdtPrmsnDtlInq04",
                    "params": {
                        "serviceKey": decoded_key,
                        "item_seq": drug_code,
                        "type": "json",
                        "pageNo": "1",
                        "numOfRows": "1",
                    },
                    "item_name_key": "ITEM_NAME",
                },
            ]

            response = None
            last_error = None

            # 여러 API 엔드포인트 순차 시도
            for i, endpoint in enumerate(api_endpoints):
                try:
                    response = requests.get(
                        endpoint["url"], params=endpoint["params"], timeout=10
                    )
                    if response.status_code == 200:
                        data = response.json()

                        # 에러 응답 체크
                        if "header" in data:
                            result_code = data["header"].get("resultCode", "")
                            result_msg = data["header"].get("resultMsg", "")

                            if result_code != "00":
                                last_error = f"엔드포인트 {i+1} API 오류: [{result_code}] {result_msg}"
                                continue

                        # 응답 구조 파싱 시도
                        if "body" in data and "items" in data["body"]:
                            items = data["body"]["items"]
                            if items and len(items) > 0:
                                item = items[0]
                                # 엔드포인트별 item_name_key 사용
                                item_name_key = endpoint.get("item_name_key", "itemName")
                                drug_name = item.get(item_name_key)

                                if drug_name:
                                    st.info(
                                        f"API 엔드포인트 {i+1} 성공 (약물: {drug_name})"
                                    )
                                    return drug_name

                        last_error = f"엔드포인트 {i+1}: 응답 데이터 없음"
                        continue
                    else:
                        last_error = f"엔드포인트 {i+1} 실패: HTTP {response.status_code}"
                        continue
                except Exception as e:
                    last_error = f"엔드포인트 {i+1} 오류: {str(e)}"
                    continue

            # 모든 엔드포인트 실패
            st.error(
                f"모든 API 엔드포인트 실패 (약물코드: {drug_code}). 마지막 오류: {last_error}"
            )
            return None

        except requests.exceptions.RequestException as e:
            st.error(f"네트워크 오류 (약물코드: {drug_code}): {str(e)}")
            return None
        except Exception as e:
            st.error(f"예상치 못한 오류 (약물코드: {drug_code}): {str(e)}")
            return None


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


    # 페이지 설정
    st.set_page_config(
        page_title="약 검출 및 상호작용 분석 시스템", page_icon="💊", layout="wide"
    )

    # 제목
    st.title("약 검출 및 상호작용 분석 시스템")
    st.markdown("YOLO 모델로 약을 검출하고, GPT를 통해 약물 상호작용을 분석합니다.")

    # 매핑 파일 로드
    class_mapping = load_class_mapping()

    # 사이드바
    st.sidebar.header("설정")

    # # 한국의약품안전나라 API 설정 (상단에 배치)
    # st.sidebar.subheader("의약품 API 설정")
    # use_drug_api = st.sidebar.checkbox(
    #     "한국의약품안전나라 API 사용",
    #     value=False,
    #     help="품목기준코드를 실제 약물명으로 변환합니다",
    # )

    # if use_drug_api:
    #     api_key_input = st.sidebar.text_input(
    #         "API 키 입력",
    #         value="",
    #         type="password",
    #         help="https://www.data.go.kr 에서 발급받은 일반 인증키(Decoding)를 입력하세요",
    #     )

    #     if api_key_input:
    #         st.session_state["drug_api_key"] = api_key_input
    #         st.sidebar.success("API 키가 설정되었습니다")

    #         # API 키 테스트 버튼
    #         if st.sidebar.button(
    #             "API 키 테스트", help="테스트 약물코드로 API 연결을 확인합니다"
    #         ):
    #             with st.spinner("API 테스트 중..."):
    #                 # 테스트용 약물코드 (타이레놀 등 흔한 약)
    #                 test_result = get_drug_name_from_api("200003092", api_key_input)
    #                 if test_result:
    #                     st.sidebar.success(f"API 연결 성공! 테스트 약물: {test_result}")
    #                 else:
    #                     st.sidebar.error("API 연결 실패. 위의 오류 메시지를 확인하세요.")
    #     else:
    #         st.sidebar.warning("API 키를 입력해주세요")
    #         st.sidebar.markdown(
    #             """
    #             **API 키 발급 방법:**
    #             1. [공공데이터포털](https://www.data.go.kr) 접속
    #             2. 회원가입 및 로그인
    #             3. 다음 중 하나의 서비스에 활용신청:
    #                - '의약품개요정보(e약은요)조회서비스' (권장)
    #                - '의약품 제품 허가정보 조회' (대체)
    #             4. 승인 완료 대기 (보통 1시간~1일 소요)
    #             5. [마이페이지 - 활용신청 현황](https://www.data.go.kr/iim/api/selectAPIAcountView.do)에서 상태 확인
    #             6. 승인 완료 후 일반 인증키(Decoding) 복사

    #             **중요 확인 사항:**
    #             - 활용신청 현황에서 상태가 '승인'인지 확인
    #             - 일반 인증키(Decoding) 사용 (Encoding 키 아님)
    #             - 여러 의약품 API 중 하나 이상에 승인되어 있어야 함

    #             **403 오류 해결:**
    #             1. 마이페이지에서 API 승인 상태 확인
    #             2. 승인 안 됨 → 승인 대기
    #             3. 승인 완료 → Decoding 키 확인
    #             """
    #         )
    # else:
    #     # API 사용 안 함 - 세션에서 제거
    #     if "drug_api_key" in st.session_state:
    #         del st.session_state["drug_api_key"]

    # st.sidebar.markdown("---")

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
        client, symptoms, detected_drug_names, user_profile
    ):
        """사용자 개인정보와 약물의 상호 작용 분석하는 맞춤형 프롬프트 생성"""
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

        # 새로운 프롬프트
        prompt_content = f"""
        너는 환자의 개인 정보를 바탕으로 맞춤형 복약 지도를 제공하는 매우 유능한 약사 AI야. 제공된 정보를 바탕으로 명확하고
          구조화된 답변을 한국어로 작성해줘.

        # 분석 정보
        ## 1. 사용자 정보 : {profile_str}
        ## 2. 현재 증상 : {symptoms_str}
        ## 3. 분석 대상 약물 : {drug_list_str}

        # 분석 상황
        {symptom_context}

        # 분석 요청 사항
        위 정보를 바탕으로 다음 항목들을 순서대로, 이해하기 쉽게 분석해줘.

        1. **종합 평가**:
        - 사용자 정보를 고려했을 때, 이 약들을 함께 복용하는 것에 대한 전반적인 [안전, 주의 필요, 위험] 중 하나로 평가해줘.
        - 증상이 있는 경우: 현재 증상에 대한 약물의 적합성도 함께 평가해줘.
        - 증상이 없는 경우: 약물 간 상호작용과 사용자 특성에 따른 일반적인 안전성을 평가해줘.

        2. **사용자 맞춤 분석**:
        - **나이/성별** : 사용자의 나이와 성별에 따라 특별히 주의해야 할 약이나 부작용이 있는지 설명해줘. (예. 소아/고령자 용량 조절)
        - **기저질환** : 사용자의 기저질환과 약물 간의 잠재적 충돌(부작용 악화, 질병 악화 등) 을 분석해줘.
        -   **알러지**: 사용자의 알러지 정보와 약물 성분 간의 위험성을 확인하고 경고해줘.

        3. 최종 권장 사항 :
        - 가장 안전하게 약을 복용할 수 있는 방법에 대해 구체적으로 조언해줘. (예 : 복용 순서, 시간 간격, 식전/식후 등)
        - 어떤 부작용이 나타나면 즉시 복용을 중단하고 전문가와 상담해야 하는지 알려줘.

        이모지는 사용하지 말고, 각 항목을 명확한 함께 구조적으로 설명해줘.
        """

        # API 호출 결과를 response 변수에 저장
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "너는 환자의 개인 정보를 바탕으로 맞춤형 복약 지도를 제공하는 유능한 약사 AI 이야.",
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

        # 검출 실행
        with st.spinner("약 검출 중..."):
            results = model(image_np, conf=confidence_threshold)

            # 결과 이미지 생성
            result_image = results[0].plot()
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

            # if use_api:
            #     with st.spinner("한국의약품안전나라 API로 약물명 조회 중..."):
            #         # 약물 정보 조회
            #         for i, box in enumerate(detections):
            #             class_id = int(box.cls[0])
            #             confidence = float(box.conf[0])

            #             # class_mapping (로드된 JSON) 에서 약물 정보 조회
            #             if class_mapping and class_id in class_mapping:
            #                 drug_info = class_mapping[class_id]
            #                 item_seq = drug_info.get(
            #                     "item_seq", None
            #                 )  # 품목기준코드 가져오기
            #                 class_name = drug_info.get(
            #                     "item_name", f"이름 없음 (ID : {class_id})"
            #                 )

            #                 # 한국의약품안전나라 API로 정확한 약물명 조회 (item_seq가 있는 경우)
            #                 if item_seq:
            #                     api_drug_name = get_drug_name_from_api(
            #                         item_seq, st.session_state["drug_api_key"]
            #                     )
            #                     if api_drug_name:
            #                         class_name = api_drug_name
            #                         st.info(
            #                             f"{i+1}번 약물: API 조회 완료 - {api_drug_name}"
            #                         )
            #             else:
            #                 # 모델 기본 이름이 있으면 사용 , 없으면 ID 표시 (Fallback)
            #                 class_name = model.names.get(
            #                     class_id, f"알 수 없는 약물 (ID : {class_id})"
            #                 )

            #             detection_data.append(
            #                 {
            #                     "번호": i + 1,
            #                     "약물명": class_name,
            #                     "클래스 ID": class_id,
            #                     "신뢰도": f"{confidence:.2%}",
            #                 }
            #             )
            #             detected_drug_names.append(class_name)
            # else:
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

                # GPT API 키 입력창 (오류 처리 추가)
                try:
                    API = st.text_input(
                        "GPT API 키를 입력하세요",
                        value="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"
                        #st.secrets["OPENAI_API_KEY"],
                        type="password",
                        help="OpenAI API 키를 입력해주세요",
                    )
                except Exception as e:
                    st.error(f"API 키를 불러오는 중 오류 발생: {e}")
                    API = ""

                if API:  # API 키가 입력된 경우에만 진행
                    client = OpenAI(api_key=API)

                    # GPT step 1 : 검출 분석
                    st.subheader("GPT 약물 분석")

                    if st.button("1단계: 검출된 약물 분석 시작", type="primary"):
                        with st.spinner("GPT가 약물을 분석하는 중..."):
                            try:
                                # GPT Prompt step 1 : 검출 이미지 분석
                                step1_result = create_drug_interaction_prompt_step1(
                                    client, detected_drug_names
                                )
                                # 세션에 저장 (초기화 방지)
                                st.session_state["step1_result"] = step1_result
                            except Exception as e:
                                error_msg = str(e)
                                st.error(f"GPT 분석 중 오류 발생: {error_msg}")
                                if "401" in error_msg or "invalid_api_key" in error_msg or "Incorrect API key" in error_msg:
                                    st.warning("OpenAI API 키가 올바르지 않거나 만료되었습니다. https://platform.openai.com/account/api-keys 에서 유효한 API 키를 확인 필요.")

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
                                    client, symptoms, detected_drug_names, user_profile
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

            # 약물 상호작용 분석 섹션
            # st.subheader("약물 상호작용 분석")

            # if len(detected_drug_names) >= 2:
            #     if st.button("약물 상호작용 분석 시작", type="primary"):
            #         with st.spinner("Ollama를 통해 약물 상호작용을 분석하는 중..."):
            #             # 프롬프트 생성
            #             prompt = create_drug_interaction_prompt(detected_drug_names)

            #             # Ollama API 호출
            #             analysis_result = call_ollama(prompt, ollama_model, ollama_url)

            #             # 결과 표시
            #             st.markdown("### 분석 결과")
            #             st.markdown(analysis_result)

            #             # 프롬프트 보기 (디버깅용)
            #             with st.expander("사용된 프롬프트 보기"):
            #                 st.code(prompt, language="text")

            # elif len(detected_drug_names) == 1:
            #     st.info("약물 상호작용 분석을 위해서는 2개 이상의 약이 필요합니다.")

        #     # 결과 다운로드
        #     st.subheader("결과 다운로드")
        #     result_pil = Image.fromarray(result_image)

        #     # 이미지를 메모리 내 버퍼에 저장
        #     buf = io.BytesIO()
        #     result_pil.save(buf, format="JPEG")
        #     buf.seek(0)

        #     st.download_button(
        #         label="결과 이미지 다운로드",
        #         data=buf,
        #         file_name="detected_pills.jpg",
        #         mime="image/jpeg",
        #     )
        # else:
        #     st.warning("검출된 약이 없습니다. 신뢰도 임계값을 낮춰보세요.")

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


    # # Ollama 설정
    # st.sidebar.subheader("Ollama 설정")
    # ollama_url = st.sidebar.text_input(
    #     "Ollama API URL",
    #     value="http://localhost:11434/api/generate",
    #     help="Ollama 서버의 API 엔드포인트",
    # )

    # ollama_model = st.sidebar.selectbox(
    #     "Ollama 모델",
    #     ["llama2", "mistral", "llama3", "gemma", "phi"],
    #     help="사용할 Ollama 모델을 선택하세요",
    # )


    # # Ollama API 호출 함수
    # def call_ollama(prompt, model_name, api_url):
    #     """Ollama API를 호출하여 약물 상호작용 분석"""
    #     try:
    #         payload = {"model": model_name, "prompt": prompt, "stream": False}

    #         response = requests.post(api_url, json=payload, timeout=60)
    #         response.raise_for_status()

    #         result = response.json()
    #         return result.get("response", "응답을 받을 수 없습니다.")

    #     except requests.exceptions.ConnectionError:
    #         return (
    #             "오류: Ollama 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인하세요."
    #         )
    #     except requests.exceptions.Timeout:
    #         return "오류: 요청 시간이 초과되었습니다."
    #     except Exception as e:
    #         return f"오류: {str(e)}"


    def batch_convert_drug_names(drug_codes, api_key):
        """
        여러 약물 식별번호를 한번에 변환

        Args:
            drug_codes: 약물 식별번호 리스트
            api_key: API 키

        Returns:
            dict: {식별번호: 약물명} 매핑
        """
        result = {}

        with st.spinner(f"약물명 변환 중... (총 {len(drug_codes)}개)"):
            progress_bar = st.progress(0)

            for i, code in enumerate(drug_codes):
                drug_name = get_drug_name_from_api(str(code), api_key)
                if drug_name:
                    result[str(code)] = drug_name
                else:
                    result[str(code)] = f"약물_{code}"  # API 실패 시 기본값

                # 진행률 업데이트
                progress_bar.progress((i + 1) / len(drug_codes))

            progress_bar.empty()

        return result


    # 사용 방법
    with st.expander("사용 방법"):
        st.markdown(
            """
        ### YOLO 약 검출

        **1. 한국의약품안전나라 API 설정 (선택사항)**
        - 사이드바 상단의 "의약품 API 설정" 체크
        - [공공데이터포털](https://www.data.go.kr)에서 API 키 발급
        - API 키를 입력하면 품목기준코드를 실제 약물명으로 자동 변환
        - **주의**: API 키가 없어도 기본 매핑 파일로 약물명 표시 가능

        **2. 모델 준비**
        - 방법 1 - 기본 모델 사용:
          - `model` 폴더에 .pt 파일 넣기
          - 사이드바에서 "기본 모델 사용" 선택
          - 드롭다운에서 모델 선택

        - 방법 2 - 커스텀 모델 업로드:
          - 사이드바에서 "커스텀 모델 업로드" 선택
          - .pt 파일 직접 업로드

        **3. 약물명 설정**
        - `yolo_class_mapping.json` 파일이 있으면 자동으로 로드됩니다
        - API 키가 설정되어 있으면 자동으로 정확한 약물명을 조회합니다

        **4. 검출 실행**
        1. 검출 신뢰도 임계값을 조정하세요 (기본값: 0.25)
        2. 약이 포함된 이미지를 업로드하세요
        3. 검출 결과를 확인하세요
        4. API 사용 시 각 약물별 조회 진행 상황을 확인할 수 있습니다

        ### GPT 약물 분석 (2단계)

        **1단계: 검출된 약물 분석**
        1. 약물 검출 후 GPT API 키를 입력하세요
           - OpenAI 계정에서 API 키 발급: https://platform.openai.com/api-keys
        2. "1단계: 검출된 약물 분석 시작" 버튼 클릭
        3. GPT가 검출된 약물의 효능을 분석합니다

        **2단계: 사용자 증상 기반 분석**
        1. 1단계 완료 후 현재 겪고 있는 증상을 입력하세요
           - 예: "두통", "복통", "발열" 등
        2. "2단계: 증상 기반 분석 시작" 버튼 클릭
        3. GPT가 증상에 맞춰 복용 주의사항을 분석합니다

        **특징:**
        - 각 단계별 결과가 저장되어 초기화되지 않습니다
        - 입력한 증상을 확인할 수 있습니다
        - GPT-4o-mini 모델을 사용하여 빠르고 정확한 분석 제공

        ### 주의사항
        - 약물 식별번호를 실제 약물명으로 변환하려면 별도의 매핑 파일이 필요합니다
        - **이 분석 결과는 참고용이며, 실제 복용 전 반드시 의사나 약사와 상담하세요**
        - GPT API 키는 안전하게 보관하고 타인과 공유하지 마세요
        """
        )

    # 프롬프트 엔지니어링 팁
    with st.expander("프롬프트 엔지니어링 가이드"):
        st.markdown(
            """
        ### 현재 프롬프트 구조

        이 앱은 **2단계 GPT 분석**을 사용하며, 다음과 같은 프롬프트 엔지니어링 기법을 적용합니다:

        **1단계: 검출된 약물 분석**
        ```python
        create_drug_interaction_prompt_step1(client, detected_drug_names)
        ```
        - **역할 부여 (Role Assignment)**: "너는 약학 전문가야" - AI에게 전문 약사 역할 부여
        - **구조화된 출력**: 번호로 구분된 명확한 분석 형식
        - **컨텍스트 제공**: 검출된 약물 목록과 개수를 명시
        - **제약 조건**: "이모지 없이 간략히 요약" - 출력 형식 제어

        **2단계: 사용자 맞춤 분석**
        ```python
        create_drug_interaction_prompt_step2(client, symptoms, detected_drug_names)
        ```
        - **역할 부여**: "너는 약사 AI이야" - 의료 상담 역할 강조
        - **다중 컨텍스트**: 사용자 증상 + 검출된 약물 정보 결합
        - **안전성 강조**: 전문의 상담 경고 포함 요청
        - **구체적 지시**: 섭취 금지 약물 명시적으로 요청

        ### 주요 프롬프트 엔지니어링 기법

        1. **Chain of Thought (단계별 사고)**
           - 1단계에서 약물 효능 분석 → 2단계에서 증상 맞춤 분석
           - 점진적으로 정보를 구체화

        2. **Few-shot Learning 예시**
           - 증상 입력 시 예시 제공: "두통, 복통, 발열 등"
           - 사용자가 어떻게 입력해야 하는지 가이드

        3. **제약 조건 명시**
           - "이모지 없이" - 불필요한 출력 제거
           - "간략히 요약" - 핵심만 전달
           - "전문의 상담 경고" - 의료 윤리 준수

        ### 프롬프트 커스터마이징 방법

        코드의 `create_drug_interaction_prompt_step1` 또는 `create_drug_interaction_prompt_step2` 함수를 수정하여:

        **더 상세한 분석:**
        ```python
        4. 약물의 부작용도 설명해줘.
        5. 복용 시간과 용량도 권장해줘.
        ```

        **특정 상황 고려:**
        ```python
        사용자 상황: 임신 중, 고령자, 어린이
        이 상황에서 특별히 주의할 점을 알려줘.
        ```

        **다른 출력 형식:**
        ```python
        JSON 형식으로 답변해줘:
        {{"약물명": "효능", "주의사항": "..."}}
        ```

        ### GPT 모델 
        - `gpt-4o-mini` (현재) - 빠르고 저렴
        """
        )
