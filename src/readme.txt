update_category_id
함수에서 idx_to_catid 에서 인덱스 번호와
csvFile의  category_id 번호가 같다면 idx_to_catid의 카테고리 번호로 바꾸어서 저장해 주세요.
MakeModel 모델에서 bBesetLoad 라는 파라미터를 넣고,
이것이 참이면, MODEL_FILES 및에 YOLOv8Model 은 yolobest.pt 를 FasterRCNN은   fasterbest.pt 를 불러오게 하고, Execute_Train 함수의 파라미터에 추가해 주고요,
yolobest.pt나 fasterbest.pt 가 없으면 그냥 로드말고 생성해 주세요.


욜로 파일 생성시,
D:\01.project\EntryPrj\data\oraldrug\1.drug_Image_annotation_allOK\labels\K-003483-025469-030308-035206_0_2_0_2_70_000_200.txt 파일을 보면 
1899 0.756660 0.734375 0.193648 0.148438
이와 같이 객체가 하나만 정의 되어 있는데, 실제로는 객체가 3,4가 있습니다. 파일 이름과 동일한 모든 것을 찾아서
넣어야 합니다.

d:\01.project\EntryPrj\.venv\Lib\site-packages\ultralytics\utils\metrics.py:832: RuntimeWarning: Mean of empty slice.
  i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
d:\01.project\EntryPrj\.venv\Lib\site-packages\numpy\_core\_methods.py:137: RuntimeWarning: invalid value encountered in divide
  ret = um.true_divide(
                   all        925          0          0          0          0          0     
WARNING no labels found in detect set, can not compute metrics without labels

CreateSubmission(self, test_loader, save_image_num = 10):  을 testModel에서 호출 하고 SUBMISSTION_DIR 아래에 
이렇게 디렉토리를 생성하고  f"submission{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}" 디렉토리 및에 f"submission{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.csv" 보고서를 저장.
및 이미지를 save_image_num 갯수만큼 저장해 주되, 박스와 카테고리 이름과 약품 이름을 명기해 주세요.
그리고 csv보고서의 형식은  D:\01.p)ㅏㄴoject\EntryPrj\src\submission.csv 와 같습니다.

TestModelByBest(pt_file): 라는 함수를 만들어서 YOLOv8Model, FasterRCNN 의 에서 이것을 가지고 모델을 로드하고 testModel을 호출하여  submission 을 생성까지 하도록 해주세요.

changesubmissiont(submisstion_file,yaml_file): 에서 submission_file의 category_id 를 yamil_file의 인덱스 값아닌 실제값으로 바꾸어 주세요.(현재 submision_file에는 인덱스가 저장되어 있음.)

D:\01.project\EntryPrj\data\submission\submission20251212112431\submission20251212112431.csv
D:\01.project\EntryPrj\data\oraldrug\yolo_yaml.yamlo



with tab1: 에서 streamit_summary의 display_summary()  을 호출하려면?


YOLOv8Model,FasterRCNN 에서 TestModelByImage 를 TestModelByBest 처럼 만들되 파라메티러 image_file path를 받아서 이미지와 박싱 그리고 ID,category_di, dl_name까지 보여주는 함수를 만들어 주세요.


GetIndexCategoryName(yaml_file,annotaion_dir) 이란 함수를 만들어서, 

 yaml 파일의 names의 인덱스/names/ 이 names 에 다른 annotaion_dir의 json을 찾아서, category_id와  names가 같은  dl_name의 3상으로 되어 있는   집판을 리턴해 주세요.

analyze_image_json_mapping(train_img_dir,train_annotation_dir):

MakeStatistic 로부터 리턴 받은 데이터를
CopyOkImageOkJson함수가 파리미터로 받아서, 
org_images_dir,org_annos_dir 로부터
이미지도 있고, annotaion도있는 것들을
images_dir(이미지),annos_dir(annotation) 을
그대로 카피해 주되, 디렉토리 구조는 동일하게 
가져가 주세요.
 

    def GetAiHubDir(self):
        return self.AI_HUB_IMAGE_DIR, self.AI_HUB_ANNOTATION_DIR

    def get_org_data(self):
        return self.org_data_imges_dir, self.org_data_annotations_dir

    def get_okImage_okAnno(self):
        return self.okImage_okAnno_imges_dir, self.okImage_okAnno_annotations_dir

    def get_noImage_okAnno(self):
        return self.noImage_okAnno_imges_dir, self.noImage_okAnno_annotations_dir

    def get_okImage_noAnno(self):
        return self.okImage_noAnno_imges_dir, self.okImage_noAnno_annotations_dir

    def get_eda_csv(self):
        return self.image2Json, self.json2Image

    def get_visualize_path(self, filename="mapping_statistics.png"):
        vis_dir = os.path.join(self.base_dir, "visualize")
        makedirs(vis_dir)
        return os.path.join(vis_dir, filename)

    def get_all(self):
        return {
            'org_data': (self.org_data_imges_dir, self.org_data_annotations_dir),
            'okImage_okAnno': (self.okImage_okAnno_imges_dir, self.okImage_okAnno_annotations_dir),
            'noImage_okAnno': (self.noImage_okAnno_imges_dir, self.noImage_okAnno_annotations_dir),
            'okImage_noAnno': (self.okImage_noAnno_imges_dir, self.okImage_noAnno_annotations_dir),
            'eda_csv': (self.image2Json, self.json2Image)
        }

train_imge_dir 에서 필요한 이미지 , MISSING_ONLY_ANNOTAIIONS_IMG 로 복사
train_annotation_dir 에서 필요한 annotation 을 MISSING_ONLY_ANNOTATIONS_ANNO 로 복사.
AI_HUB_LABELING_DIR 에서 필요한  annotation을 MISSING_ONLY_ANNOTATIONS_ANNO  로복사(파일 이름이 겹치지 않게 01/02/03 등 순서별로 디렉토리 
만들어서 복사.)

]CopyImageWithOnlyAnno: Train에 없는 이미지: 238개 인데,
실제 카피된 파일 갯수는 137개 밖에 안되요.
dataset_info.get_addJson_result()


tab2 : streamit_modelStudy.py
tab3 : streamit_edastudy.py
tab4 : streamit_edaexecute.py
tab5 : streamit_modelexecute.py
제목은 그대로고 제목에 맞게 스크립트로 일단 이 프로젝트 맞게 채워   주세요.

소규모 프로젝트 수행 후 발표를 하려고 합니다.
대충 큰 제목은 이렇게 하려고 하는데,
수정하거나 추가/삭제할 부분은?
(서브 제목도 필요하면 알아서 추가해 주세요)
너무 자세하거나 과하지 않게 간결하게 해주세요.


1. 프로젝트 개요.
2. 목표 시스템.
3. 개발 전략.
4. 개발 일정 및 R&R
5. 개발 경과.
6. 산출물 및 결과.


1. 프로젝트 개요 (Overview)
배경 및 목적: 이 프로젝트를 왜 시작했는가? (What & Why)
기대 효과: 무엇을 해결/달성하고자 했는가?

2. 핵심 기능 및 아키텍처 (Key Features & Architecture)
시스템 구성도: 전체적인 구조 시각화 (Diagram)
주요 기능 요약: 핵심 기능 3~4가지

3. 개발 전략 (Tech Stack & Strategy)
사용 기술: 프론트엔드, 백엔드, DB, 협업 툴 등 )
선정 이유: 왜 이 기술을 선택했는가? (간략하게)

4. 개발 일정 및 R&R (Schedule & Team)
WBS (일정) 이어야 하나. 그냥 간단하게 달력 형태로 해도 괜찮을듯.
역할 분담: 팀원별 핵심 기여 부분

5. 트러블 슈팅 (Issue & Solution) 
개발 과정에서 발생한 가장 큰 문제점(기술적 난관)과 이를 
어떻게 해결했는지를 보여주는 것이 실력을 어필

6. 결과물 및 회고 (Demo & Retrospective)
기대 효과.
서비스 시연: 스크린샷 또는 짧은 시연 영상
아쉬운 점 & 보완 계획: 향후 발전 방향

예시) 프로젝트 개요
이 프로젝트는 약물 이미지 분류를 위한 딥러닝 모델을 개발하는 것을 목표로 합니다.

예시) 사용 기눌
모델
YOLOv8 (Ultralytics)
Faster R-CNN (torchvision)

프레임워크
PyTorch
Streamlit

데이터 처리
OpenCV, PIL
Pandas, NumPy

시각화
Matplotlib
Streamlit Charts

예시)주요 기능
데이터 분석: 이미지와 어노테이션 매핑 분석, 클래스 분포 확인
모델 학습: YOLOv8 및 Faster R-CNN 모델 학습 및 평가
실시간 예측: 업로드된 이미지에 대한 실시간 약물 탐지
성능 비교: 여러 모델의 성능 지표 비교 및 시각화
Submission 생성: 대회 제출용 CSV 파일 및 시각화 이미지 자동 생성


예시 ( 프로젝트 구조)
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

예시 (주요기능)
 1. **데이터 분석**: 이미지와 어노테이션 매핑 분석, 클래스 분포 확인
    2. **모델 학습**: YOLOv8 및 Faster R-CNN 모델 학습 및 평가
    3. **실시간 예측**: 업로드된 이미지에 대한 실시간 약물 탐지
    4. **성능 비교**: 여러 모델의 성능 지표 비교 및 시각화
    5. **Submission 생성**: 대회 제출용 CSV 파일 및 시각화 이미지 자동 생성

기대 효과
의약품 식별의 자동화 및 효율성 증대
약물 오인 방지를 통한 의료 안전성 향상
실시간 처리가 가능한 경량화된 모델 개발
대규모 약물 데이터베이스 구축 및 관리 용이성