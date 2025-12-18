from A04 import Execute_Train

if __name__ == "__main__":
   # data_dir = r"D:\01.project\EntryPrj\data\oraldrug\2.drug_no_image_ok_Anno"
   # data_dir = r"D:\01.project\EntryPrj\data\oraldrug\3.drug_ok_Image_no_Anno"
    
    data_dir = r"D:\01.project\EntryPrj\data\oraldrug\1.drug_Image_annotation_allOK"
    
    trans_type = ["default", "A", "B"]
    for transform_type in trans_type:
        # ────────────────────────────────────────
        # 예제 : YOLOv8 Nano - 모든 파라미터 명시
        # ────────────────────────────────────────
        Execute_Train(
            model_type="yolov8",            # 모델 타입: "yolov8" 또는 "faster"
            data_dir=data_dir,              # 데이터 디렉토리 경로
            model_size="n",                 # YOLOv8 모델 크기: "n", "s", "m", "l", "x"
            epochs=1,                      # 학습 에포크 수
            batch_size=16,                  # 배치 크기
            lr=0.001,                       # 학습률 (YOLOv8 권장: 0.001)
            bBestLoad=True,                # Best 모델 로드 여부 (yolobest.pt)
            imgsz=640,                      # 이미지 크기
            patience=10,                    # Early stopping patience (에포크 수)
            train_ratio=0.8,                # 학습/검증 데이터 분할 비율
            num_workers=4,                  # 데이터 로더 워커 수
            transform_type=transform_type              # 데이터 증강 타입: "default", "A", "B"
        )


        # ────────────────────────────────────────
        # 예제 : FasterRCNN - 모든 파라미터 명시
        # ────────────────────────────────────────
        
        # Execute_Train(
        #     model_type="faster",           # 모델 타입: "faster" 또는 "yolov8"
        #     data_dir=data_dir,              # 데이터 디렉토리 경로
        #     backbone="resnet50",            # FasterRCNN 백본: "resnet50" 또는 "mobilenet"
        #     epochs=50,                      # 학습 에포크 수
        #     batch_size=16,                  # 배치 크기
        #     lr=0.005,                       # 학습률 (FasterRCNN 권장: 0.005)
        #     bBestLoad=False,                # Best 모델 로드 여부 (fasterbest.pt)
        #     imgsz=640,                      # 이미지 크기
        #     patience=10,                    # Early stopping patience (에포크 수)
        #     gubun="partial",                # 최적화 방식: "freeze", "partial", "all"
        #     train_ratio=0.8,                # 학습/검증 데이터 분할 비율
        #     num_workers=4,                  # 데이터 로더 워커 수
        #     transform_type=transform_type              # 데이터 증강 타입: "default", "A", "B"
        # )




    # # ────────────────────────────────────────
    # # 예제 : YOLOv8 Small - 고급 설정
    # # ────────────────────────────────────────
    # Execute_Train(
    #     model_type="yolov8",
    #     data_dir=data_dir,
    #     model_size="s",                 # Small 모델 (Nano보다 큼)
    #     epochs=100,                     # 더 긴 학습
    #     batch_size=32,                  # 더 큰 배치
    #     lr=0.01,                        # 더 높은 학습률
    #     bBestLoad=True,                 # Best 모델부터 시작 (yolobest.pt)
    #     imgsz=1024,                     # 더 큰 이미지 크기
    #     patience=20,                    # 더 긴 patience
    #     train_ratio=0.9,                # 더 많은 학습 데이터
    #     num_workers=0,                  # 더 많은 워커
    #     transform_type="B"              # 다른 증강 타입
    # )

    # # ────────────────────────────────────────
    # # 예제 : FasterRCNN MobileNet - 경량화 설정
    # # ────────────────────────────────────────
    # Execute_Train(
    #     model_type="faster",
    #     data_dir=data_dir,
    #     backbone="mobilenet",           # 경량 백본
    #     epochs=30,                      # 짧은 학습
    #     batch_size=8,                   # 작은 배치 (메모리 절약)
    #     lr=0.003,                       # 낮은 학습률
    #     bBestLoad=True,                 # Best 모델부터 시작 (fasterbest.pt)
    #     imgsz=512,                      # 작은 이미지 크기
    #     patience=10,                    # Early stopping patience (에포크 수)
    #     gubun="freeze",                 # Backbone 고정 (빠른 학습)
    #     train_ratio=0.75,               # 더 많은 검증 데이터
    #     num_workers=0,
    #     transform_type="default"        # 기본 증강
    # )
