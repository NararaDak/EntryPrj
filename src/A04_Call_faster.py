from A04 import Execute_Train

if __name__ == "__main__":
    data_dir = r"D:\01.project\EntryPrj\data\oraldrug\1.drug_Image_annotation_allOK"
    
    # ────────────────────────────────────────
    # 예제 : FasterRCNN - 모든 파라미터 명시
    # ────────────────────────────────────────
    trans_type = ["default", "A", "B"]
    for transform_type in trans_type:
        Execute_Train(
            model_type="faster",           # 모델 타입: "faster" 또는 "yolov8"
            data_dir=data_dir,              # 데이터 디렉토리 경로
            backbone="resnet50",            # FasterRCNN 백본: "resnet50" 또는 "mobilenet"
            epochs=50,                      # 학습 에포크 수
            batch_size=16,                  # 배치 크기
            lr=0.005,                       # 학습률 (FasterRCNN 권장: 0.005)
            bBestLoad=True,                # Best 모델 로드 여부 (fasterbest.pt)
            imgsz=640,                      # 이미지 크기
            patience=10,                    # Early stopping patience (에포크 수)
            gubun="partial",                # 최적화 방식: "freeze", "partial", "all"
            train_ratio=0.8,                # 학습/검증 데이터 분할 비율
            num_workers=4,                  # 데이터 로더 워커 수
            transform_type=transform_type              # 데이터 증강 타입: "default", "A", "B"
        )



   