# ════════════════════════════════════════
# 모델 평가 및 Submission 생성 통합 스크립트
# ════════════════════════════════════════

import os
import sys

# A03.py와 create_submission.py import
from A03 import (
    OpLog,
    Lines,
    YOLOv8Model,
    FasterRCNNModel,
    GetLoaders,
    ANNOTATION_DIR,
    TRAIN_IMG_DIR,
    MODEL_FILES,
    DEVICE_TYPE,
)

from create_submission import (
    evalModel_improved_yolo,
    evalModel_improved_fasterrcnn,
    create_submission_csv_improved,
    visualize_submission_results,
    create_submission_report,
)


def evaluate_and_create_submission(
    model_type="yolov8s",
    model_weights_path=None,
    confidence_threshold=0.25,
    evaluate_on_val=True,
    skip_prediction=False,
    use_existing_csv=None,
):
    """
    모델 평가 (mAP@[0.75:0.95]) 및 Submission CSV 생성

    Args:
        model_type: 모델 타입 ("yolov8s", "yolov8n", "fasterrcnn_resnet50" 등)
        model_weights_path: 학습된 모델 가중치 경로 (None이면 자동 탐색)
        confidence_threshold: 신뢰도 임계값
        evaluate_on_val: Validation 데이터로 mAP 평가 수행 여부
        skip_prediction: True면 예측을 건너뛰고 시각화만 수행
        use_existing_csv: 기존 submission CSV 경로 (None이면 새로 생성)
    """

    Lines(f"평가 시작: {model_type}")

    # ════════════════════════════════════════
    # 0. 기존 CSV 사용 또는 예측 건너뛰기 모드
    # ════════════════════════════════════════

    if use_existing_csv and os.path.exists(use_existing_csv):
        OpLog(f"기존 Submission CSV 사용: {use_existing_csv}", bLines=True)
        submission_path = use_existing_csv
        skip_prediction = True
        evaluate_on_val = False

    if skip_prediction:
        OpLog("예측 단계를 건너뛰고 시각화만 수행합니다.", bLines=True)
        if not use_existing_csv:
            # 자동으로 submission 파일 찾기
            submission_filename = (
                f"submission_{model_type}_conf{confidence_threshold}.csv"
            )
            if os.path.exists(submission_filename):
                submission_path = submission_filename
                OpLog(f"발견된 Submission 파일 사용: {submission_path}", bLines=True)
            else:
                OpLog(
                    "Submission CSV 파일을 찾을 수 없습니다. 예측을 먼저 수행해주세요.",
                    bLines=True,
                )
                return

        # 시각화 및 리포트만 생성
        output_dir = f"submission_visualizations_{model_type}"
        visualize_submission_results(
            submission_csv_path=submission_path,
            output_dir=output_dir,
            max_images=10,
            random_sampling=True,
        )
        create_submission_report(submission_csv_path=submission_path)
        Lines(f"시각화 완료! 결과: {output_dir}")
        return

    Lines(f"평가 시작: {model_type}")

    # ════════════════════════════════════════
    # 1. 모델 로드
    # ════════════════════════════════════════

    model = None

    if "yolo" in model_type.lower():
        # YOLOv8 모델
        model_size = model_type.lower().replace("yolov8", "")
        model = YOLOv8Model(model_size=model_size)

        # 모델 가중치 경로 자동 탐색
        if model_weights_path is None:
            # YOLOv8는 학습 후 runs/detect/train/weights/best.pt에 저장됨
            possible_paths = [
                os.path.join(MODEL_FILES, f"yolov8{model_size}_best_model.pt"),
                os.path.join("runs", "detect", "train", "weights", "best.pt"),
                f"yolov8{model_size}.pt",  # 사전학습 모델
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    model_weights_path = path
                    OpLog(f"모델 가중치 발견: {path}", bLines=True)
                    break

        if model_weights_path and os.path.exists(model_weights_path):
            if not model.load_yolo_model(model_weights_path):
                OpLog("모델 로드 실패", bLines=True)
                return
        else:
            OpLog(
                f"경고: 학습된 모델을 찾을 수 없습니다. 사전학습 모델을 사용합니다.",
                bLines=True,
            )

    elif "faster" in model_type.lower() or "rcnn" in model_type.lower():
        # Faster R-CNN 모델
        model = FasterRCNNModel(backbone="resnet50")

        # 모델 가중치 경로 자동 탐색
        if model_weights_path is None:
            possible_paths = [
                os.path.join(MODEL_FILES, "FasterRCNNModel_resnet50_best_model.pth"),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    model_weights_path = path
                    OpLog(f"모델 가중치 발견: {path}", bLines=True)
                    break

        if model_weights_path and os.path.exists(model_weights_path):
            checkpoint = model.load_model(model_weights_path)
            if checkpoint:
                model.model.load_state_dict(checkpoint["model_state_dict"])
                model.model.to(DEVICE_TYPE)
                model.model.eval()
            else:
                OpLog("모델 로드 실패", bLines=True)
                return
        else:
            OpLog(f"경고: 학습된 모델을 찾을 수 없습니다.", bLines=True)
            return

    else:
        OpLog(f"지원하지 않는 모델 타입: {model_type}", bLines=True)
        return

    # ════════════════════════════════════════
    # 2. Validation 데이터로 mAP@[0.75:0.95] 평가
    # ════════════════════════════════════════

    if evaluate_on_val:
        Lines("Validation 데이터로 mAP@[0.75:0.95] 평가 시작")

        # DataLoader 생성
        train_loader, val_loader, test_loader = GetLoaders(
            annotations_dir=ANNOTATION_DIR,
            transform_type="default",  # 평가시에는 증강 없음
            img_dir=TRAIN_IMG_DIR,
            batch_size=16,
            train_ratio=0.8,
            num_workers=4,
        )

        # 개선된 evalModel 함수 사용
        if "yolo" in model_type.lower():
            # YOLOv8용 평가 함수 바인딩
            model.evalModel = (
                lambda val_loader, epoch, max_epochs: evalModel_improved_yolo(
                    model, val_loader, epoch, max_epochs
                )
            )

            val_loss = model.evalModel(val_loader, epoch=1, max_epochs=1)
            OpLog(f"Validation Loss: {val_loss:.4f}", bLines=True)

        elif "faster" in model_type.lower() or "rcnn" in model_type.lower():
            # Faster R-CNN용 평가 함수 바인딩
            model.evalModel = (
                lambda val_loader, epoch, max_epochs: evalModel_improved_fasterrcnn(
                    model, val_loader, epoch, max_epochs
                )
            )

            val_loss = model.evalModel(val_loader, epoch=1, max_epochs=1)
            OpLog(f"Validation Loss: {val_loss:.4f}", bLines=True)

    # ════════════════════════════════════════
    # 3. 테스트 데이터로 Submission CSV 생성
    # ════════════════════════════════════════

    Lines("테스트 데이터로 Submission CSV 생성 시작")

    submission_filename = f"submission_{model_type}_conf{confidence_threshold}.csv"

    submission_path = create_submission_csv_improved(
        model_type_to_use=model_type,
        model_weights_path=model_weights_path,
        submission_filename=submission_filename,
        confidence_threshold=confidence_threshold,
    )

    if submission_path:
        # ════════════════════════════════════════
        # 4. 결과 시각화
        # ════════════════════════════════════════

        Lines("결과 시각화 시작")

        # 예측 결과 시각화
        output_dir = f"submission_visualizations_{model_type}"
        visualize_submission_results(
            submission_csv_path=submission_path, output_dir=output_dir, max_images=10
        )

        # 통계 리포트 생성
        create_submission_report(submission_csv_path=submission_path)

        Lines(f"평가 완료! Submission 파일: {submission_path}")


# ════════════════════════════════════════
# 시각화만 따로 실행하는 함수
# ════════════════════════════════════════


def visualize_only(
    submission_csv_path, output_dir=None, max_images=10, random_sampling=True
):
    """
    기존 submission.csv 파일을 사용하여 시각화만 수행

    Args:
        submission_csv_path: submission CSV 파일 경로
        output_dir: 출력 디렉토리 (None이면 자동 생성)
        max_images: 최대 시각화할 이미지 수
        random_sampling: 랜덤 샘플링 여부
    """
    if not os.path.exists(submission_csv_path):
        OpLog(
            f"Submission CSV 파일을 찾을 수 없습니다: {submission_csv_path}",
            bLines=True,
        )
        return

    if output_dir is None:
        # CSV 파일명에서 디렉토리 이름 생성
        csv_basename = os.path.basename(submission_csv_path).replace(".csv", "")
        output_dir = f"visualizations_{csv_basename}"

    Lines(f"시각화 시작: {submission_csv_path}")

    # 시각화
    visualize_submission_results(
        submission_csv_path=submission_csv_path,
        output_dir=output_dir,
        max_images=max_images,
        random_sampling=random_sampling,
    )

    # 통계 리포트
    create_submission_report(submission_csv_path=submission_csv_path)

    Lines(f"시각화 완료! 결과: {output_dir}")


# ════════════════════════════════════════
# 실행 예시
# ════════════════════════════════════════

if __name__ == "__main__":

    # ═══════════════════════════════════════
    # 예시 1: 전체 파이프라인 (평가 + 예측 + 시각화)
    # ═══════════════════════════════════════
    # evaluate_and_create_submission(
    #     model_type="yolov8n",
    #     model_weights_path=None,  # 자동 탐색
    #     confidence_threshold=0.25,
    #     evaluate_on_val=True,
    # )

    # ═══════════════════════════════════════
    # 예시 2: 기존 CSV 파일로 시각화만 수행
    # ═══════════════════════════════════════
    visualize_only(
        submission_csv_path=r"D:\01.project\EntryPrj\data\submission_yolov8n_conf0.25.csv",
        max_images=10,
        output_dir=r"D:\01.project\EntryPrj\data\oraldrug\out_dir",
        random_sampling=True,
    )

    # ═══════════════════════════════════════
    # 예시 3: 예측 건너뛰고 시각화만
    # ═══════════════════════════════════════
    # evaluate_and_create_submission(
    #     model_type="yolov8n",
    #     skip_prediction=True,  # 예측 건너뛰기
    #     confidence_threshold=0.25
    # )

    # ═══════════════════════════════════════
    # 예시 4: 특정 CSV 파일 사용
    # ═══════════════════════════════════════
    # evaluate_and_create_submission(
    #     model_type="yolov8n",
    #     use_existing_csv="my_submission.csv"
    # )

    # ═══════════════════════════════════════
    # 예시 5: Faster R-CNN 모델 평가
    # ═══════════════════════════════════════
    # evaluate_and_create_submission(
    #     model_type="fasterrcnn_resnet50",
    #     model_weights_path=None,
    #     confidence_threshold=0.25,
    #     evaluate_on_val=True
    # )

    # 예시 3: 여러 confidence threshold로 테스트
    # for conf in [0.1, 0.25, 0.5]:
    #     evaluate_and_create_submission(
    #         model_type="yolov8s",
    #         confidence_threshold=conf,
    #         evaluate_on_val=False  # 한 번만 평가
    #     )
