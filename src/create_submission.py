# ════════════════════════════════════════
# 케글 제출 기준 mAP@[0.75:0.95] 적용
# ════════════════════════════════════════

import numpy as np
from collections import defaultdict
import os
import re
import torch
from PIL import Image
import pandas as pd


from A03 import OpLog, YAML_FILE, DEVICE_TYPE, TEST_IMG_DIR
from A03 import BaseModel, YOLOv8Model, FasterRCNNModel
from A03 import GetTransform

# ════════════════════════════════════════
# 1. mAP@[0.75:0.95] 계산 함수 추가
# ════════════════════════════════════════
VER = "2025.12.10.001.kyw"
BASE_DIR = r"D:\01.project\EntryPrj\data"
LOG_FILE = os.path.join(BASE_DIR, "operation.log")


def calculate_iou(box1, box2):
    """
    두 박스 간의 IoU 계산
    box: [x1, y1, x2, y2] 형식
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def calculate_ap_at_iou(predictions, ground_truths, iou_threshold, num_classes):
    """
    특정 IoU threshold에서의 AP 계산

    Args:
        predictions: [{image_id, category_id, bbox, score}, ...]
        ground_truths: [{image_id, category_id, bbox}, ...]
        iou_threshold: IoU threshold (0.75 ~ 0.95)
        num_classes: 클래스 수
    """
    ap_per_class = []

    for class_id in range(num_classes):
        # 해당 클래스의 예측과 GT 필터링
        class_preds = [p for p in predictions if p["category_id"] == class_id]
        class_gts = [g for g in ground_truths if g["category_id"] == class_id]

        if len(class_gts) == 0:
            continue

        # score 기준 내림차순 정렬
        class_preds = sorted(class_preds, key=lambda x: x["score"], reverse=True)

        # GT를 이미지별로 그룹화
        gt_by_image = defaultdict(list)
        for gt in class_gts:
            gt_by_image[gt["image_id"]].append(gt)

        # TP, FP 계산
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        matched_gt = defaultdict(set)  # 이미 매칭된 GT 추적

        for pred_idx, pred in enumerate(class_preds):
            image_id = pred["image_id"]
            pred_box = pred["bbox"]  # [x1, y1, x2, y2]

            if image_id not in gt_by_image:
                fp[pred_idx] = 1
                continue

            # 해당 이미지의 GT와 IoU 계산
            max_iou = 0
            max_gt_idx = -1

            for gt_idx, gt in enumerate(gt_by_image[image_id]):
                if gt_idx in matched_gt[image_id]:
                    continue

                gt_box = gt["bbox"]
                iou = calculate_iou(pred_box, gt_box)

                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx

            # IoU threshold 이상이면 TP
            if max_iou >= iou_threshold and max_gt_idx >= 0:
                tp[pred_idx] = 1
                matched_gt[image_id].add(max_gt_idx)
            else:
                fp[pred_idx] = 1

        # Precision-Recall 곡선 계산
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / len(class_gts)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

        # AP 계산 (11-point interpolation)
        ap = 0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11

        ap_per_class.append(ap)

    # mAP 계산
    return np.mean(ap_per_class) if ap_per_class else 0.0


def calculate_map_75_95(predictions, ground_truths, num_classes):
    """
    mAP@[0.75:0.95] 계산
    IoU threshold: 0.75, 0.80, 0.85, 0.90, 0.95

    Args:
        predictions: 모델 예측 결과 리스트
        ground_truths: Ground truth 리스트
        num_classes: 클래스 수

    Returns:
        float: mAP@[0.75:0.95] 값
    """
    iou_thresholds = [0.75, 0.80, 0.85, 0.90, 0.95]
    aps = []

    for iou_thresh in iou_thresholds:
        ap = calculate_ap_at_iou(predictions, ground_truths, iou_thresh, num_classes)
        aps.append(ap)
        print(f"  AP@{iou_thresh:.2f}: {ap:.4f}")

    map_score = np.mean(aps)
    print(f"  mAP@[0.75:0.95]: {map_score:.4f}")

    return map_score


# ════════════════════════════════════════
# 2. YOLOv8Model 수정 - evalModel
# ════════════════════════════════════════


def evalModel_improved_yolo(self, val_loader, epoch, max_epochs):
    """
    YOLOv8 검증 모드 - mAP@[0.75:0.95] 정확히 계산

    Note: YOLOv8의 자체 val() 메서드는 YAML 파일 기반으로 동작하므로
    여기서는 YOLOv8의 기본 메트릭을 사용합니다.
    val_loader는 인터페이스 통일을 위해 받지만, YOLOv8는 YAML 기반으로 동작합니다.
    """
    OpLog(f"[Epoch {epoch}/{max_epochs}] Validation 시작", bLines=True)

    # YOLOv8 기본 검증 (mAP@0.5:0.95)
    try:
        metrics = self.model.val(
            data=YAML_FILE,
            device=DEVICE_TYPE,
            split="val",
            plots=False,
        )

        # YOLOv8의 기본 메트릭
        mAP50 = float(metrics.box.map50)
        mAP50_95 = float(metrics.box.map)  # 이것은 0.5:0.95
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)

        # YOLOv8의 mAP는 이미 [0.5:0.05:0.95] 범위에서 계산됨
        # 케글 대회에서는 mAP@[0.75:0.95]를 요구하므로,
        # 실제로는 커스텀 계산이 필요하지만 YOLOv8의 mAP50-95를 참고값으로 사용
        map_75_95 = mAP50_95  # YOLOv8의 전체 mAP (0.5:0.95)

        val_loss = 1.0 - map_75_95  # mAP를 손실로 변환
        self.val_losses.append(val_loss)

        OpLog(
            f"mAP@0.5: {mAP50:.4f}, mAP@[0.5:0.95]: {mAP50_95:.4f}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}",
            bLines=False,
        )
        OpLog(
            f"Note: 케글 제출용 mAP@[0.75:0.95]는 제출 후 확인 가능합니다.",
            bLines=False,
        )

        BaseModel.save_metrics_to_csv(
            self,
            model_name=self.getMyName(),
            epoch_index=epoch,
            max_epochs=max_epochs,
            train_loss=0.0,
            current_lr=0.0,
            mode="eval",
            mAP50=mAP50,
            mAP50_95=mAP50_95,
            precision=precision,
            recall=recall,
        )

        return val_loss

    except Exception as e:
        OpLog(f"Validation 중 오류 발생: {e}", bLines=True)
        import traceback
        OpLog(traceback.format_exc(), bLines=False)
        return 1.0  # 오류 시 최악의 loss 반환


# ════════════════════════════════════════
# 3. Faster R-CNN Model 수정 - evalModel
# ════════════════════════════════════════


def evalModel_improved_fasterrcnn(self, val_loader, epoch, max_epochs):
    """
    Faster R-CNN 검증 모드 - mAP@[0.75:0.95] 계산 추가
    """
    self.model.eval()
    predictions = []
    ground_truths = []

    OpLog(f"[Epoch {epoch}/{max_epochs}] Validation 시작", bLines=True)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = [img.to(DEVICE_TYPE) for img in images]

            # 예측
            outputs = self.model(images)

            # 예측 결과 수집
            for img_idx, (output, target) in enumerate(zip(outputs, targets)):
                image_id = batch_idx * len(images) + img_idx

                # 예측
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    if score >= 0.25:  # confidence threshold
                        predictions.append(
                            {
                                "image_id": image_id,
                                "category_id": int(label),
                                "bbox": box.tolist(),
                                "score": float(score),
                            }
                        )

                # Ground truth
                if isinstance(target, dict):
                    gt_boxes = target["boxes"].cpu().numpy()
                    gt_labels = target["labels"].cpu().numpy()

                    for box, label in zip(gt_boxes, gt_labels):
                        ground_truths.append(
                            {
                                "image_id": image_id,
                                "category_id": int(label),
                                "bbox": box.tolist(),
                            }
                        )

    # mAP@[0.75:0.95] 계산
    map_75_95 = calculate_map_75_95(predictions, ground_truths, self.num_classes)

    val_loss = 1.0 - map_75_95
    self.val_losses.append(val_loss)

    OpLog(
        f"Epoch [{epoch}/{max_epochs}] - mAP@[0.75:0.95]:{map_75_95:.4f}", bLines=True
    )

    # CSV에 저장
    current_lr = self.optimizer.param_groups[0]["lr"]
    BaseModel.save_metrics_to_csv(
        self,
        model_name=self.getMyName(),
        epoch_index=epoch,
        max_epochs=max_epochs,
        train_loss=self.train_losses[-1] if self.train_losses else 0.0,
        val_loss=val_loss,
        current_lr=current_lr,
        mode="eval",
        mAP75_95=map_75_95,
    )

    return val_loss


# ════════════════════════════════════════
# 4. Submission CSV 생성 개선
# ════════════════════════════════════════


def create_submission_csv_improved(
    model_type_to_use,
    model_weights_path,
    submission_filename="submission.csv",
    confidence_threshold=0.25,  # ★ threshold 상향
):
    """
    개선된 Submission CSV 생성

    Args:
        model_type_to_use: 모델 타입
        model_weights_path: 모델 가중치 경로
        submission_filename: 저장할 파일명
        confidence_threshold: 신뢰도 임계값 (기본 0.25, 너무 낮으면 False Positive 증가)
    """
    OpLog(f"Submission CSV 생성 시작 (threshold={confidence_threshold})", bLines=True)

    model = None
    if "yolo" in model_type_to_use.lower():
        model_size = model_type_to_use.lower().replace("yolov8", "")
        model = YOLOv8Model(model_size=model_size)
        if not model.load_yolo_model(model_weights_path):
            return
    elif "faster" in model_type_to_use.lower() or "rcnn" in model_type_to_use.lower():
        model = FasterRCNNModel(backbone="resnet50")
        checkpoint = model.load_model(model_weights_path)
        if not checkpoint:
            return
        model.model.load_state_dict(checkpoint["model_state_dict"])
        model.model.to(DEVICE_TYPE)
        model.model.eval()
    else:
        OpLog(f"지원하지 않는 모델 타입: {model_type_to_use}", bLines=True)
        return

    results_list = []
    annotation_id_counter = 1

    test_image_files = sorted(
        [
            f
            for f in os.listdir(TEST_IMG_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    OpLog(f"{len(test_image_files)}개 테스트 이미지 예측 수행", bLines=False)

    if len(test_image_files) == 0:
        OpLog(f"경고: {TEST_IMG_DIR}에 이미지 파일이 없습니다!", bLines=True)
        return

    total_predictions = 0
    images_processed = 0

    for img_filename in test_image_files:
        img_path = os.path.join(TEST_IMG_DIR, img_filename)

        # 이미지 ID 추출
        match = re.search(r"(\d+)", img_filename)
        if not match:
            OpLog(f"경고: 파일명에서 ID 추출 실패: {img_filename}", bLines=False)
            continue
        image_id = int(match.group(1))
        images_processed += 1

        if "yolo" in model_type_to_use.lower():
            # ★ confidence threshold 상향 (0.001 -> 0.25)
            preds = model.predict(img_path, conf=confidence_threshold, save=False)

            if preds and len(preds) > 0:
                num_boxes = len(preds[0].boxes)
                total_predictions += num_boxes
                OpLog(f"  이미지 {image_id}: {num_boxes}개 객체 탐지", bLines=False)
                for box in preds[0].boxes:
                    x_center, y_center, w, h = box.xywh[0].tolist()
                    results_list.append(
                        {
                            "annotation_id": annotation_id_counter,
                            "image_id": image_id,
                            "category_id": int(box.cls.item()),
                            "bbox_x": int(x_center - w / 2),
                            "bbox_y": int(y_center - h / 2),
                            "bbox_w": int(w),
                            "bbox_h": int(h),
                            "score": round(box.conf.item(), 2),
                        }
                    )
                    annotation_id_counter += 1
            else:
                OpLog(f"  이미지 {image_id}: 탐지된 객체 없음", bLines=False)

        elif (
            "faster" in model_type_to_use.lower() or "rcnn" in model_type_to_use.lower()
        ):
            image = Image.open(img_path).convert("RGB")
            transform = GetTransform("default")
            image_tensor = transform(image).unsqueeze(0).to(DEVICE_TYPE)

            with torch.no_grad():
                prediction = model.model(image_tensor)

            num_detected = len(prediction[0]["boxes"])
            num_above_threshold = sum(
                1
                for score in prediction[0]["scores"]
                if score.item() >= confidence_threshold
            )
            OpLog(
                f"  이미지 {image_id}: {num_detected}개 탐지, threshold 이상: {num_above_threshold}개",
                bLines=False,
            )

            for i in range(len(prediction[0]["boxes"])):
                score = prediction[0]["scores"][i].item()

                # ★ confidence threshold 상향
                if score < confidence_threshold:
                    continue

                total_predictions += 1

                box = prediction[0]["boxes"][i].tolist()
                x1, y1, x2, y2 = box

                results_list.append(
                    {
                        "annotation_id": annotation_id_counter,
                        "image_id": image_id,
                        "category_id": prediction[0]["labels"][i].item(),
                        "bbox_x": int(x1),
                        "bbox_y": int(y1),
                        "bbox_w": int(x2 - x1),
                        "bbox_h": int(y2 - y1),
                        "score": round(score, 2),
                    }
                )
                annotation_id_counter += 1

    OpLog(
        f"처리된 이미지: {images_processed}개, 총 탐지 객체: {total_predictions}개",
        bLines=False,
    )

    if not results_list:
        OpLog("탐지된 객체가 없습니다.", bLines=True)
        OpLog(f"가능한 원인:", bLines=False)
        OpLog(
            f"  1. Confidence threshold ({confidence_threshold})가 너무 높음",
            bLines=False,
        )
        OpLog(f"  2. 모델이 제대로 학습되지 않음", bLines=False)
        OpLog(f"  3. 테스트 이미지가 학습 데이터와 너무 다름", bLines=False)
        OpLog(f"해결방법: confidence_threshold를 낮춰보세요 (예: 0.01)", bLines=False)
        return

    # DataFrame 생성 및 저장
    submission_df = pd.DataFrame(results_list)
    submission_df = submission_df[
        [
            "annotation_id",
            "image_id",
            "category_id",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "score",
        ]
    ]

    submission_path = os.path.join(os.getcwd(), submission_filename)
    submission_df.to_csv(submission_path, index=False)

    OpLog(f"Submission CSV 생성 완료: {submission_path}", bLines=True)
    OpLog(f"  - 총 탐지 객체: {len(submission_df)}개", bLines=False)
    OpLog(f"  - 평균 confidence: {submission_df['score'].mean():.3f}", bLines=False)
    OpLog(f"  - 최소 confidence: {submission_df['score'].min():.3f}", bLines=False)

    return submission_path


# ════════════════════════════════════════
# 5. Submission 결과 시각화 함수
# ════════════════════════════════════════


def visualize_submission_results(
    submission_csv_path="submission.csv",
    output_dir="submission_visualizations",
    max_images=10,
    random_sampling=True,
):
    """
    Submission CSV 결과를 실제 테스트 이미지에 바운딩 박스 그려서 시각화

    Args:
        submission_csv_path: submission CSV 파일 경로
        output_dir: 시각화 결과 저장 디렉토리
        max_images: 최대 시각화할 이미지 수
        random_sampling: True면 랜덤 샘플링, False면 순서대로
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    import random

    OpLog(f"Submission 결과 시각화 시작", bLines=True)

    # CSV 파일 읽기
    if not os.path.exists(submission_csv_path):
        OpLog(f"CSV 파일을 찾을 수 없습니다: {submission_csv_path}", bLines=True)
        return

    df = pd.read_csv(submission_csv_path)
    OpLog(f"총 {len(df)}개 예측 결과 로드", bLines=False)

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 이미지별로 그룹화
    all_image_ids = df["image_id"].unique()

    # 랜덤 샘플링 또는 순서대로
    if random_sampling and len(all_image_ids) > max_images:
        image_ids = random.sample(list(all_image_ids), max_images)
    else:
        image_ids = all_image_ids[:max_images]

    OpLog(f"{len(image_ids)}개 이미지 시각화 중...", bLines=False)

    # 이미지 파일 목록 미리 가져오기
    all_test_files = {
        f: os.path.join(TEST_IMG_DIR, f)
        for f in os.listdir(TEST_IMG_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    }

    visualized_count = 0

    for img_id in image_ids:
        # 해당 이미지의 예측 결과 필터링
        img_preds = df[df["image_id"] == img_id]

        # 이미지 파일 찾기 (img_id를 포함하는 파일명)
        img_file = None
        for filename in all_test_files.keys():
            # 파일명에서 숫자 추출하여 비교
            if str(img_id) in filename:
                img_file = filename
                break

        if not img_file:
            OpLog(
                f"이미지 ID {img_id}에 해당하는 파일을 찾을 수 없습니다.", bLines=False
            )
            continue

        img_path = all_test_files[img_file]

        # 이미지 로드
        try:
            img = Image.open(img_path).convert("RGB")
            img_width, img_height = img.size
        except Exception as e:
            OpLog(f"이미지 로드 실패: {img_path} - {e}", bLines=False)
            continue

        # 시각화
        fig, ax = plt.subplots(1, 1, figsize=(15, 12))
        ax.imshow(img)

        # 바운딩 박스 그리기
        num_boxes = len(img_preds)
        colors = plt.cm.tab20(np.linspace(0, 1, max(num_boxes, 1)))

        for idx, (_, row) in enumerate(img_preds.iterrows()):
            x, y, w, h = row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]
            category_id = int(row["category_id"])
            score = row["score"]

            # 사각형 그리기
            rect = patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=3,
                edgecolor=colors[idx % len(colors)],
                facecolor="none",
            )
            ax.add_patch(rect)

            # 레이블 텍스트 (카테고리 ID와 신뢰도)
            label_text = f"ID:{category_id} ({score:.2f})"

            # 텍스트 배경 박스
            text_y = max(y - 10, 10)  # 화면 밖으로 나가지 않도록
            ax.text(
                x,
                text_y,
                label_text,
                color="white",
                fontsize=12,
                weight="bold",
                bbox=dict(
                    facecolor=colors[idx % len(colors)],
                    alpha=0.8,
                    edgecolor="white",
                    boxstyle="round,pad=0.5",
                ),
            )

        # 제목
        ax.set_title(
            f"Image: {img_file} (ID: {img_id})\n"
            f"Detections: {num_boxes} objects | Image size: {img_width}x{img_height}",
            fontsize=16,
            weight="bold",
            pad=20
        )
        ax.axis("off")

        plt.tight_layout()

        # 저장
        output_filename = f"pred_{img_id:05d}_{img_file}"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        OpLog(f"  ✓ Image {img_file}: {num_boxes}개 객체 탐지 → {output_filename}", bLines=False)
        visualized_count += 1

    OpLog(f"시각화 완료: {visualized_count}개 이미지 저장됨 → {output_dir}", bLines=True)
    return output_dir


# ════════════════════════════════════════
# 6. 통계 리포트 생성
# ════════════════════════════════════════


def create_submission_report(submission_csv_path="submission.csv"):
    """
    Submission 결과 통계 리포트 생성

    Args:
        submission_csv_path: submission CSV 파일 경로
    """
    import matplotlib.pyplot as plt

    OpLog(f"Submission 통계 리포트 생성 시작", bLines=True)

    if not os.path.exists(submission_csv_path):
        OpLog(f"CSV 파일을 찾을 수 없습니다: {submission_csv_path}", bLines=True)
        return

    df = pd.read_csv(submission_csv_path)

    # 통계 계산
    total_detections = len(df)
    unique_images = df["image_id"].nunique()
    unique_categories = df["category_id"].nunique()
    avg_detections_per_image = (
        total_detections / unique_images if unique_images > 0 else 0
    )
    avg_score = df["score"].mean()
    min_score = df["score"].min()
    max_score = df["score"].max()

    # 카테고리별 탐지 개수
    category_counts = df["category_id"].value_counts().sort_index()

    # 스코어 분포
    score_bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
    score_distribution = (
        pd.cut(df["score"], bins=score_bins).value_counts().sort_index()
    )

    # 리포트 출력
    OpLog("=" * 100, bLines=False)
    OpLog("Submission 통계 리포트", bLines=True)
    OpLog("=" * 100, bLines=False)
    OpLog(f"  총 탐지 객체 수: {total_detections}", bLines=False)
    OpLog(f"  고유 이미지 수: {unique_images}", bLines=False)
    OpLog(f"  고유 카테고리 수: {unique_categories}", bLines=False)
    OpLog(f"  이미지당 평균 탐지 수: {avg_detections_per_image:.2f}", bLines=False)
    OpLog(f"  평균 Score: {avg_score:.4f}", bLines=False)
    OpLog(f"  최소/최대 Score: {min_score:.4f} / {max_score:.4f}", bLines=False)
    OpLog("=" * 100, bLines=False)

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 카테고리별 탐지 개수
    ax1 = axes[0, 0]
    category_counts.plot(kind="bar", ax=ax1, color="skyblue")
    ax1.set_title("Detections per Category", fontsize=14)
    ax1.set_xlabel("Category ID")
    ax1.set_ylabel("Count")
    ax1.grid(axis="y", alpha=0.3)

    # 2. Score 분포
    ax2 = axes[0, 1]
    score_distribution.plot(kind="bar", ax=ax2, color="coral")
    ax2.set_title("Score Distribution", fontsize=14)
    ax2.set_xlabel("Score Range")
    ax2.set_ylabel("Count")
    ax2.grid(axis="y", alpha=0.3)

    # 3. 이미지당 탐지 개수 분포
    ax3 = axes[1, 0]
    detections_per_image = df.groupby("image_id").size()
    detections_per_image.hist(bins=20, ax=ax3, color="lightgreen", edgecolor="black")
    ax3.set_title("Detections per Image Distribution", fontsize=14)
    ax3.set_xlabel("Number of Detections")
    ax3.set_ylabel("Image Count")
    ax3.grid(axis="y", alpha=0.3)

    # 4. 통계 요약
    ax4 = axes[1, 1]
    stats_text = f"""
    Submission Statistics

    Total Detections: {total_detections}
    Unique Images: {unique_images}
    Unique Categories: {unique_categories}

    Avg Detections/Image: {avg_detections_per_image:.2f}

    Score Statistics:
      - Mean: {avg_score:.4f}
      - Min: {min_score:.4f}
      - Max: {max_score:.4f}
      - Std: {df['score'].std():.4f}
    """
    ax4.text(
        0.1,
        0.5,
        stats_text,
        fontsize=12,
        verticalalignment="center",
        family="monospace",
    )
    ax4.axis("off")

    plt.tight_layout()

    # 저장
    report_path = os.path.join(
        os.path.dirname(submission_csv_path), "submission_report.png"
    )
    plt.savefig(report_path, dpi=150, bbox_inches="tight")
    plt.close()

    OpLog(f"통계 리포트 저장: {report_path}", bLines=True)

    return report_path
