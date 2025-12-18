import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import datetime
import sys
from filelock import FileLock
import matplotlib.pyplot as plt
import torch.optim as optim
from ultralytics import YOLO
import pandas as pd

# ════════════════════════════════════════
# ▣ 01. 디렉토리 및 유틸 함수 설정
# ════════════════════════════════════════
VER = "2025.12.10.001.kyw"
BASE_DIR = "/content/drive/MyDrive/codeit/data"
BASE_DIR = r"D:\01.project\EntryPrj\data"
LOG_FILE = os.path.join(BASE_DIR, "operation.log")

# 단계별 학습을 위한 데이터 경로 설정
STAGE_CONFIGS = {
    "stage1": {
        "name": "1.drug_Image_annotation_allOK",  # 공백 제거
        "annotation_dir": os.path.join(
            BASE_DIR, "oraldrug", "1.drug_Image_annotation_allOK", "train_annotations"
        ),
        "img_dir": os.path.join(
            BASE_DIR, "oraldrug", "1.drug_Image_annotation_allOK", "train_images"
        ),
        "description": "완벽한 데이터 (이미지 + 어노테이션)",
    },
    "stage2": {
        "name": "2.drug_no_image_ok_Anno",  # 공백 제거
        "annotation_dir": os.path.join(
            BASE_DIR, "oraldrug", "2.drug_no_image_ok_Anno", "train_annotations"
        ),
        "img_dir": os.path.join(
            BASE_DIR, "oraldrug", "2.drug_no_image_ok_Anno", "train_images"
        ),
        "description": "어노테이션만 있는 데이터",
    },
    "stage3": {
        "name": "3.drug_ok_Image_no_Anno",  # 공백 제거
        "annotation_dir": os.path.join(
            BASE_DIR, "oraldrug", "3.drug_ok_Image_no_Anno", "train_annotations"
        ),
        "img_dir": os.path.join(
            BASE_DIR, "oraldrug", "3.drug_ok_Image_no_Anno", "train_images"
        ),
        "description": "이미지만 있는 데이터",
    },
}

# 기본 경로 (하위 호환성 유지 - Stage 1이 기본)
ANNOTATION_DIR = STAGE_CONFIGS["stage1"]["annotation_dir"]
TRAIN_IMG_DIR = STAGE_CONFIGS["stage1"]["img_dir"]

# 공통 경로
TEST_IMG_DIR = os.path.join(BASE_DIR, "oraldrug", "test_images")
YAML_FILE = os.path.join(BASE_DIR, "oraldrug", "yolo_yaml.yaml")
MODEL_FILES = os.path.join(BASE_DIR, "oraldrug", "models")
RESULT_CSV = f"{BASE_DIR}/entryprj.csv"
IMG_TO_JSON = os.path.join(BASE_DIR, "oraldrug", "img_to_json.csv")
JSON_TO_IMG = os.path.join(BASE_DIR, "oraldrug", "json_to_img.csv")
DEVICE_TYPE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## 구분선 출력 함수
def Lines(text="", count=100):
    print("═" * count)
    if text != "":
        print(f"{text}")
        print("═" * count)


## 현재 시간 문자열 반환 함수
def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


## 디렉토리 생성 함수
def makedirs(d):
    os.makedirs(d, exist_ok=True)


## 운영 로그 함수
def OpLog(log, bLines=True):
    if bLines:
        Lines(f"[{now_str()}] {log}")
    try:
        caller_name = sys._getframe(1).f_code.co_name
    except Exception:
        caller_name = "UnknownFunction"

    log_filename = LOG_FILE
    log_lock_filename = log_filename + ".lock"
    log_content = f"[{now_str()}] {caller_name}: {log}\n"
    try:
        lock = FileLock(log_lock_filename, timeout=10)
        with lock:
            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(log_content)
    except Exception as e:
        print(f"Log write error: {e}")

OpLog(f"Start program.{VER}")

# ════════════════════════════════════════
# ▣ 02. 클래스 수 계산 및 클래스 매핑 생성
# ════════════════════════════════════════
# train_annotations에서 JSON 파일을 읽어서 category_id로 클래스 수 계산
def count_classes(annotations_dir):
    """
    모든 서브 디렉토리에서 JSON을 읽어서 category_id를 classes로 카운트
    os.walk()를 사용하여 모든 하위 디렉토리를 재귀적으로 검사

    Returns:
        int: 고유 category_id 개수
    """
    unique_category_ids = set()

    # os.walk()로 모든 하위 디렉토리 재귀적으로 탐색
    for root, dirs, files in os.walk(annotations_dir):
        for json_file in files:
            if json_file.endswith(".json"):
                json_path = os.path.join(root, json_file)
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # category_id 추출
                    if "annotations" in data and len(data["annotations"]) > 0:
                        for ann in data["annotations"]:
                            category_id = ann.get("category_id")
                            if category_id is not None:
                                unique_category_ids.add(category_id)
                except Exception as e:
                    continue

    return len(unique_category_ids)


def get_class_mapping(annotations_dir):
    """
    annotation 디렉토리에서 JSON을 읽어 category_id 기반으로 클래스 매핑 정보를 반환
    os.walk()를 사용하여 모든 하위 디렉토리를 재귀적으로 검사

    Args:
        annotations_dir: annotation 디렉토리 경로

    Returns:
        tuple: (class_dirs, class_to_idx, idx_to_class, unique_classes)
            - class_dirs: [(category_id, [json_paths]), ...] 리스트
            - class_to_idx: {category_id: index} 딕셔너리 (category_id를 그대로 사용)
            - idx_to_class: {index: category_id} 딕셔너리
            - unique_classes: 정렬된 고유 category_id 리스트
    """
    class_info = {}  # {category_id: [json_path1, json_path2, ...]}
    class_dirs = []

    # os.walk()로 모든 하위 디렉토리 재귀적으로 탐색
    json_count = 0
    for root, dirs, files in os.walk(annotations_dir):
        for json_file in files:
            if json_file.endswith(".json"):
                json_path = os.path.join(root, json_file)
                json_count += 1
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if "annotations" in data and len(data["annotations"]) > 0:
                        for ann in data["annotations"]:
                            category_id = ann.get("category_id")
                            if category_id is not None:
                                # 동일 category_id가 여러 JSON 파일에 있을 수 있으므로 모두 저장
                                if category_id not in class_info:
                                    class_info[category_id] = []
                                if json_path not in class_info[category_id]:
                                    class_info[category_id].append(json_path)
                                break  # 한 JSON에서 category_id 하나만 찾으면 됨
                except Exception as e:
                    OpLog(f"Error reading {json_path}: {e}", bLines=False)
                    continue

    OpLog(f"get_class_mapping: 총 {json_count}개 JSON 파일 스캔, {len(class_info)}개 클래스 발견", bLines=False)

    # class_dirs 생성 (category_id, JSON 파일 목록)
    for category_id, json_paths in class_info.items():
        class_dirs.append((category_id, json_paths))

    # 클래스 정렬 및 인덱스 매핑 (category_id 기준)
    unique_classes = sorted(class_info.keys())
    class_to_idx = {category_id: category_id for category_id in unique_classes}  # category_id를 그대로 index로 사용
    idx_to_class = {category_id: category_id for category_id in unique_classes}

    return class_dirs, class_to_idx, idx_to_class, unique_classes


Lines(f"ANNOTATION_DIR: {ANNOTATION_DIR}")

# category_id를 dl_idx로 업데이트 (최초 1회만 실행 필요)
# change_category_id 함수가 eda.py에 있다면 여기서 직접 구현
def update_category_id_from_dl_idx(annotations_dir):
    """
    JSON 파일의 category_id를 images[].dl_idx 값으로 업데이트
    """
    updated_count = 0
    skipped_count = 0
    
    for root, dirs, files in os.walk(annotations_dir):
        for fname in files:
            if not fname.lower().endswith('.json'):
                continue
            
            json_path = os.path.join(root, fname)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # dl_idx 추출
                dl_idx = None
                if 'images' in data and isinstance(data['images'], list) and len(data['images']) > 0:
                    dl_idx_str = data['images'][0].get('dl_idx')
                    if dl_idx_str:
                        try:
                            dl_idx = int(dl_idx_str)
                        except:
                            dl_idx = None
                
                if dl_idx is None:
                    skipped_count += 1
                    continue
                
                # annotations의 category_id 업데이트
                modified = False
                if 'annotations' in data and isinstance(data['annotations'], list):
                    for ann in data['annotations']:
                        if isinstance(ann, dict):
                            category_id = ann.get('category_id')
                            if category_id != dl_idx:
                                ann['category_id'] = dl_idx
                                modified = True
                
                if modified:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    updated_count += 1
                else:
                    skipped_count += 1
                    
            except Exception as e:
                OpLog(f"Error updating {json_path}: {e}", bLines=False)
                continue
    
    if updated_count > 0:
        OpLog(f"category_id 업데이트 완료: updated={updated_count}, skipped={skipped_count}", bLines=True)
    
    return updated_count

# category_id 업데이트 실행 (필요시)
update_count = update_category_id_from_dl_idx(ANNOTATION_DIR)

num_classes = count_classes(ANNOTATION_DIR)
OpLog(f"총 클래스 수 (category_id 기준): {num_classes}", bLines=True)


def analyze_image_json_mapping():
    """
    이미지와 JSON 파일 간의 매핑 관계를 분석하여 CSV 파일로 저장

    생성 파일:
        - IMG_TO_JSON: 이미지 -> JSON 매핑 (이미지명, JSON경로)
        - JSON_TO_IMG: JSON -> 이미지 매핑 (JSON경로, 이미지명)
    """
    OpLog("이미지-JSON 매핑 분석 시작", bLines=True)

    # 1. TRAIN_IMG_DIR 밑의 모든 이미지 파일 수집
    image_files = {}  # {filename: full_path}
    if os.path.exists(TRAIN_IMG_DIR):
        for img_file in os.listdir(TRAIN_IMG_DIR):
            if img_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                image_files[img_file] = os.path.join(TRAIN_IMG_DIR, img_file)

    OpLog(f"이미지 파일 수: {len(image_files)}", bLines=False)

    # 2. ANNOTATION_DIR 밑의 모든 JSON 파일 수집
    json_files = []  # [(json_path, images_in_json), ...]
    if os.path.exists(ANNOTATION_DIR):
        for subdir in os.listdir(ANNOTATION_DIR):
            subdir_path = os.path.join(ANNOTATION_DIR, subdir)
            if os.path.isdir(subdir_path):
                for class_dir in os.listdir(subdir_path):
                    class_dir_path = os.path.join(subdir_path, class_dir)
                    if os.path.isdir(class_dir_path):
                        for json_file in os.listdir(class_dir_path):
                            if json_file.endswith(".json"):
                                json_path = os.path.join(class_dir_path, json_file)
                                # JSON 파일에서 이미지 정보 추출
                                try:
                                    with open(json_path, "r", encoding="utf-8") as f:
                                        data = json.load(f)

                                    images_in_json = []
                                    if "images" in data:
                                        for img_info in data["images"]:
                                            img_filename = img_info.get("file_name", "")
                                            if img_filename:
                                                images_in_json.append(img_filename)

                                    json_files.append((json_path, images_in_json))
                                except Exception as e:
                                    OpLog(
                                        f"JSON 파일 읽기 오류 {json_path}: {e}",
                                        bLines=False,
                                    )

    OpLog(f"JSON 파일 수: {len(json_files)}", bLines=False)

    # 3. IMG_TO_JSON 매핑 생성
    img_to_json_mapping = []  # [(img_name, json_path), ...]

    for img_name in sorted(image_files.keys()):
        found_jsons = []

        # 이 이미지를 포함하는 JSON 파일 찾기
        for json_path, images_in_json in json_files:
            if img_name in images_in_json:
                found_jsons.append(json_path)

        if found_jsons:
            for json_path in found_jsons:
                img_to_json_mapping.append((img_name, json_path))
        else:
            # JSON 파일이 없는 이미지
            img_to_json_mapping.append((img_name, "NONE"))

    # 4. JSON_TO_IMG 매핑 생성
    json_to_img_mapping = []  # [(json_path, img_name), ...]

    for json_path, images_in_json in json_files:
        if images_in_json:
            for img_name in images_in_json:
                # 실제 이미지 파일이 존재하는지 확인
                if img_name in image_files:
                    json_to_img_mapping.append((json_path, img_name))
                else:
                    json_to_img_mapping.append((json_path, "NONE"))
        else:
            # 이미지 정보가 없는 JSON
            json_to_img_mapping.append((json_path, "NONE"))

    # 5. IMG_TO_JSON CSV 파일 저장
    makedirs(os.path.dirname(IMG_TO_JSON))
    with open(IMG_TO_JSON, "w", encoding="utf-8") as f:
        f.write("Image,JSON\n")
        for img_name, json_path in img_to_json_mapping:
            f.write(f"{img_name},{json_path}\n")

    OpLog(
        f"IMG_TO_JSON 저장 완료: {IMG_TO_JSON} ({len(img_to_json_mapping)}개 매핑)",
        bLines=False,
    )

    # 6. JSON_TO_IMG CSV 파일 저장
    with open(JSON_TO_IMG, "w", encoding="utf-8") as f:
        f.write("JSON,Image\n")
        for json_path, img_name in json_to_img_mapping:
            f.write(f"{json_path},{img_name}\n")

    OpLog(
        f"JSON_TO_IMG 저장 완료: {JSON_TO_IMG} ({len(json_to_img_mapping)}개 매핑)",
        bLines=False,
    )

    # 7. 통계 정보 출력
    img_without_json = sum(
        1 for _, json_path in img_to_json_mapping if json_path == "NONE"
    )
    json_without_img = sum(
        1 for _, img_name in json_to_img_mapping if img_name == "NONE"
    )

    OpLog(f"매핑 분석 완료:", bLines=True)
    OpLog(f"  - 전체 이미지: {len(image_files)}개", bLines=False)
    OpLog(f"  - 전체 JSON: {len(json_files)}개", bLines=False)
    OpLog(f"  - JSON 없는 이미지: {img_without_json}개", bLines=False)
    OpLog(f"  - 이미지 없는 JSON: {json_without_img}개", bLines=False)

    return img_to_json_mapping, json_to_img_mapping


# ════════════════════════════════════════
# ▣ 03. 데이터셋 및 데이터 증강 함수 정의
# ════════════════════════════════════════
# 다양한 데이터 증강(transform) 함수 정의
def GetTransform(transform_type="default"):
    if transform_type == "default":
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
    if transform_type == "A":
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
                ),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
            ]
        )
    elif transform_type == "B":
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
                transforms.ToTensor(),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )


# 커스텀 데이터셋 클래스 정의
class PillDataset(Dataset):
    def __init__(self, annotations_dir, img_dir, transform=None, is_test=False):
        """
        Args:
            annotations_dir: train_annotations 경로 (is_test=True일 경우 무시됨)
            img_dir: train_images 또는 test_images 경로
            transform: 이미지 변환 함수
            is_test: 테스트 데이터셋 여부 (True이면 annotation 없이 이미지만 로드)
        """
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.samples = (
            []
        )  # (img_path, label_idx, class_name) 튜플 리스트 또는 (img_path,) 튜플
        self.class_to_idx = {}  # {class_name: idx}
        self.idx_to_class = {}  # {idx: class_name}

        if is_test:
            # 테스트 데이터셋: annotation 없이 이미지만 로드
            if not os.path.exists(img_dir):
                OpLog(
                    f"테스트 이미지 디렉토리를 찾을 수 없습니다: {img_dir}", bLines=True
                )
                return

            # 이미지 디렉토리의 모든 이미지 파일 수집
            for img_file in os.listdir(img_dir):
                if img_file.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
                ):
                    img_path = os.path.join(img_dir, img_file)
                    self.samples.append((img_path,))  # 테스트는 레이블 없음

            OpLog(f"테스트 이미지 {len(self.samples)}개 로드 완료", bLines=False)
        else:
            # 학습/검증 데이터셋: annotation 사용
            # get_class_mapping 함수 사용하여 클래스 매핑 정보 가져오기
            class_dirs, self.class_to_idx, self.idx_to_class, self._unique_classes = (
                get_class_mapping(annotations_dir)
            )

            # 각 클래스의 annotation 파일 읽기
            for category_id, json_paths in class_dirs:
                label_idx = self.class_to_idx[category_id]

                # 해당 category_id를 가진 모든 JSON 파일 읽기
                for json_path in json_paths:
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        # images 정보 추출
                        if "images" in data:
                            for img_info in data["images"]:
                                img_filename = img_info["file_name"]
                                img_path = os.path.join(self.img_dir, img_filename)

                                # 이미지 파일이 실제로 존재하는지 확인
                                if os.path.exists(img_path):
                                    self.samples.append(
                                        (img_path, label_idx, category_id)
                                    )
                    except Exception as e:
                        OpLog(f"Error reading {json_path}: {e}", bLines=False)

            OpLog(f"PillDataset 로드 완료: {len(self.samples)}개 샘플", bLines=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_test:
            # 테스트 데이터: 이미지만 반환 (레이블 없음)
            img_path = self.samples[idx][0]
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            # 이미지와 파일명 반환 (예측 후 결과 매칭용)
            return image, os.path.basename(img_path)
        else:
            # 학습/검증 데이터: 이미지와 레이블 반환
            img_path, label, category_id = self.samples[idx]  # category_id는 클래스 ID
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)
            return image, label


def GetDataset(annotations_dir, img_dir, transform_type="default", is_test=False):
    """
    데이터셋 생성

    Args:
        annotations_dir: annotation 디렉토리 경로 (is_test=True일 경우 무시됨)
        img_dir: 이미지 디렉토리 경로
        transform_type: 변환 타입 ('default', 'A', 'B')
        is_test: 테스트 데이터셋 여부
    """
    transform = GetTransform(transform_type)
    dataset = PillDataset(annotations_dir, img_dir, transform, is_test=is_test)
    return dataset


def GetLoaders(
    annotations_dir,
    transform_type,
    img_dir,
    batch_size=32,
    train_ratio=0.8,
    num_workers=4,
):
    """
    전체 데이터셋을 train/val로 분할하여 DataLoader 생성
    """
    from torch.utils.data import DataLoader, random_split

    # 전체 데이터셋 로드 (train용 augmentation)
    full_dataset = GetDataset(annotations_dir, img_dir, transform_type=transform_type)
    # Train/Val 분할
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    # Validation 데이터셋에는 augmentation 없이 기본 transform만 적용
    val_dataset_plain = GetDataset(
        annotations_dir, img_dir, transform_type=transform_type
    )
    val_indices = val_dataset.indices
    val_dataset = torch.utils.data.Subset(val_dataset_plain, val_indices)
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = GetTestLoader(batch_size=batch_size, num_workers=num_workers)
    OpLog(
        f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}",
        bLines=False,
    )
    OpLog(f"Total classes: {len(full_dataset.class_to_idx)}", bLines=False)
    return train_loader, val_loader, test_loader


def GetTestLoader(batch_size=16, num_workers=4):
    """
    테스트 데이터셋 로더 생성 (annotation 없음)

    Args:
        batch_size: 배치 크기
        num_workers: 워커 수

    Returns:
        test_loader: 테스트 데이터 로더
    """
    from torch.utils.data import DataLoader

    # 테스트 데이터셋 로드 (annotation 없음, 증강 없음)
    test_dataset = GetDataset(
        annotations_dir=None,  # 테스트는 annotation 불필요
        img_dir=TEST_IMG_DIR,
        transform_type="default",  # 증강 없음
        is_test=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 테스트는 shuffle 안함
        num_workers=num_workers,
    )

    OpLog(f"Test samples: {len(test_dataset)}", bLines=False)
    return test_loader


def TestLoader():
    # 데이터 디렉토리 존재 확인
    OpLog(f"ANNOTATION_DIR: {ANNOTATION_DIR}", bLines=False)
    OpLog(f"TRAIN_IMG_DIR: {TRAIN_IMG_DIR}", bLines=False)
    OpLog(f"TEST_IMG_DIR: {TEST_IMG_DIR}", bLines=False)
    
    if not os.path.exists(ANNOTATION_DIR):
        OpLog(f"ERROR: ANNOTATION_DIR이 존재하지 않습니다: {ANNOTATION_DIR}", bLines=True)
        return None, None, None
    
    if not os.path.exists(TRAIN_IMG_DIR):
        OpLog(f"ERROR: TRAIN_IMG_DIR이 존재하지 않습니다: {TRAIN_IMG_DIR}", bLines=True)
        return None, None, None
    
    # 어노테이션 파일 수 확인
    json_count = 0
    for root, dirs, files in os.walk(ANNOTATION_DIR):
        for f in files:
            if f.endswith('.json'):
                json_count += 1
    
    OpLog(f"발견된 JSON 파일 수: {json_count}", bLines=False)
    
    # 이미지 파일 수 확인
    img_count = 0
    if os.path.exists(TRAIN_IMG_DIR):
        for f in os.listdir(TRAIN_IMG_DIR):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_count += 1
    
    OpLog(f"발견된 이미지 파일 수: {img_count}", bLines=False)
    
    if json_count == 0:
        OpLog(f"ERROR: JSON 파일이 없습니다!", bLines=True)
        return None, None, None
    
    if img_count == 0:
        OpLog(f"ERROR: 이미지 파일이 없습니다!", bLines=True)
        return None, None, None
    
    train_loader, val_loader, test_loader = GetLoaders(
        ANNOTATION_DIR,
        "A",
        TRAIN_IMG_DIR,
        batch_size=16,
        train_ratio=0.8,
        num_workers=2,
    )
    return train_loader, val_loader, test_loader


# TestLoader()  # 주석 처리 - 자동 실행 방지


# ════════════════════════════════════════
# ▣ 04. 기본 모델 클래스 정의
# ════════════════════════════════════════


class BaseModel(nn.Module):
    """모델의 기본 클래스 - save/load 등 공통 기능 제공"""

    def __init__(self):
        super(BaseModel, self).__init__()
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.best_val_loss = float("inf")

    def getMyName(self):
        return self.__class__.__name__

    def getOptimizer(self, lr=0.001, gubun="freeze"):
        if gubun == "partial":
            params = [
                {
                    "params": self._model.backbone.parameters(),
                    "lr": self._lr * self._backbone_lr_ratio,
                },
                {"params": self._model.head.parameters(), "lr": self._lr},
            ]
        elif gubun == "freeze":
            for param in self._model.backbone.parameters():
                param.requires_grad = False
            params = self._model.head.parameters()
        else:
            params = self._model.parameters

        optimizer = torch.optim.SGD(
            params, lr=self._lr, momentum=0.9, weight_decay=5e-4
        )
        return optimizer

    def PreJob(self):
        """전처리 작업 (필요시 서브클래스에서 구현)"""
        pass

    ## 모델 저장 함수
    def save_model(self, epoch_index, is_best=False, **kwargs):
        """현재 모델 상태를 저장

        Args:
            epoch_index: 현재 에포크 번호
            is_best: Best 모델인지 여부
            **kwargs: 추가로 저장할 데이터 (model_state_dict, train_losses 등)
        """
        save_dir = MODEL_FILES
        makedirs(save_dir)
        model_name = self.getMyName()

        # Best 모델 파일명
        if is_best:
            filename = os.path.join(save_dir, f"{model_name}_best_model.pth")
        else:
            filename = os.path.join(save_dir, f"{model_name}_epoch_{epoch_index}.pth")

        # 기본 저장 데이터
        checkpoint = {
            "epoch": epoch_index,
            "is_best": is_best,
            "model_name": model_name,
        }

        # kwargs로 전달된 추가 데이터 저장
        checkpoint.update(kwargs)

        torch.save(checkpoint, filename)

        if is_best:
            print(f"  Best 모델 저장됨: {filename}")
            OpLog(f"Best model saved: {filename}")
        else:
            OpLog(f"모델 저장됨: {filename}", bLines=False)

    ## 모델 로드 함수
    def load_model(self, model_file, **kwargs):
        """저장된 모델 상태를 로드

        Args:
            model_file: 모델 파일 경로
            **kwargs: 로드 관련 추가 옵션

        Returns:
            dict: 체크포인트 데이터 또는 None
        """
        if not os.path.exists(model_file):
            OpLog(f"모델 파일을 찾을 수 없습니다: {model_file}", bLines=True)
            return None

        checkpoint = torch.load(model_file, map_location=DEVICE_TYPE)

        OpLog(
            f"모델 로드 완료: {model_file} (Epoch {checkpoint['epoch']})", bLines=True
        )
        return checkpoint

    ## 학습 이력 저장 함수 (객체 탐지 모델용)
    def save_metrics_to_csv(
        self,
        model_name,
        epoch_index,
        max_epochs,
        train_loss,
        current_lr,
        val_loss=None,
        test_loss=None,
        mAP50=None,
        mAP50_95=None,
        precision=None,
        recall=None,
        total_detections=None,
        avg_confidence=None,
        mode="train",
    ):
        """객체 탐지 모델 학습 메트릭을 CSV 파일에 저장
        Args:
            model_name: 모델 이름 (FasterRCNNModel, YOLOv8Model 등)
            epoch_index: 현재 에포크 (1-based)
            max_epochs: 최대 에포크
            train_loss: 학습 손실
            current_lr: 현재 학습률
            val_loss: 검증 손실 (optional)
            test_loss: 테스트 손실 (optional)
            mAP50: mAP@0.5 메트릭 (optional)
            mAP50_95: mAP@0.5:0.95 메트릭 (optional)
            precision: Precision 메트릭 (optional)
            recall: Recall 메트릭 (optional)
            total_detections: 총 탐지 개수 (optional)
            avg_confidence: 평균 신뢰도 (optional)
            mode: 'train', 'eval', 'test' 모드 표시
        """

        new_data = {
            "timestamp": [now_str()],
            "Model": [model_name],
            "Mode": [mode],
            "Max_Epochs": [max_epochs],
            "Epoch": [epoch_index],
            "Train_Loss": [round(train_loss, 6) if train_loss is not None else None],
            "Val_Loss": [round(val_loss, 6) if val_loss is not None else None],
            "Test_Loss": [round(test_loss, 6) if test_loss is not None else None],
            "mAP50": [round(mAP50, 4) if mAP50 is not None else None],
            "mAP50_95": [round(mAP50_95, 4) if mAP50_95 is not None else None],
            "Precision": [round(precision, 4) if precision is not None else None],
            "Recall": [round(recall, 4) if recall is not None else None],
            "Total_Detections": [total_detections if total_detections is not None else None],
            "Avg_Confidence": [round(avg_confidence, 4) if avg_confidence is not None else None],
            "Learning_Rate": [round(current_lr, 8) if current_lr is not None else None],
        }
        filename = RESULT_CSV
        lock_filename = filename + ".lock"
        new_df = pd.DataFrame(new_data)

        try:
            makedirs(os.path.dirname(filename))
            lock = FileLock(lock_filename, timeout=10)
            with lock:
                if os.path.exists(filename):
                    try:
                        existing_df = pd.read_csv(filename)
                        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                        updated_df.to_csv(filename, index=False)
                    except:
                        new_df.to_csv(filename, index=False)
                else:
                    new_df.to_csv(filename, index=False)
        except Exception as e:
            print(f"CSV 저장 중 오류 발생: {e}")
            OpLog(f"Error saving CSV: {e}")

    def _visualize_results(self, epoch, max_epochs, predictions, mode="eval"):
        """검증/테스트 결과 시각화 (서브클래스에서 구현)"""
        pass

    def fit(
        self,
        gubun="freeze",
        train_loader=None,
        val_loader=None,
        test_loader=None,
        epochs=50,
        imgsz=640,
        batch_size=16,
        lr=0.001,
    ):
        """모델 학습 - 서브클래스에서 구현 필요

        Args:
            gubun: 최적화 방식 ('freeze', 'partial', 'all') - FasterRCNN에서 사용
            train_loader: 학습 데이터로더
            val_loader: 검증 데이터로더
            test_loader: 테스트 데이터로더
            epochs: 학습 에포크 수
            imgsz: 입력 이미지 크기 - YOLOv8에서 사용
            batch_size: 배치 크기 - YOLOv8에서 사용
            lr: 학습률
        """
        raise NotImplementedError("fit must be implemented by subclass")

    def evalModel(self, val_loader, epoch, max_epochs):
        """검증 모드 - 서브클래스에서 구현 필요"""
        raise NotImplementedError("evalModel must be implemented by subclass")

    def testMode(self, test_loader, epoch, max_epochs):
        """테스트 모드 - 서브클래스에서 구현 필요"""
        raise NotImplementedError("testMode must be implemented by subclass")


# ════════════════════════════════════════
# ▣ 05. YOLOv8 모델 정의
# ════════════════════════════════════════


class YOLOv8Model(BaseModel):
    """
    YOLOv8 기반 객체 탐지 모델
    - Ultralytics YOLOv8 사용
    - 객체 탐지 및 분류 동시 수행
    """

    def __init__(self, model_size="n", num_classes=None):
        """
        Args:
            model_size: YOLOv8 모델 크기 ('n', 's', 'm', 'l', 'x')
            num_classes: 클래스 수 (None이면 자동 계산)
        """
        super(YOLOv8Model, self).__init__()
        self.model_size = model_size
        self.num_classes = num_classes if num_classes else count_classes(ANNOTATION_DIR)

        # YOLOv8 모델 초기화 (사전 학습된 가중치 사용)
        model_path = f"yolov8{model_size}.pt"
        
        # 모델 파일이 손상된 경우 삭제하고 재다운로드
        if os.path.exists(model_path):
            try:
                # 파일 검증 시도
                test_model = YOLO(model_path)
                self.model = test_model
                OpLog(f"YOLOv8{model_size} 모델 로드 완료", bLines=False)
            except Exception as e:
                OpLog(f"YOLOv8 모델 파일 손상 감지, 재다운로드 중...: {e}", bLines=True)
                try:
                    os.remove(model_path)
                    OpLog(f"손상된 모델 파일 삭제: {model_path}", bLines=False)
                except:
                    pass
                self.model = YOLO(model_path)  # 자동 재다운로드
        else:
            OpLog(f"YOLOv8{model_size} 모델 다운로드 중...", bLines=True)
            self.model = YOLO(model_path)  # 자동 다운로드
            
        self.optimizer = None
        self.lr_scheduler = None

    def getMyName(self):
        return f"YOLOv8Model_{self.model_size}"

    def getOptimizer(self, lr=0.001, gubun="freeze"):
        """
        YOLOv8 모델의 optimizer를 반환
        YOLOv8는 내부적으로 optimizer를 관리하므로 이 메서드는 인터페이스 통일을 위해 제공됨

        Args:
            lr: 학습률
            gubun: 최적화 방식 ('freeze', 'partial', 'all')

        Returns:
            optimizer: YOLOv8는 내부 optimizer를 사용하므로 None 반환
        """
        OpLog(
            f"YOLOv8는 내부 optimizer를 사용합니다. (lr={lr}, mode={gubun})",
            bLines=False,
        )
        # YOLOv8는 ultralytics 내부에서 optimizer를 자동 관리
        return None

    def preJob(self):
        """전처리 작업: YOLO YAML 파일, 클래스 매핑, YOLO 형식 레이블 생성"""
        import yaml

        class_mapping_file = os.path.join(BASE_DIR, "oraldrug", "class_mapping.json")
        labels_dir = os.path.join(BASE_DIR, "oraldrug", "labels")

        # 기존 YAML 파일과 labels 삭제 (새로 생성하기 위해)
        if os.path.exists(YAML_FILE):
            try:
                os.remove(YAML_FILE)
                OpLog(f"기존 YAML 파일 삭제: {YAML_FILE}", bLines=False)
            except Exception as e:
                OpLog(f"YAML 파일 삭제 실패: {e}", bLines=False)
        
        if os.path.exists(labels_dir):
            try:
                import shutil
                shutil.rmtree(labels_dir)
                OpLog(f"기존 labels 디렉토리 삭제: {labels_dir}", bLines=False)
            except Exception as e:
                OpLog(f"labels 디렉토리 삭제 실패: {e}", bLines=False)

        OpLog("YOLO 데이터셋 준비 시작", bLines=True)

        # get_class_mapping 함수 사용하여 클래스 정보 가져오기
        class_dirs, class_to_idx, idx_to_class, class_names = get_class_mapping(
            ANNOTATION_DIR
        )

        # 클래스 매핑 정보 저장 (category_id: index)
        class_mapping = {}
        for category_id in class_names:
            class_mapping[str(category_id)] = {"index": class_to_idx[category_id]}

        # 클래스 매핑 JSON 파일 저장
        with open(class_mapping_file, "w", encoding="utf-8") as f:
            json.dump(class_mapping, f, ensure_ascii=False, indent=2)

        # JSON annotation을 YOLO 형식(.txt)으로 변환
        OpLog("JSON annotation을 YOLO 형식으로 변환 중...", bLines=False)
        makedirs(labels_dir)

        converted_count = 0
        for category_id, json_paths in class_dirs:
            class_id = class_to_idx[category_id]

            for json_path in json_paths:
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # 각 이미지에 대한 annotation 처리
                    if "images" in data and "annotations" in data:
                            for img_info in data["images"]:
                                img_filename = img_info["file_name"]
                                img_id = img_info["id"]
                                img_width = img_info.get("width", 640)
                                img_height = img_info.get("height", 640)

                                # 해당 이미지의 annotation 찾기
                                img_annotations = [
                                    ann
                                    for ann in data["annotations"]
                                    if ann.get("image_id") == img_id
                                ]

                                if img_annotations:
                                    # YOLO 형식 레이블 파일 생성 (이미지 파일명과 동일, 확장자만 .txt)
                                    label_filename = (
                                        os.path.splitext(img_filename)[0] + ".txt"
                                    )
                                    label_path = os.path.join(
                                        labels_dir, label_filename
                                    )

                                    # 이미 파일이 존재하면 건너뛰기
                                    if os.path.exists(label_path):
                                        converted_count += 1
                                        continue

                                    with open(label_path, "w", encoding="utf-8") as lf:
                                        for ann in img_annotations:
                                            bbox = ann.get("bbox", [])
                                            if len(bbox) == 4:
                                                # bbox: [x, y, width, height] (픽셀 단위)
                                                x, y, w, h = bbox

                                                # YOLO 형식으로 변환: [x_center, y_center, width, height] (0~1 정규화)
                                                x_center = (x + w / 2) / img_width
                                                y_center = (y + h / 2) / img_height
                                                norm_width = w / img_width
                                                norm_height = h / img_height

                                                # YOLO 형식: <class_id> <x_center> <y_center> <width> <height>
                                                # category_id를 YOLO class index(0-based)로 변환
                                                ann_category_id = ann.get("category_id", class_id)
                                                try:
                                                    yolo_class_idx = class_names.index(ann_category_id)
                                                except ValueError:
                                                    OpLog(f"Warning: category_id {ann_category_id} not in class_names, using class_id {class_id}", bLines=False)
                                                    yolo_class_idx = class_id
                                                
                                                lf.write(
                                                    f"{yolo_class_idx} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n"
                                                )

                                    converted_count += 1
                except Exception as e:
                    OpLog(f"Error converting {json_path}: {e}", bLines=False)

        OpLog(f"YOLO 레이블 변환 완료: {converted_count}개 파일", bLines=False)

        # YAML 데이터 구조 생성 (모든 경로를 절대 경로로 사용)
        # names를 리스트로 설정 (YOLO는 0-based index 사용)
        # class 0 = category_id class_names[0], class 1 = category_id class_names[1], ...
        
        yaml_data = {
            "path": os.path.abspath(BASE_DIR).replace("\\", "/"),
            "train": os.path.abspath(TRAIN_IMG_DIR).replace("\\", "/"),
            "val": os.path.abspath(TRAIN_IMG_DIR).replace("\\", "/"),
            "test": os.path.abspath(TEST_IMG_DIR).replace("\\", "/"),
            "nc": len(class_names),
            "names": [str(cat_id) for cat_id in class_names],  # 리스트 형태: ['1899', '2482', '3350', ...]
        }

        # YAML 파일 저장
        makedirs(os.path.dirname(YAML_FILE))
        with open(YAML_FILE, "w", encoding="utf-8") as f:
            yaml.dump(
                yaml_data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        OpLog(f"YAML 파일 생성 완료: {YAML_FILE}", bLines=False)
        OpLog(f"  - train: {yaml_data['train']}", bLines=False)
        OpLog(f"  - val: {yaml_data['val']}", bLines=False)
        OpLog(f"  - test: {yaml_data['test']}", bLines=False)
        OpLog(f"클래스 매핑 파일 생성 완료: {class_mapping_file}", bLines=False)
        OpLog(f"총 클래스 수: {len(class_names)}", bLines=False)

    def fit(
        self,
        gubun="freeze",
        train_loader=None,
        val_loader=None,
        test_loader=None,
        epochs=50,
        imgsz=640,
        batch_size=16,
        lr=0.001,
    ):
        """
        YOLOv8 모델 학습 (BaseModel 인터페이스 준수)

        Args:
            gubun: 최적화 방식 (YOLOv8는 사용하지 않음, 인터페이스 통일용)
            train_loader: 학습 데이터로더 (YOLOv8는 내부적으로 사용하지 않지만 인터페이스 통일)
            val_loader: 검증 데이터로더
            test_loader: 테스트 데이터로더
            epochs: 학습 에포크 수
            imgsz: 입력 이미지 크기
            batch_size: 배치 크기
            lr: 학습률 (YOLOv8는 내부적으로 관리)
        """
        OpLog(f"YOLOv8{self.model_size} 모델 학습 시작", bLines=True)

        # YAML 파일과 labels를 항상 새로 생성
        OpLog("YAML 파일 및 레이블 재생성 중...", bLines=True)
        self.preJob()

        # YOLOv8 학습 시작 (YOLOv8는 내부적으로 전체 에포크를 학습하고 검증까지 자동 수행)
        results = self.model.train(
            data=YAML_FILE,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=DEVICE_TYPE,
            project=os.path.join(BASE_DIR, "yolo_results"),
            name=f"yolov8{self.model_size}_train",
            exist_ok=True,
            patience=10,  # Early stopping
            save=True,
            plots=True,
        )

        OpLog(f"YOLOv8 학습 완료!", bLines=True)

        # 학습 완료 후 최종 검증 및 테스트 (1회만 수행)
        if val_loader is not None:
            OpLog("최종 검증 수행 중...", bLines=True)
            self.evalModel(val_loader, epochs, epochs)

        if test_loader is not None:
            OpLog("최종 테스트 수행 중...", bLines=True)
            self.testMode(test_loader, epochs, epochs)

        # 학습 결과 시각화
        self.plot_results()

        return results

    def evalModel(self, val_loader, epoch, max_epochs):
        """
        검증 데이터셋에 대한 모델 평가 (BaseModel 인터페이스 구현)

        Args:
            val_loader: 검증 데이터로더
            epoch: 현재 에포크 번호
            max_epochs: 전체 에포크 수
        """
        OpLog(f"[Epoch {epoch}/{max_epochs}] Validation 시작", bLines=True)

        # 모델 검증
        metrics = self.model.val(
            data=YAML_FILE,
            device=DEVICE_TYPE,
            split="val",
            plots=False,  # 매 에포크마다 플롯 생성 방지
        )

        # 주요 메트릭 추출
        mAP50 = float(metrics.box.map50)
        mAP50_95 = float(metrics.box.map)
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        val_loss = 1.0 - mAP50_95  # mAP를 손실로 변환 (높을수록 좋으므로 1에서 빼기)

        # 검증 손실 저장
        self.val_losses.append(val_loss)

        OpLog(
            f"mAP50: {mAP50:.4f}, mAP50-95: {mAP50_95:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}",
            bLines=False,
        )

        # CSV에 평가 메트릭 저장
        self.save_metrics_to_csv(
            model_name=self.getMyName(),
            epoch_index=epoch,
            max_epochs=max_epochs,
            train_loss=0.0,  # YOLOv8는 내부적으로 train loss 관리
            current_lr=0.0,
            mode="eval",
            mAP50=mAP50,
            mAP50_95=mAP50_95,
            precision=precision,
            recall=recall,
        )

        # 시각화 (간단한 예측 결과만)
        predictions = {
            "mAP50": mAP50,
            "mAP50_95": mAP50_95,
            "precision": precision,
            "recall": recall,
        }
        self._visualize_results(epoch, max_epochs, predictions, mode="eval")

        return val_loss

    def testMode(self, test_loader, epoch, max_epochs):
        """
        테스트 데이터셋에 대한 모델 평가 (BaseModel 인터페이스 구현)

        Args:
            test_loader: 테스트 데이터로더
            epoch: 현재 에포크 번호
            max_epochs: 전체 에포크 수
        """
        OpLog(f"[Epoch {epoch}/{max_epochs}] Test 시작", bLines=True)

        # 테스트 이미지에 대한 예측 수행
        results = self.model.predict(
            source=TEST_IMG_DIR,
            conf=0.25,
            save=False,
            device=DEVICE_TYPE,
            verbose=False,
        )

        # 예측 결과 수집
        predictions = []
        for result in results:
            pred_dict = {
                "boxes": result.boxes.xyxy.cpu().numpy() if result.boxes else [],
                "scores": result.boxes.conf.cpu().numpy() if result.boxes else [],
                "labels": result.boxes.cls.cpu().numpy() if result.boxes else [],
                "filename": os.path.basename(result.path),
            }
            predictions.append(pred_dict)

        OpLog(f"테스트 이미지 {len(predictions)}개 예측 완료", bLines=False)

        # 간단한 통계
        total_detections = sum(len(p["boxes"]) for p in predictions)
        avg_conf = sum(
            p["scores"].mean() if len(p["scores"]) > 0 else 0.0 for p in predictions
        ) / max(len(predictions), 1)
        test_loss = 1.0 - avg_conf  # 평균 신뢰도를 손실로 변환

        # CSV에 테스트 메트릭 저장
        self.save_metrics_to_csv(
            model_name=self.getMyName(),
            epoch_index=epoch,
            max_epochs=max_epochs,
            train_loss=0.0,
            current_lr=0.0,
            mode="test",
            test_loss=test_loss,
            total_detections=total_detections,
            avg_confidence=avg_conf,
        )

        # 시각화
        self._visualize_results(epoch, max_epochs, predictions, mode="test")

        return test_loss

    def _visualize_results(self, epoch, max_epochs, predictions, mode="eval"):
        """
        검증/테스트 결과 시각화 (BaseModel 인터페이스 구현)

        Args:
            epoch: 현재 에포크 번호
            max_epochs: 전체 에포크 수
            predictions: 예측 결과 (dict 또는 list)
            mode: 'eval' 또는 'test'
        """
        results_dir = os.path.join(BASE_DIR, "oraldrug", "results", self.getMyName())
        makedirs(results_dir)

        if mode == "eval":
            # 검증 모드: 메트릭 표시
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            metrics_text = f"Epoch {epoch}/{max_epochs} - Validation Metrics\n\n"
            metrics_text += f"mAP50: {predictions.get('mAP50', 0):.4f}\n"
            metrics_text += f"mAP50-95: {predictions.get('mAP50_95', 0):.4f}\n"
            metrics_text += f"Precision: {predictions.get('precision', 0):.4f}\n"
            metrics_text += f"Recall: {predictions.get('recall', 0):.4f}"

            ax.text(
                0.5,
                0.5,
                metrics_text,
                ha="center",
                va="center",
                fontsize=14,
                family="monospace",
            )
            ax.axis("off")

            save_path = os.path.join(results_dir, f"eval_epoch{epoch:03d}.png")
            plt.savefig(save_path, bbox_inches="tight", dpi=100)
            plt.close()

            OpLog(f"검증 결과 저장: {save_path}", bLines=False)

        elif mode == "test":
            # 테스트 모드: 예측 결과 샘플 시각화 (최대 6개)
            num_samples = min(6, len(predictions))
            if num_samples == 0:
                return

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for i in range(num_samples):
                pred = predictions[i]
                img_path = os.path.join(TEST_IMG_DIR, pred["filename"])

                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")
                    axes[i].imshow(img)
                    axes[i].set_title(
                        f"{pred['filename']}\nDetections: {len(pred['boxes'])}"
                    )
                    axes[i].axis("off")
                else:
                    axes[i].text(
                        0.5,
                        0.5,
                        f"Image not found:\n{pred['filename']}",
                        ha="center",
                        va="center",
                    )
                    axes[i].axis("off")

            # 빈 서브플롯 숨기기
            for i in range(num_samples, 6):
                axes[i].axis("off")

            plt.suptitle(
                f"Epoch {epoch}/{max_epochs} - Test Predictions (Sample)", fontsize=16
            )
            plt.tight_layout()

            save_path = os.path.join(results_dir, f"test_epoch{epoch:03d}.png")
            plt.savefig(save_path, bbox_inches="tight", dpi=100)
            plt.close()

            OpLog(f"테스트 결과 저장: {save_path}", bLines=False)

    def predict(self, source, conf=0.25, save=True):
        """
        이미지에 대한 예측 수행

        Args:
            source: 이미지 경로, 폴더 경로, 또는 이미지 URL
            conf: 신뢰도 임계값
            save: 결과 저장 여부
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            save=save,
            project=os.path.join(BASE_DIR, "yolo_results"),
            name=f"yolov8{self.model_size}_predict",
            exist_ok=True,
        )

        return results

    def load_yolo_model(self, model_path):
        """YOLOv8 모델 로드"""
        if not os.path.exists(model_path):
            OpLog(f"모델 파일을 찾을 수 없습니다: {model_path}", bLines=True)
            return False

        self.model = YOLO(model_path)
        OpLog(f"YOLOv8 모델 로드 완료: {model_path}", bLines=True)
        return True

    def save_yolo_model(self, save_path=None):
        """YOLOv8 모델 저장"""
        if save_path is None:
            save_path = os.path.join(MODEL_FILES, f"yolov8{self.model_size}_final.pt")

        makedirs(os.path.dirname(save_path))

        # YOLOv8 모델 내보내기
        self.model.export(format="torchscript", dynamic=False)

        OpLog(f"YOLOv8 모델 저장됨: {save_path}", bLines=True)
        return save_path

    def plot_results(self):
        """학습 결과 시각화"""
        results_dir = os.path.join(
            BASE_DIR, "yolo_results", f"yolov8{self.model_size}_train"
        )
        results_file = os.path.join(results_dir, "results.png")

        if os.path.exists(results_file):
            OpLog(f"학습 결과 그래프: {results_file}", bLines=False)
        else:
            OpLog("학습 결과 파일을 찾을 수 없습니다.", bLines=False)
        plt.close()


# ════════════════════════════════════════
# ▣ 06. Faster R-CNN 모델 정의
# ════════════════════════════════════════
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNNModel(BaseModel):
    """
    Faster R-CNN 기반 객체 탐지 모델
    - torchvision의 사전 학습된 Faster R-CNN 사용
    - ResNet50-FPN 백본 활용
    - 객체 탐지 및 분류 동시 수행
    """

    def __init__(self, num_classes=None, backbone="resnet50"):
        """
        Args:
            num_classes: 클래스 수 (None이면 자동 계산, +1은 배경 클래스)
            backbone: 백본 네트워크 ('resnet50', 'mobilenet' 등)
        """
        super(FasterRCNNModel, self).__init__()
        self.backbone = backbone
        self.num_classes = (
            num_classes if num_classes else count_classes(ANNOTATION_DIR)
        ) + 1  # +1 for background

        # 사전 훈련된 Faster R-CNN 모델 로드
        if backbone == "resnet50":
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=True
            )
        elif backbone == "mobilenet":
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                pretrained=True
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 분류 헤드를 클래스 수에 맞게 변경
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, self.num_classes
            )
        )

        # 학습 이력 저장용
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")

        OpLog(
            f"Faster R-CNN 모델 초기화 완료 (backbone: {backbone}, classes: {self.num_classes})",
            bLines=False,
        )

    def getMyName(self):
        return f"FasterRCNNModel_{self.backbone}"

    def getOptimizer(self, lr=0.001, gubun="freeze"):
        """
        Faster R-CNN 모델의 optimizer를 반환

        Args:
            lr: 학습률
            gubun: 최적화 방식
                - 'freeze': backbone을 고정하고 head만 학습
                - 'partial': backbone은 낮은 lr, head는 일반 lr로 학습
                - 'all': 전체 모델 학습

        Returns:
            optimizer: torch.optim.SGD optimizer
        """
        if gubun == "partial":
            # Backbone과 head를 다른 학습률로 설정
            params = [
                {
                    "params": self.model.backbone.parameters(),
                    "lr": lr * 0.1,
                },  # backbone은 10% lr
                {
                    "params": self.model.roi_heads.parameters(),
                    "lr": lr,
                },  # head는 100% lr
            ]
            OpLog(
                f"Optimizer 생성: partial mode (backbone_lr={lr*0.1:.6f}, head_lr={lr:.6f})",
                bLines=False,
            )
        elif gubun == "freeze":
            # Backbone을 고정하고 head만 학습
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            params = self.model.roi_heads.parameters()
            OpLog(f"Optimizer 생성: freeze mode (only head, lr={lr:.6f})", bLines=False)
        else:  # 'all' or default
            # 전체 모델 학습
            params = self.model.parameters()
            OpLog(f"Optimizer 생성: all mode (lr={lr:.6f})", bLines=False)

        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
        return optimizer

    def prepare_targets(self, annotations_list):
        """
        JSON 어노테이션을 Faster R-CNN 형식의 타겟으로 변환

        Args:
            annotations_list: 어노테이션 딕셔너리 리스트

        Returns:
            list of dict: [{'boxes': tensor, 'labels': tensor}, ...]
        """
        targets = []
        for ann in annotations_list:
            boxes = []
            labels = []

            if "annotations" in ann:
                for obj in ann["annotations"]:
                    bbox = obj.get("bbox", [])
                    if len(bbox) == 4:
                        # bbox: [x, y, width, height] -> [x1, y1, x2, y2]
                        x, y, w, h = bbox
                        boxes.append([x, y, x + w, y + h])

                        # 카테고리 ID를 레이블로 사용
                        label = obj.get("category_id", 0)
                        labels.append(label)

            if len(boxes) > 0:
                target = {
                    "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                    "labels": torch.as_tensor(labels, dtype=torch.int64),
                }
            else:
                # 빈 어노테이션 처리
                target = {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                }

            targets.append(target)

        return targets

    def fit(
        self,
        gubun="freeze",
        train_loader=None,
        val_loader=None,
        test_loader=None,
        epochs=50,
        imgsz=640,
        batch_size=16,
        lr=0.005,
    ):
        """
        Faster R-CNN 모델 학습 (BaseModel 인터페이스 준수)

        Args:
            gubun: 최적화 방식 ('freeze', 'partial', 'all')
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더 (선택)
            test_loader: 테스트 데이터 로더 (선택)
            epochs: 학습 에포크 수
            imgsz: 입력 이미지 크기 (FasterRCNN은 사용하지 않음, 인터페이스 통일용)
            batch_size: 배치 크기 (FasterRCNN은 사용하지 않음, 인터페이스 통일용)
            lr: 학습률
        """
        OpLog(f"Faster R-CNN 모델 학습 시작 (Epochs: {epochs}, LR: {lr})", bLines=True)

        self.model.to(DEVICE_TYPE)

        # 옵티마이저 및 스케줄러 설정
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = self.getOptimizer(lr=lr, gubun="partial")
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=3, gamma=0.1
        )

        # 학습 루프
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0

            for batch_idx, (images, targets) in enumerate(train_loader):
                # 이미지를 디바이스로 이동
                images = [img.to(DEVICE_TYPE) for img in images]

                # 타겟 준비
                if isinstance(targets, torch.Tensor):
                    targets = [
                        {
                            "boxes": torch.tensor(
                                [[0, 0, 224, 224]], dtype=torch.float32
                            ).to(DEVICE_TYPE),
                            "labels": torch.tensor(
                                [label.item()], dtype=torch.int64
                            ).to(DEVICE_TYPE),
                        }
                        for label in targets
                    ]
                else:
                    targets = [
                        {k: v.to(DEVICE_TYPE) for k, v in t.items()} for t in targets
                    ]

                # Forward pass
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                # Backward pass
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                epoch_loss += losses.item()
                batch_count += 1

                if batch_idx % 10 == 0:
                    OpLog(
                        f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {losses.item():.4f}",
                        bLines=False,
                    )

            # 에포크 평균 손실
            avg_train_loss = epoch_loss / batch_count
            self.train_losses.append(avg_train_loss)

            # 현재 학습률 가져오기
            current_lr = self.optimizer.param_groups[0]["lr"]

            # 매 epoch 검증
            if val_loader is not None:
                self.evalModel(val_loader, DEVICE_TYPE, epoch + 1, epochs)

                # Best 모델 저장
                if self.val_losses[-1] < self.best_val_loss:
                    self.best_val_loss = self.val_losses[-1]
                    self.save_model(
                        epoch_index=epoch + 1,
                        is_best=True,
                        model_state_dict=self.model.state_dict(),
                        optimizer_state_dict=self.optimizer.state_dict(),
                        train_loss=avg_train_loss,
                        val_loss=self.val_losses[-1],
                    )

            # 10 epoch마다 테스트
            if test_loader is not None and (epoch + 1) % 10 == 0:
                self.testMode(test_loader, DEVICE_TYPE, epoch + 1, epochs)

            # 매 epoch마다 모델 저장
            self.save_model(
                epoch_index=epoch + 1,
                is_best=False,
                model_state_dict=self.model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                train_loss=avg_train_loss,
                val_loss=self.val_losses[-1] if val_loader else None,
                test_loss=(
                    self.test_losses[-1]
                    if (test_loader and len(self.test_losses) > 0)
                    else None
                ),
            )

            # 학습률 스케줄러 업데이트
            self.lr_scheduler.step()

        OpLog("Faster R-CNN 학습 완료!", bLines=True)
        self.plot_results()

    def evalModel(self, val_loader, epoch, max_epochs):
        """검증 모드"""
        self.model.train()  # Faster R-CNN은 train 모드에서 loss 반환
        val_loss = 0.0
        batch_count = 0
        predictions_all = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(DEVICE_TYPE) for img in images]

                # 타겟 준비
                if isinstance(targets, torch.Tensor):
                    targets = [
                        {
                            "boxes": torch.tensor(
                                [[0, 0, 224, 224]], dtype=torch.float32
                            ).to(DEVICE_TYPE),
                            "labels": torch.tensor(
                                [label.item()], dtype=torch.int64
                            ).to(DEVICE_TYPE),
                        }
                        for label in targets
                    ]
                else:
                    targets = [
                        {k: v.to(DEVICE_TYPE) for k, v in t.items()} for t in targets
                    ]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                batch_count += 1

                # 예측 수집 (시각화용)
                self.model.eval()
                preds = self.model(images)
                predictions_all.extend(preds)
                self.model.train()

        avg_val_loss = val_loss / batch_count if batch_count > 0 else 0.0
        self.val_losses.append(avg_val_loss)

        OpLog(
            f"Epoch [{epoch}/{max_epochs}] - Val Loss: {avg_val_loss:.4f}", bLines=True
        )

        # 메트릭 CSV 저장
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.save_metrics_to_csv(
            model_name=self.getMyName(),
            epoch_index=epoch,
            max_epochs=max_epochs,
            train_loss=self.train_losses[-1],
            val_loss=avg_val_loss,
            current_lr=current_lr,
            mode="eval",
        )

        # 시각화
        self._visualize_results(epoch, max_epochs, predictions_all[:5], mode="eval")

    def testMode(self, test_loader, epoch, max_epochs):
        """테스트 모드"""
        self.model.eval()
        predictions_all = []
        filenames_all = []

        with torch.no_grad():
            for images, filenames in test_loader:
                images = [img.to(DEVICE_TYPE) for img in images]

                # 예측
                preds = self.model(images)
                predictions_all.extend(preds)
                filenames_all.extend(filenames)

        # 테스트 손실은 계산 불가 (레이블 없음)
        OpLog(
            f"Epoch [{epoch}/{max_epochs}] - Test predictions: {len(predictions_all)}",
            bLines=True,
        )

        # 메트릭 CSV 저장
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.save_metrics_to_csv(
            model_name=self.getMyName(),
            epoch_index=epoch,
            max_epochs=max_epochs,
            train_loss=self.train_losses[-1],
            current_lr=current_lr,
            mode="test",
        )

        # 시각화
        self._visualize_results(epoch, max_epochs, predictions_all[:5], mode="test")

    def validate(self, val_loader):
        """검증 데이터셋에 대한 손실 계산 (하위 호환성 유지)"""
        self.model.train()  # Faster R-CNN은 train 모드에서 loss 반환
        val_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(DEVICE_TYPE) for img in images]

                # 타겟 준비
                if isinstance(targets, torch.Tensor):
                    targets = [
                        {
                            "boxes": torch.tensor(
                                [[0, 0, 224, 224]], dtype=torch.float32
                            ).to(DEVICE_TYPE),
                            "labels": torch.tensor(
                                [label.item()], dtype=torch.int64
                            ).to(DEVICE_TYPE),
                        }
                        for label in targets
                    ]
                else:
                    targets = [
                        {k: v.to(DEVICE_TYPE) for k, v in t.items()} for t in targets
                    ]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                batch_count += 1

        return val_loss / batch_count if batch_count > 0 else 0.0

    def predict(self, images, conf_threshold=0.5, save_to_csv=False):
        """
        이미지에 대한 예측 수행

        Args:
            images: 이미지 텐서 리스트 또는 단일 이미지
            conf_threshold: 신뢰도 임계값
            save_to_csv: CSV에 예측 통계 저장 여부

        Returns:
            list of dict: [{'boxes': tensor, 'labels': tensor, 'scores': tensor}, ...]
        """
        self.model.eval()
        self.model.to(DEVICE_TYPE)

        if not isinstance(images, list):
            images = [images]

        images = [img.to(DEVICE_TYPE) for img in images]

        with torch.no_grad():
            predictions = self.model(images)

        # 신뢰도 필터링
        filtered_predictions = []
        total_detections = 0
        avg_confidence = 0.0

        for pred in predictions:
            mask = pred["scores"] >= conf_threshold
            filtered_pred = {
                "boxes": pred["boxes"][mask],
                "labels": pred["labels"][mask],
                "scores": pred["scores"][mask],
            }
            filtered_predictions.append(filtered_pred)

            # 통계 수집
            total_detections += mask.sum().item()
            if mask.sum() > 0:
                avg_confidence += pred["scores"][mask].mean().item()

        if len(filtered_predictions) > 0 and total_detections > 0:
            avg_confidence /= len(
                [p for p in filtered_predictions if len(p["scores"]) > 0]
            )

        # CSV에 예측 통계 저장
        if save_to_csv:
            self.save_metrics_to_csv(
                model_name=f"{self.__class__.__name__}_Prediction",
                epoch_index=total_detections,  # 탐지 개수를 epoch 필드에 저장
                max_epochs=len(images),  # 전체 이미지 수
                train_loss=avg_confidence,  # 평균 신뢰도를 train_loss 필드에 저장
                current_lr=conf_threshold,
            )
            OpLog(
                f"예측 통계 - 총 탐지: {total_detections}, 평균 신뢰도: {avg_confidence:.4f}",
                bLines=False,
            )

        return filtered_predictions

    def _visualize_results(self, epoch, max_epochs, predictions, mode="eval"):
        """검증/테스트 결과 시각화"""
        if len(predictions) == 0:
            return

        # 시각화할 샘플 수 (최대 5개)
        num_samples = min(5, len(predictions))

        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 4, 4))
        if num_samples == 1:
            axes = [axes]

        for idx, pred in enumerate(predictions[:num_samples]):
            ax = axes[idx]

            # 예측 결과 표시
            num_detections = len(pred["boxes"])
            avg_score = pred["scores"].mean().item() if len(pred["scores"]) > 0 else 0

            ax.text(
                0.5,
                0.5,
                f"Detections: {num_detections}\nAvg Score: {avg_score:.3f}",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title(f"Sample {idx+1}")
            ax.axis("off")

        plt.tight_layout()

        # 저장
        result_dir = os.path.join(BASE_DIR, "oraldrug", "results", self.getMyName())
        makedirs(result_dir)
        filename = os.path.join(result_dir, f"{mode}_epoch{epoch}_of_{max_epochs}.png")
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        OpLog(f"{mode.capitalize()} 결과 시각화 저장: {filename}", bLines=False)
        plt.close()

    def plot_results(self):
        """학습 결과 시각화"""
        if len(self.train_losses) == 0:
            OpLog("학습 이력이 없습니다.", bLines=False)
            return

        plt.figure(figsize=(12, 5))

        # 손실 그래프
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Train Loss", marker="o")
        if len(self.val_losses) > 0:
            plt.plot(self.val_losses, label="Val Loss", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Faster R-CNN Training/Validation Loss")
        plt.legend()
        plt.grid(True)

        # 손실 통계
        plt.subplot(1, 2, 2)
        stats_text = f"Training Statistics\n\n"
        stats_text += f"Epochs: {len(self.train_losses)}\n"
        stats_text += f"Final Train Loss: {self.train_losses[-1]:.4f}\n"
        if len(self.val_losses) > 0:
            stats_text += f"Final Val Loss: {self.val_losses[-1]:.4f}\n"
            stats_text += f"Best Val Loss: {self.best_val_loss:.4f}\n"
        plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment="center")
        plt.axis("off")

        plt.tight_layout()

        # 그래프 저장
        save_dir = os.path.join(BASE_DIR, "oraldrug", "results")
        makedirs(save_dir)
        save_path = os.path.join(save_dir, "fasterrcnn_training_results.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        OpLog(f"학습 결과 그래프 저장: {save_path}", bLines=False)
        plt.close()


# ════════════════════════════════════════
# ▣ 07. 모델 생성 및 학습 실행
# ════════════════════════════════════════
def LoadModel(model_type, file_path):
    """
    체크포인트 파일에서 모델을 로드
    
    Args:
        model_type: 모델 타입 ('yolov8s', 'yolov8n', 'fasterrcnn_resnet50' 등)
        file_path: 체크포인트 파일 경로
    
    Returns:
        model: 로드된 모델 객체
    """
    OpLog(f"LoadModel 시작: model_type={model_type}, file_path={file_path}", bLines=True)
    
    # 파일 존재 확인
    if not os.path.exists(file_path):
        OpLog(f"모델 파일을 찾을 수 없습니다: {file_path}", bLines=True)
        return None
    
    # 모델 타입별 로드
    if model_type.startswith("yolov8"):
        # YOLOv8 모델 생성
        model_size = model_type.replace("yolov8", "")
        model = YOLOv8Model(model_size=model_size)
        
        # YOLOv8 모델 로드
        success = model.load_yolo_model(file_path)
        if not success:
            OpLog(f"YOLOv8 모델 로드 실패: {file_path}", bLines=True)
            return None
            
    elif model_type.startswith("fasterrcnn"):
        # Faster R-CNN 모델 생성
        backbone = model_type.replace("fasterrcnn_", "")
        model = FasterRCNNModel(backbone=backbone)
        
        # 체크포인트 로드
        checkpoint = model.load_model(file_path)
        if checkpoint is None:
            OpLog(f"Faster R-CNN 모델 로드 실패: {file_path}", bLines=True)
            return None
        
        # 모델 state_dict 로드
        if 'model_state_dict' in checkpoint:
            model.model.load_state_dict(checkpoint['model_state_dict'])
            OpLog(f"Faster R-CNN model_state_dict 로드 완료", bLines=False)
        else:
            OpLog(f"체크포인트에 model_state_dict가 없습니다", bLines=True)
            return None
            
    else:
        OpLog(f"지원되지 않는 모델 타입: {model_type}", bLines=True)
        return None
    
    OpLog(f"LoadModel 완료: {model_type}", bLines=True)
    return model


def TestModel(model_type, model_file, test_dir=None, batch_size=16):
    """
    저장된 모델을 로드하여 테스트 수행
    
    Args:
        model_type: 모델 타입 ('yolov8n', 'yolov8s', 'fasterrcnn_resnet50' 등)
        model_file: 모델 체크포인트 파일 경로
        test_dir: 테스트 이미지 디렉토리 (None이면 TEST_IMG_DIR 사용)
        batch_size: 배치 크기
    
    Returns:
        test_loss: 테스트 손실값
    """
    OpLog(f"TestModel 시작: model_type={model_type}, model_file={model_file}", bLines=True)
    
    # 모델 로드
    model = LoadModel(model_type, model_file)
    if model is None:
        OpLog("모델 로드 실패, 테스트 중단", bLines=True)
        return None
    
    # 테스트 디렉토리 설정
    if test_dir is None:
        test_dir = TEST_IMG_DIR
    
    OpLog(f"테스트 디렉토리: {test_dir}", bLines=False)
    
    # 테스트 데이터 로더 생성
    test_loader = GetTestLoader(batch_size=batch_size, num_workers=4)
    
    if len(test_loader.dataset) == 0:
        OpLog("테스트 데이터셋이 비어있습니다", bLines=True)
        return None
    
    # 모델을 디바이스로 이동 (FasterRCNN의 경우)
    if hasattr(model, 'model') and isinstance(model, FasterRCNNModel):
        model.model.to(DEVICE_TYPE)
    
    # testMode 호출 (epoch=1, max_epochs=1로 설정)
    OpLog("테스트 모드 실행 중...", bLines=True)
    test_loss = model.testMode(test_loader, epoch=1, max_epochs=1)
    
    OpLog(f"TestModel 완료: test_loss={test_loss}", bLines=True)
    return test_loss

def MakeModel(model_type="yolov8s"):
    if model_type.startswith("yolov8"):
        model_size = model_type.replace("yolov8", "")
        model = YOLOv8Model(model_size=model_size)
    elif model_type.startswith("fasterrcnn"):
        backbone = model_type.replace("fasterrcnn_", "")
        model = FasterRCNNModel(backbone=backbone)
    else:
        raise ValueError(f"지원되지 않는 모델 타입: {model_type}")
    model.preJob()
    return model


def Single_train(
    model_type="yolov8s",
    transform_type="A",
    gubun="freeze",
    epochs=50,
    imgsz=640,
    batch_size=16,
    lr=0.001,
):
    train_loader, val_loader, test_loader = GetLoaders(
        annotations_dir=ANNOTATION_DIR,
        transform_type=transform_type,
        img_dir=TRAIN_IMG_DIR,
        batch_size=batch_size,
        train_ratio=0.8,
        num_workers=4,
    )
    model = MakeModel(model_type=model_type)
    model.preJob()

    # 통일된 인터페이스로 fit 호출
    model.fit(
        gubun=gubun,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=epochs,
        imgsz=imgsz,
        batch_size=batch_size,
        lr=lr,
    )


# ════════════════════════════════════════
# ▣ 08. 단계별 학습 함수 -jwj 추가
# ════════════════════════════════════════


def get_stage_paths(stage="stage1"):
    """
    단계별 데이터 경로를 반환

    Args:
        stage: 'stage1', 'stage2', 'stage3' 중 하나

    Returns:
        tuple: (annotation_dir, img_dir, stage_info)
    """
    if stage not in STAGE_CONFIGS:
        raise ValueError(
            f"잘못된 stage: {stage}. 'stage1', 'stage2', 'stage3' 중 하나를 선택하세요."
        )

    config = STAGE_CONFIGS[stage]
    return config["annotation_dir"], config["img_dir"], config


def Stage_train(
    stage="stage1",
    model_type="yolov8s",
    transform_type="A",
    gubun="freeze",
    epochs=50,
    imgsz=640,
    batch_size=16,
    lr=0.001,
    pretrained_model=None,
):
    """
    단계별 모델 학습

    Args:
        stage: 'stage1', 'stage2', 'stage3' 중 하나
        model_type: 모델 타입 ('yolov8s', 'fasterrcnn_resnet50' 등)
        transform_type: 데이터 증강 타입
        gubun: 최적화 방식
        epochs: 학습 에포크 수
        imgsz: 이미지 크기
        batch_size: 배치 크기
        lr: 학습률
        pretrained_model: 이전 단계에서 학습된 모델 경로 (전이학습용)

    Returns:
        model: 학습된 모델
    """
    # 단계별 경로 가져오기
    annotation_dir, img_dir, stage_info = get_stage_paths(stage)

    OpLog(f"{'='*100}", bLines=False)
    OpLog(f"{stage.upper()} 학습 시작: {stage_info['description']}", bLines=True)
    OpLog(f"   Annotation: {annotation_dir}", bLines=False)
    OpLog(f"   Images: {img_dir}", bLines=False)
    OpLog(f"{'='*100}", bLines=False)

    # 데이터 로더 생성
    train_loader, val_loader, test_loader = GetLoaders(
        annotations_dir=annotation_dir,
        transform_type=transform_type,
        img_dir=img_dir,
        batch_size=batch_size,
        train_ratio=0.8,
        num_workers=4,
    )

    # 모델 생성
    model = MakeModel(model_type=model_type)

    # 이전 단계 모델 로드 (전이학습)
    if pretrained_model and os.path.exists(pretrained_model):
        OpLog(f"이전 단계 모델 로드: {pretrained_model}", bLines=True)
        if model_type.startswith("yolov8"):
            model.load_yolo_model(pretrained_model)
        else:
            checkpoint = model.load_model(pretrained_model)
            if checkpoint:
                model.model.load_state_dict(checkpoint["model_state_dict"])

    # 모델 학습
    model.fit(
        gubun=gubun,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=epochs,
        imgsz=imgsz,
        batch_size=batch_size,
        lr=lr,
    )

    # 단계별 모델 저장
    stage_model_path = os.path.join(MODEL_FILES, f"{stage}_{model_type}_final.pt")
    if model_type.startswith("yolov8"):
        model.save_yolo_model(stage_model_path)
    else:
        model.save_model(
            epoch_index=epochs,
            is_best=True,
            model_state_dict=model.model.state_dict(),
        )

    OpLog(f"{stage.upper()} 학습 완료! 모델 저장: {stage_model_path}", bLines=True)

    return model, stage_model_path


def Progressive_train(
    model_type="yolov8s",
    transform_type="A",
    gubun="freeze",
    epochs_per_stage=[30, 20, 20],
    imgsz=640,
    batch_size=16,
    lr_per_stage=[0.001, 0.0005, 0.0002],
):
    """
    3단계 순차 학습 (Progressive Training)

    Args:
        model_type: 모델 타입
        transform_type: 데이터 증강 타입
        gubun: 최적화 방식
        epochs_per_stage: 각 단계별 에포크 수 [stage1, stage2, stage3]
        imgsz: 이미지 크기
        batch_size: 배치 크기
        lr_per_stage: 각 단계별 학습률 [stage1, stage2, stage3]

    Returns:
        dict: 각 단계별 모델 경로
    """
    OpLog("=" * 100, bLines=False)
    OpLog("단계별 순차 학습 시작 (Progressive Training)", bLines=True)
    OpLog("=" * 100, bLines=False)

    model_paths = {}
    pretrained_model = None

    stages = ["stage1", "stage2", "stage3"]

    for idx, stage in enumerate(stages):
        OpLog(f"\n{'='*100}", bLines=False)
        OpLog(f"{stage.upper()} ({idx+1}/3) 시작", bLines=True)
        OpLog(f"{'='*100}", bLines=False)

        _, model_path = Stage_train(
            stage=stage,
            model_type=model_type,
            transform_type=transform_type,
            gubun=gubun,
            epochs=epochs_per_stage[idx],
            imgsz=imgsz,
            batch_size=batch_size,
            lr=lr_per_stage[idx],
            pretrained_model=pretrained_model,
        )

        model_paths[stage] = model_path
        pretrained_model = model_path  # 다음 단계에서 사용

        OpLog(f"{stage.upper()} 완료\n", bLines=True)

    OpLog("=" * 100, bLines=False)
    OpLog("전체 단계별 학습 완료!", bLines=True)
    OpLog("=" * 100, bLines=False)

    for stage, path in model_paths.items():
        OpLog(f"  {stage}: {path}", bLines=False)

    return model_paths


# ════════════════════════════════════════
# ▣ 실행 예제
# ════════════════════════════════════════
if __name__ == "__main__":
    updateCount = update_category_id_from_dl_idx(ANNOTATION_DIR)
    Lines(f"Update Count:{updateCount}")
    
  