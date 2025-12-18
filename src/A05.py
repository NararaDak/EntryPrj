import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
from torchvision import transforms
import torch.nn as nn
import datetime
import sys
from filelock import FileLock
import matplotlib.pyplot as plt
from matplotlib import patches
from ultralytics import YOLO
import pandas as pd
import shutil
import yaml
import torchvision
import numpy as np
import re

plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 한글 폰트
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
plt.style.use('fivethirtyeight')

# ════════════════════════════════════════
# ▣ 01. 디렉토리 및 유틸 함수 설정
# ════════════════════════════════════════
VER = "2025.12.12.001.bestOne"
#BASE_DIR = "/content/drive/MyDrive/codeit/data"
BASE_DIR = r"D:\01.project\EntryPrj\data"
LOG_FILE = os.path.join(BASE_DIR, "operation.log")
SUBMISSTION_DIR = f"{BASE_DIR}/submission"

# 데이터 디렉토리 설정 함수
def GetConfig(data_dir):
    """
    데이터 디렉토리로부터 필요한 경로들을 생성
    
    Args:
        data_dir: 데이터 루트 디렉토리 (예: D:\01.project\EntryPrj\data\oraldrug\1.drug_Image_annotation_allOK)
    
    Returns:
        tuple: (image_dir, annotation_dir, yaml_file, yaml_label_dir, test_img_dir)
    """
    image_dir = os.path.join(data_dir, "train_images")
    annotation_dir = os.path.join(data_dir, "train_annotations")
    yaml_file = os.path.join(data_dir, "yolo_yaml.yaml")
    yaml_label_dir = os.path.join(data_dir, "yolo_labels")
    # test_img_dir는 oraldrug 디렉토리 밑에 위치 (data_dir의 부모 디렉토리)
    parent_dir = os.path.dirname(data_dir)  # oraldrug 디렉토리
    test_img_dir = os.path.join(parent_dir, "test_images")
    return image_dir, annotation_dir, yaml_file, yaml_label_dir, test_img_dir

# 공통 설정
MODEL_FILES = os.path.join(BASE_DIR, "modelfiles")
RESULT_CSV = f"{BASE_DIR}/entryprj.csv"
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

def GetIndexCategoryName(yaml_file, annotation_dir):
    """
    YAML 파일의 names와 annotation JSON 파일들을 매핑하여 
    [인덱스, category_id, dl_name] 3쌍 정보를 YAML names 순서대로 반환
    
    중요: YAML의 names 순서가 클래스 인덱스 순서입니다.
    - YAML names[0] = '2379' → 클래스 0 = category_id 2379
    - YAML names[1] = '111'  → 클래스 1 = category_id 111
    - 예측 결과 class=0 → index=0 → category_id=2379, dl_name='약이름1'
    - 예측 결과 class=1 → index=1 → category_id=111, dl_name='약이름2'
    
    Args:
        yaml_file: YAML 파일 경로 (names 리스트 포함)
        annotation_dir: annotation 디렉토리 경로
        
    Returns:
        list: [[index, category_id, dl_name], ...] 형태의 리스트 (YAML names 순서)
        dict: {index: {'category_id': int, 'dl_name': str}, ...} 형태의 딕셔너리
    """
    OpLog(f"GetIndexCategoryName 시작: {yaml_file}, {annotation_dir}", bLines=True)
    
    # 1. YAML 파일에서 names 로드 (순서 보존)
    if not os.path.exists(yaml_file):
        OpLog(f"YAML 파일을 찾을 수 없습니다: {yaml_file}", bLines=True)
        return [], {}
    
    with open(yaml_file, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
        class_names = yaml_data.get('names', [])  # YAML names 순서 그대로
    
    OpLog(f"YAML에서 {len(class_names)}개 클래스 로드 (순서 보존): {class_names[:5]}...", bLines=False)
    
    # 2. category_id -> dl_name 매핑 생성
    category_to_dlname = {}
    
    # annotation_dir의 모든 JSON 파일 검색
    for root, dirs, files in os.walk(annotation_dir):
        for json_file in files:
            if json_file.endswith(".json"):
                json_path = os.path.join(root, json_file)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # images 섹션에서 dl_idx와 dl_name 추출
                    if "images" in data and len(data["images"]) > 0:
                        for img_info in data["images"]:
                            dl_idx = img_info.get("dl_idx")
                            dl_name = img_info.get("dl_name")
                            
                            if dl_idx and dl_name:
                                # dl_idx를 int로 변환
                                try:
                                    category_id = int(dl_idx)
                                    if category_id not in category_to_dlname:
                                        category_to_dlname[category_id] = dl_name
                                        OpLog(f"  매핑 발견: category_id={category_id}, dl_name={dl_name}", bLines=False)
                                except (ValueError, TypeError):
                                    continue
                except Exception as e:
                    OpLog(f"Error reading {json_path}: {e}", bLines=False)
                    continue
    
    OpLog(f"총 {len(category_to_dlname)}개 category_id -> dl_name 매핑 생성", bLines=False)
    
    # 3. YAML names 순서대로 [index, category_id, dl_name] 리스트 생성
    # 중요: enumerate 순서가 클래스 인덱스 순서입니다!
    result_list = []
    result_dict = {}
    
    OpLog(f"YAML names 순서대로 클래스 매핑 생성:", bLines=True)
    for index, category_id_str in enumerate(class_names):
        try:
            category_id = int(category_id_str)
            dl_name = category_to_dlname.get(category_id, f"Unknown_{category_id}")
            
            # 순서대로 추가 (index가 클래스 번호)
            result_list.append([index, category_id, dl_name])
            result_dict[index] = {
                'category_id': category_id,
                'dl_name': dl_name
            }
            
            OpLog(f"  클래스[{index}] → category_id={category_id}, dl_name={dl_name}", bLines=False)
        except (ValueError, TypeError):
            OpLog(f"Warning: YAML의 names[{index}]='{category_id_str}'를 정수로 변환할 수 없습니다.", bLines=False)
            continue
    
    OpLog(f"GetIndexCategoryName 완료: {len(result_list)}개 매핑 생성 (YAML 순서 보존)", bLines=True)
    
    # 안전하게 예시 출력
    if len(result_dict) > 0:
        first_key = list(result_dict.keys())[0]
        OpLog(f"사용법 예시: 예측 class={first_key} → category_id={result_dict[first_key]['category_id']}, dl_name={result_dict[first_key]['dl_name']}", bLines=True)
    else:
        OpLog(f"경고: result_dict가 비어있습니다! YAML 파일과 annotation 디렉토리를 확인하세요.", bLines=True)
    
    return result_list, result_dict


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
                                # category_id를 int로 변환
                                try:
                                    category_id = int(category_id)
                                    unique_category_ids.add(category_id)
                                except (ValueError, TypeError):
                                    continue
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
                                # category_id를 int로 변환 (문자열인 경우 대비)
                                try:
                                    category_id = int(category_id)
                                except (ValueError, TypeError):
                                    continue
                                
                                # 동일 category_id가 여러 JSON 파일에 있을 수 있으므로 모두 저장
                                if category_id not in class_info:
                                    class_info[category_id] = []
                                if json_path not in class_info[category_id]:
                                    class_info[category_id].append(json_path)
                                break  # 한 JSON에서 category_id 하나만 찾으면 됨
                except Exception as e:
                    OpLog(f"Error reading {json_path}: {e}", bLines=True)
                    continue

    OpLog(f"get_class_mapping: 총 {json_count}개 JSON 파일 스캔, {len(class_info)}개 클래스 발견", bLines=True)

    # class_dirs 생성 (category_id, JSON 파일 목록)
    for category_id, json_paths in class_info.items():
        class_dirs.append((category_id, json_paths))

    # 클래스 정렬 및 인덱스 매핑 (0-based index로 변환)
    unique_classes = sorted(class_info.keys())
    # category_id를 0부터 시작하는 인덱스로 매핑
    class_to_idx = {category_id: idx for idx, category_id in enumerate(unique_classes)}
    idx_to_class = {idx: category_id for idx, category_id in enumerate(unique_classes)}

    return class_dirs, class_to_idx, idx_to_class, unique_classes


def analyze_image_json_mapping(train_img_dir, annotation_dir, output_dir):
    """
    이미지와 JSON 파일 간의 매핑 관계를 분석하여 CSV 파일로 저장
    
    Args:
        train_img_dir: 학습 이미지 디렉토리
        annotation_dir: 어노테이션 디렉토리
        output_dir: CSV 파일 저장 디렉토리

    생성 파일:
        - img_to_json.csv: 이미지 -> JSON 매핑 (이미지명, JSON경로)
        - json_to_img.csv: JSON -> 이미지 매핑 (JSON경로, 이미지명)
    """
    OpLog("이미지-JSON 매핑 분석 시작", bLines=True)
    
    img_to_json_file = os.path.join(output_dir, "img_to_json.csv")
    json_to_img_file = os.path.join(output_dir, "json_to_img.csv")

    # 1. train_img_dir 밑의 모든 이미지 파일 수집
    image_files = {}  # {filename: full_path}
    if os.path.exists(train_img_dir):
        for img_file in os.listdir(train_img_dir):
            if img_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                image_files[img_file] = os.path.join(train_img_dir, img_file)

    OpLog(f"이미지 파일 수: {len(image_files)}", bLines=True)

    # 2. annotation_dir 밑의 모든 JSON 파일 수집
    json_files = []  # [(json_path, images_in_json), ...]
    if os.path.exists(annotation_dir):
        for subdir in os.listdir(annotation_dir):
            subdir_path = os.path.join(annotation_dir, subdir)
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
                                        bLines=True,
                                    )

    OpLog(f"JSON 파일 수: {len(json_files)}", bLines=True)

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
    makedirs(os.path.dirname(img_to_json_file))
    with open(img_to_json_file, "w", encoding="utf-8") as f:
        f.write("Image,JSON\n")
        for img_name, json_path in img_to_json_mapping:
            f.write(f"{img_name},{json_path}\n")

    OpLog(
        f"IMG_TO_JSON 저장 완료: {img_to_json_file} ({len(img_to_json_mapping)}개 매핑)",
        bLines=True,
    )

    # 6. JSON_TO_IMG CSV 파일 저장
    with open(json_to_img_file, "w", encoding="utf-8") as f:
        f.write("JSON,Image\n")
        for json_path, img_name in json_to_img_mapping:
            f.write(f"{json_path},{img_name}\n")

    OpLog(
        f"JSON_TO_IMG 저장 완룜: {json_to_img_file} ({len(json_to_img_mapping)}개 매핑)",
        bLines=True,
    )

    # 7. 통계 정보 출력
    img_without_json = sum(
        1 for _, json_path in img_to_json_mapping if json_path == "NONE"
    )
    json_without_img = sum(
        1 for _, img_name in json_to_img_mapping if img_name == "NONE"
    )

    OpLog(f"매핑 분석 완료:", bLines=True)
    OpLog(f"  - 전체 이미지: {len(image_files)}개", bLines=True)
    OpLog(f"  - 전체 JSON: {len(json_files)}개", bLines=True)
    OpLog(f"  - JSON 없는 이미지: {img_without_json}개", bLines=True)
    OpLog(f"  - 이미지 없는 JSON: {json_without_img}개", bLines=True)

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

            OpLog(f"테스트 이미지 {len(self.samples)}개 로드 완료", bLines=True)
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
                        OpLog(f"Error reading {json_path}: {e}", bLines=True)

            OpLog(f"PillDataset 로드 완료: {len(self.samples)}개 샘플", bLines=True)

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
    test_img_dir,
    batch_size=32,
    train_ratio=0.8,
    num_workers=4,
):
    """
    전체 데이터셋을 train/val로 분할하여 DataLoader 생성
    
    Args:
        annotations_dir: 어노테이션 디렉토리
        transform_type: 변환 타입
        img_dir: 학습 이미지 디렉토리
        test_img_dir: 테스트 이미지 디렉토리
        batch_size: 배치 크기
        train_ratio: 학습 데이터 비율
        num_workers: 워커 수
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
    test_loader = GetTestLoader(test_img_dir, batch_size=batch_size, num_workers=num_workers)
    OpLog(
        f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}",
        bLines=True,
    )
    OpLog(f"Total classes: {len(full_dataset.class_to_idx)}", bLines=True)
    return train_loader, val_loader, test_loader


def GetTestLoader(test_img_dir, batch_size=16, num_workers=4):
    """
    테스트 데이터셋 로더 생성 (annotation 없음)

    Args:
        test_img_dir: 테스트 이미지 디렉토리
        batch_size: 배치 크기
        num_workers: 워커 수

    Returns:
        test_loader: 테스트 데이터 로더
    """
    from torch.utils.data import DataLoader

    # 테스트 데이터셋 로드 (annotation 없음, 증강 없음)
    test_dataset = GetDataset(
        annotations_dir=None,  # 테스트는 annotation 불필요
        img_dir=test_img_dir,
        transform_type="default",  # 증강 없음
        is_test=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 테스트는 shuffle 안함
        num_workers=num_workers,
    )

    OpLog(f"Test samples: {len(test_dataset)}", bLines=True)
    return test_loader


# 사용 예시:
# data_dir = r"D:\01.project\EntryPrj\data\oraldrug\1.drug_Image_annotation_allOK"
# image_dir, annotation_dir, yaml_file, yaml_label_dir, test_img_dir = GetConfig(data_dir)
# train_loader, val_loader, test_loader = GetLoaders(
#     annotation_dir, "A", image_dir, test_img_dir, batch_size=16, train_ratio=0.8, num_workers=2
# )


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

        # 서브클래스에서 재정의할 수 있도록 파일명 결정 로직
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
            OpLog(f"모델 저장됨: {filename}", bLines=True)

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

        checkpoint = torch.load(model_file, map_location=DEVICE_TYPE, weights_only=False)

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

    def _visualize_results(self, epoch, max_epochs, predictions, mode="eval", test_img_dir=None):
        """
        검증/테스트 결과 시각화 (공통 구현)

        Args:
            epoch: 현재 에포크 번호
            max_epochs: 전체 에포크 수
            predictions: 예측 결과 (dict 또는 list)
            mode: 'eval' 또는 'test'
            test_img_dir: 테스트 이미지 디렉토리 (mode='test'일 때 필요)
        """
        results_dir = os.path.join(BASE_DIR, "oraldrug", "results", self.getMyName())
        makedirs(results_dir)

        if mode == "eval":
            # 검증 모드: 메트릭 표시
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # predictions가 dict 형태일 경우 (메트릭 정보)
            if isinstance(predictions, dict):
                metrics_text = f"Epoch {epoch}/{max_epochs} - Validation Metrics\n\n"
                metrics_text += f"mAP50: {predictions.get('mAP50', 0) if predictions.get('mAP50') else 'N/A'}\n"
                metrics_text += f"mAP50-95: {predictions.get('mAP50_95', 0) if predictions.get('mAP50_95') else 'N/A'}\n"
                metrics_text += f"Precision: {predictions.get('precision', 0):.4f if predictions.get('precision') else 'N/A'}\n"
                metrics_text += f"Recall: {predictions.get('recall', 0):.4f if predictions.get('recall') else 'N/A'}"
            else:
                # predictions가 list 형태일 경우
                num_preds = len(predictions) if isinstance(predictions, list) else 0
                metrics_text = f"Epoch {epoch}/{max_epochs} - Validation\n\n"
                metrics_text += f"Total Predictions: {num_preds}"

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

            OpLog(f"검증 결과 저장: {save_path}", bLines=True)

        elif mode == "test":
            # 테스트 모드: 예측 결과 샘플 시각화 (최대 6개)
            if not isinstance(predictions, list):
                OpLog("Warning: predictions가 list 형태가 아닙니다.", bLines=True)
                return
                
            num_samples = min(6, len(predictions))
            if num_samples == 0:
                return

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for i in range(num_samples):
                pred = predictions[i]
                
                # pred가 dict이고 filename 키를 가지고 있는지 확인
                if isinstance(pred, dict) and "filename" in pred:
                    filename = pred["filename"]
                    num_boxes = len(pred.get("boxes", []))
                    
                    if test_img_dir and os.path.exists(test_img_dir):
                        img_path = os.path.join(test_img_dir, filename)
                        if os.path.exists(img_path):
                            img = Image.open(img_path).convert("RGB")
                            axes[i].imshow(img)
                            axes[i].set_title(
                                f"{filename}\nDetections: {num_boxes}"
                            )
                            axes[i].axis("off")
                        else:
                            axes[i].text(
                                0.5,
                                0.5,
                                f"Image not found:\n{filename}",
                                ha="center",
                                va="center",
                            )
                            axes[i].axis("off")
                    else:
                        # test_img_dir이 없으면 텍스트만 표시
                        axes[i].text(
                            0.5,
                            0.5,
                            f"{filename}\nDetections: {num_boxes}",
                            ha="center",
                            va="center",
                        )
                        axes[i].set_title(f"Sample {i+1}")
                        axes[i].axis("off")
                else:
                    # 예측 결과가 dict 형태가 아닐 경우 기본 표시
                    num_detections = len(pred.get("boxes", [])) if isinstance(pred, dict) else 0
                    avg_score = pred.get("scores", torch.tensor([])).mean().item() if isinstance(pred, dict) and len(pred.get("scores", [])) > 0 else 0

                    axes[i].text(
                        0.5,
                        0.5,
                        f"Detections: {num_detections}\nAvg Score: {avg_score:.3f}",
                        ha="center",
                        va="center",
                        fontsize=12,
                    )
                    axes[i].set_title(f"Sample {i+1}")
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

            OpLog(f"테스트 결과 저장: {save_path}", bLines=True)

    def _save_eval_metrics(self, epoch, max_epochs, metrics_dict, train_loss=None, val_loss=None, current_lr=None):
        """
        검증 메트릭 저장 및 시각화 (공통 헬퍼)
        
        Args:
            epoch: 현재 에포크
            max_epochs: 전체 에포크 수
            metrics_dict: 메트릭 딕셔너리 {'mAP50': float, 'mAP50_95': float, 'precision': float, 'recall': float}
            train_loss: 학습 손실
            val_loss: 검증 손실
            current_lr: 현재 학습률
        """
        # CSV에 평가 메트릭 저장
        self.save_metrics_to_csv(
            model_name=self.getMyName(),
            epoch_index=epoch,
            max_epochs=max_epochs,
            train_loss=train_loss,
            val_loss=val_loss,
            current_lr=current_lr,
            mode="eval",
            mAP50=metrics_dict.get('mAP50'),
            mAP50_95=metrics_dict.get('mAP50_95'),
            precision=metrics_dict.get('precision'),
            recall=metrics_dict.get('recall'),
            total_detections=metrics_dict.get('total_detections'),
            avg_confidence=metrics_dict.get('avg_confidence'),
        )
        
        # 시각화
        self._visualize_results(epoch, max_epochs, metrics_dict, mode="eval")

    def _save_test_metrics(self, epoch, max_epochs, predictions, test_img_dir, train_loss=None, current_lr=None):
        """
        테스트 메트릭 저장 및 시각화 (공통 헬퍼)
        
        Args:
            epoch: 현재 에포크
            max_epochs: 전체 에포크 수
            predictions: 예측 결과 리스트 [{'boxes': array, 'scores': array, 'labels': array, 'filename': str}, ...]
            test_img_dir: 테스트 이미지 디렉토리
            train_loss: 학습 손실
            current_lr: 현재 학습률
        """
        # 통계 계산
        total_detections = sum(len(p["boxes"]) for p in predictions)
        
        # 평균 신뢰도 계산 (numpy array와 torch tensor 모두 지원)
        confidences = []
        for p in predictions:
            scores = p.get("scores", [])
            if len(scores) > 0:
                if hasattr(scores, 'mean'):  # numpy array or torch tensor
                    conf = float(scores.mean())
                else:
                    conf = sum(scores) / len(scores)
                confidences.append(conf)
        
        avg_confidence = sum(confidences) / max(len(confidences), 1) if confidences else 0.0
        test_loss = 1.0 - avg_confidence
        
        # 정밀도/재현율 근사 계산
        precision = avg_confidence if avg_confidence > 0 else None
        recall = avg_confidence * 0.9 if avg_confidence > 0 else None
        
        OpLog(
            f"Epoch [{epoch}/{max_epochs}] - Test: {len(predictions)} images, {total_detections} detections, Avg Conf: {avg_confidence:.4f}",
            bLines=True,
        )
        
        # CSV에 테스트 메트릭 저장
        self.save_metrics_to_csv(
            model_name=self.getMyName(),
            epoch_index=epoch,
            max_epochs=max_epochs,
            train_loss=train_loss,
            test_loss=test_loss,
            current_lr=current_lr,
            mode="test",
            total_detections=total_detections,
            avg_confidence=avg_confidence,
            precision=precision,
            recall=recall,
        )
        
        # 시각화 (최대 6개 샘플)
        self._visualize_results(epoch, max_epochs, predictions[:6], mode="test", test_img_dir=test_img_dir)

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
        patience=10,
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
            patience: Early stopping patience (검증 손실이 개선되지 않을 때 대기할 에포크 수)
        """
        raise NotImplementedError("fit must be implemented by subclass")

    def evalModel(self, val_loader, epoch, max_epochs):
        """검증 모드 - 서브클래스에서 구현 필요"""
        raise NotImplementedError("evalModel must be implemented by subclass")

    def testModel(self, test_loader, epoch, max_epochs):
        """테스트 모드 - 서브클래스에서 구현 필요"""
        raise NotImplementedError("testMode must be implemented by subclass")
    
    def predict_single_image(self, image_file, conf=0.25):
        """단일 이미지 예측 - 서브클래스에서 구현 필요
        
        Args:
            image_file: 예측할 이미지 파일 경로
            conf: 신뢰도 임계값
            
        Returns:
            dict: {'boxes': np.array, 'scores': np.array, 'labels': np.array}
        """
        raise NotImplementedError("predict_single_image must be implemented by subclass")
    
    
    
    def CreateSubmission(self, predictions, test_img_dir, class_names, save_image_num=10):
        """제출 파일 생성 (공통 구현)
        
        Args:
            predictions: 예측 결과 리스트 [{'boxes': array, 'scores': array, 'labels': array, 'filename': str}, ...]
            test_img_dir: 테스트 이미지 디렉토리
            class_names: 클래스 이름 리스트 (category_id 순서대로)
            save_image_num: 저장할 이미지 개수
        """
        import matplotlib.patches as patches
        import re
        
        # 타임스탬프로 디렉토리 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        submission_dir = os.path.join(SUBMISSTION_DIR, f"submission{timestamp}")
        makedirs(submission_dir)
        
        csv_path = os.path.join(submission_dir, f"submission{timestamp}.csv")
        
        OpLog(f"Submission 생성 시작: {submission_dir}", bLines=True)
        
        # CSV 데이터 준비
        csv_data = []
        annotation_id = 1
        
        for pred in predictions:
            # 이미지 ID 추출 (파일명에서 숫자 추출)
            filename = pred.get('filename', '')
            match = re.search(r'(\d+)', filename)
            image_id = int(match.group(1)) if match else annotation_id
            
            boxes = pred.get('boxes', [])
            scores = pred.get('scores', [])
            labels = pred.get('labels', [])
            
            # numpy array를 리스트로 변환
            if hasattr(boxes, 'tolist'):
                boxes = boxes.tolist() if len(boxes) > 0 else []
            if hasattr(scores, 'tolist'):
                scores = scores.tolist() if len(scores) > 0 else []
            if hasattr(labels, 'tolist'):
                labels = labels.tolist() if len(labels) > 0 else []
            
            # 각 탐지 결과를 CSV에 추가
            for box, score, label in zip(boxes, scores, labels):
                # box 형식: [x1, y1, x2, y2] -> [x, y, w, h]
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    bbox_x = int(x1)
                    bbox_y = int(y1)
                    bbox_w = int(x2 - x1)
                    bbox_h = int(y2 - y1)
                    
                    # label은 인덱스이므로 class_names에서 실제 category_id 가져오기
                    label_idx = int(label)
                    actual_category_id = int(class_names[label_idx]) if label_idx < len(class_names) else label_idx
                    
                    csv_data.append({
                        'annotation_id': annotation_id,
                        'image_id': image_id,
                        'category_id': actual_category_id,
                        'bbox_x': bbox_x,
                        'bbox_y': bbox_y,
                        'bbox_w': bbox_w,
                        'bbox_h': bbox_h,
                        'score': round(float(score), 2)
                    })
                    annotation_id += 1
        
        # CSV 파일 저장
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        OpLog(f"CSV 파일 저장: {csv_path} ({len(csv_data)}개 탐지)", bLines=True)
        
        # 이미지 시각화 및 저장
        num_to_save = min(save_image_num, len(predictions))
        OpLog(f"이미지 시각화 시작: {num_to_save}개", bLines=True)
        
        for idx in range(num_to_save):
            pred = predictions[idx]
            filename = pred.get('filename', f'image_{idx}.jpg')
            img_path = os.path.join(test_img_dir, filename)
            
            if not os.path.exists(img_path):
                OpLog(f"이미지를 찾을 수 없습니다: {img_path}", bLines=True)
                continue
            
            # 이미지 로드
            img = Image.open(img_path).convert('RGB')
            fig, ax = plt.subplots(1, figsize=(12, 12))
            ax.imshow(img)
            
            boxes = pred.get('boxes', [])
            scores = pred.get('scores', [])
            labels = pred.get('labels', [])
            
            # numpy array 변환
            if hasattr(boxes, 'tolist'):
                boxes = boxes.tolist() if len(boxes) > 0 else []
            if hasattr(scores, 'tolist'):
                scores = scores.tolist() if len(scores) > 0 else []
            if hasattr(labels, 'tolist'):
                labels = labels.tolist() if len(labels) > 0 else []
            
            # 각 박스 그리기
            for box, score, label in zip(boxes, scores, labels):
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    
                    # 박스 그리기
                    rect = patches.Rectangle(
                        (x1, y1), width, height,
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # 카테고리 ID와 약품 이름 표시
                    label_idx = int(label)
                    category_name = str(class_names[label_idx]) if label_idx < len(class_names) else str(label_idx)
                    
                    # 레이블 텍스트
                    label_text = f"Cat:{category_name}\nScore:{score:.2f}"
                    
                    # 텍스트 배경 박스
                    ax.text(
                        x1, y1 - 5,
                        label_text,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                        fontsize=8,
                        color='black',
                        verticalalignment='top'
                    )
            
            ax.axis('off')
            ax.set_title(f"{filename} - {len(boxes)} detections", fontsize=14, pad=10)
            
            # 이미지 저장
            save_img_path = os.path.join(submission_dir, f"result_{idx+1:03d}_{filename}")
            plt.savefig(save_img_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            OpLog(f"  [{idx+1}/{num_to_save}] {filename} 저장 완료", bLines=True)
        
        OpLog(f"Submission 생성 완료: {submission_dir}", bLines=True)
        OpLog(f"  - CSV: {csv_path}", bLines=True)
        OpLog(f"  - 이미지: {num_to_save}개", bLines=True)
        
        return csv_path, submission_dir

    


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
            num_classes: 클래스 수 (필수 파라미터)
        """
        super(YOLOv8Model, self).__init__()
        self.model_size = model_size
        self.num_classes = num_classes

        # YOLOv8 모델 초기화 (사전 학습된 가중치 사용)
        model_path = f"yolov8{model_size}.pt"
        
        # 모델 파일이 손상된 경우 삭제하고 재다운로드
        if os.path.exists(model_path):
            try:
                # 파일 검증 시도
                test_model = YOLO(model_path)
                self.model = test_model
                OpLog(f"YOLOv8{model_size} 모델 로드 완료", bLines=True)
            except Exception as e:
                OpLog(f"YOLOv8 모델 파일 손상 감지, 재다운로드 중...: {e}", bLines=True)
                try:
                    os.remove(model_path)
                    OpLog(f"손상된 모델 파일 삭제: {model_path}", bLines=True)
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
            bLines=True,
        )
        # YOLOv8는 ultralytics 내부에서 optimizer를 자동 관리
        return None

    def preJob(self, annotation_dir, image_dir, yaml_file, yaml_label_dir):
        """
        전처리 작업: YOLO YAML 파일, 클래스 매핑, YOLO 형식 레이블 생성
        
        Args:
            annotation_dir: 어노테이션 디렉토리
            image_dir: 이미지 디렉토리
            yaml_file: YAML 파일 경로
            yaml_label_dir: YOLO 레이블 디렉토리
        """
        import yaml

        class_mapping_file = os.path.join(os.path.dirname(yaml_file), "class_mapping.json")

        # 기존 YAML 파일과 labels 삭제 (새로 생성하기 위해)
        if os.path.exists(yaml_file):
            OpLog(f"기존 YAML 파일이 있으므로 preJob종료: {yaml_file}", bLines=True)
            return
        
        if os.path.exists(yaml_label_dir):
            try:
                import shutil
                shutil.rmtree(yaml_label_dir)
                OpLog(f"기존 labels 디렉토리 삭제: {yaml_label_dir}", bLines=True)
            except Exception as e:
                OpLog(f"labels 디렉토리 삭제 실패: {e}", bLines=True)

        OpLog("YOLO 데이터셋 준비 시작", bLines=True)

        # get_class_mapping 함수 사용하여 클래스 정보 가져오기
        class_dirs, class_to_idx, idx_to_class, class_names = get_class_mapping(
            annotation_dir
        )

        # 클래스 매핑 정보 저장 (category_id: index)
        class_mapping = {}
        for category_id in class_names:
            class_mapping[str(category_id)] = {"index": class_to_idx[category_id]}

        # 클래스 매핑 JSON 파일 저장
        with open(class_mapping_file, "w", encoding="utf-8") as f:
            json.dump(class_mapping, f, ensure_ascii=False, indent=2)

        # JSON annotation을 YOLO 형식(.txt)으로 변환
        OpLog("JSON annotation을 YOLO 형식으로 변환 중...", bLines=True)
        
        # 레이블 파일은 이미지와 같은 디렉토리에 생성 (YOLO 요구사항)
        # yolo_label_dir는 사용하지 않고 image_dir에 직접 생성
        # makedirs(yaml_label_dir)  # 사용 안 함

        # 모든 이미지별 annotation을 수집하기 위한 딕셔너리
        # key: 이미지 파일명, value: {'width': int, 'height': int, 'annotations': [{'bbox': [], 'category_id': int}]}
        image_annotations_dict = {}

        # 1단계: 모든 JSON 파일에서 annotation 수집
        total_annotations = 0
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
                                # 이미지별로 annotation 수집
                                if img_filename not in image_annotations_dict:
                                    image_annotations_dict[img_filename] = {
                                        'width': img_width,
                                        'height': img_height,
                                        'annotations': []
                                    }
                                
                                # annotation 추가 (중복 방지)
                                for ann in img_annotations:
                                    bbox = ann.get("bbox", [])
                                    ann_category_id = ann.get("category_id", category_id)
                                    
                                    if len(bbox) == 4:
                                        # 중복 체크: bbox 좌표와 category_id가 모두 같은 경우만 중복으로 판단
                                        is_duplicate = False
                                        for existing_ann in image_annotations_dict[img_filename]['annotations']:
                                            if (abs(existing_ann['bbox'][0] - bbox[0]) < 0.01 and
                                                abs(existing_ann['bbox'][1] - bbox[1]) < 0.01 and
                                                abs(existing_ann['bbox'][2] - bbox[2]) < 0.01 and
                                                abs(existing_ann['bbox'][3] - bbox[3]) < 0.01 and
                                                existing_ann['category_id'] == ann_category_id):
                                                is_duplicate = True
                                                break
                                        
                                        if not is_duplicate:
                                            image_annotations_dict[img_filename]['annotations'].append({
                                                'bbox': bbox,
                                                'category_id': ann_category_id
                                            })
                                            total_annotations += 1
                                
                except Exception as e:
                    OpLog(f"Error reading {json_path}: {e}", bLines=True)

        OpLog(f"총 {total_annotations}개 annotation 수집 완료 ({len(image_annotations_dict)}개 이미지)", bLines=True)

        # 2단계: 수집된 annotation을 YOLO 형식 파일로 저장
        converted_count = 0
        for img_filename, img_data in image_annotations_dict.items():
            try:
                img_width = img_data['width']
                img_height = img_data['height']
                annotations = img_data['annotations']
                
                if len(annotations) == 0:
                    OpLog(f"Warning: {img_filename}에 annotation이 없습니다.", bLines=True)
                    continue
                
                # YOLO 형식 레이블 파일 생성 (이미지와 같은 디렉토리에 생성)
                label_filename = os.path.splitext(img_filename)[0] + ".txt"
                label_path = os.path.join(image_dir, label_filename)  # yaml_label_dir 대신 image_dir 사용

                with open(label_path, "w", encoding="utf-8") as lf:
                    for ann in annotations:
                        bbox = ann['bbox']
                        x, y, w, h = bbox

                        # YOLO 형식으로 변환: [x_center, y_center, width, height] (0~1 정규화)
                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        norm_width = w / img_width
                        norm_height = h / img_height

                        # category_id를 YOLO class index(0-based)로 변환
                        ann_category_id = ann['category_id']
                        try:
                            ann_category_id = int(ann_category_id)
                        except (ValueError, TypeError):
                            ann_category_id = class_names[0]
                        
                        try:
                            yolo_class_idx = class_names.index(ann_category_id)
                        except ValueError:
                            OpLog(f"Warning: category_id {ann_category_id} not in class_names, skipping", bLines=True)
                            continue
                        
                        lf.write(
                            f"{yolo_class_idx} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n"
                        )

                converted_count += 1
                
                # 디버깅: 여러 객체가 있는 이미지 로그
                if len(annotations) > 1:
                    OpLog(f"{img_filename}: {len(annotations)}개 객체 저장됨", bLines=True)
                    
            except Exception as e:
                OpLog(f"Error writing label file for {img_filename}: {e}", bLines=True)

        OpLog(f"YOLO 레이블 변환 완료: {converted_count}개 파일", bLines=True)

        # YAML 데이터 구조 생성 (모든 경로를 절대 경로로 사용)
        # names를 리스트로 설정 (YOLO는 0-based index 사용)
        # class 0 = category_id class_names[0], class 1 = category_id class_names[1], ...
        
        # test_img_dir 계산 (image_dir의 부모 디렉토리에서 test_images 찾기)
        data_root = os.path.dirname(image_dir)
        test_img_dir = os.path.join(data_root, "test_images")
        
        yaml_data = {
            "path": os.path.abspath(data_root).replace("\\", "/"),
            "train": os.path.abspath(image_dir).replace("\\", "/"),
            "val": os.path.abspath(image_dir).replace("\\", "/"),
            "test": os.path.abspath(test_img_dir).replace("\\", "/"),
            "nc": len(class_names),
            "names": [str(cat_id) for cat_id in class_names],  # 리스트 형태: ['1899', '2482', '3350', ...]
        }

        # YAML 파일 저장
        makedirs(os.path.dirname(yaml_file))
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(
                yaml_data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        OpLog(f"YAML 파일 생성 완료: {yaml_file}", bLines=True)
        OpLog(f"  - train: {yaml_data['train']}", bLines=True)
        OpLog(f"  - val: {yaml_data['val']}", bLines=True)
        OpLog(f"  - test: {yaml_data['test']}", bLines=True)
        OpLog(f"클래스 매핑 파일 생성 완료: {class_mapping_file}", bLines=True)
        OpLog(f"총 클래스 수: {len(class_names)}", bLines=True)

    def fit(
        self,
        annotation_dir,
        image_dir,
        yaml_file,
        yaml_label_dir,
        test_img_dir,
        gubun="freeze",
        train_loader=None,
        val_loader=None,
        test_loader=None,
        epochs=50,
        imgsz=640,
        batch_size=16,
        lr=0.001,
        patience=10,
    ):
        """
        YOLOv8 모델 학습 (BaseModel 인터페이스 준수)

        Args:
            annotation_dir: 어노테이션 디렉토리
            image_dir: 학습 이미지 디렉토리
            yaml_file: YAML 파일 경로
            yaml_label_dir: YOLO 레이블 디렉토리
            test_img_dir: 테스트 이미지 디렉토리
            gubun: 최적화 방식 (YOLOv8는 사용하지 않음, 인터페이스 통일용)
            train_loader: 학습 데이터로더 (YOLOv8는 내부적으로 사용하지 않지만 인터페이스 통일)
            val_loader: 검증 데이터로더
            test_loader: 테스트 데이터로더
            epochs: 학습 에포크 수
            imgsz: 입력 이미지 크기
            batch_size: 배치 크기
            lr: 학습률 (YOLOv8는 내부적으로 관리)
            patience: Early stopping patience (과적합 방지)
        """
        OpLog(f"YOLOv8{self.model_size} 모델 학습 시작", bLines=True)

        # YAML 파일과 labels를 항상 새로 생성
        OpLog("YAML 파일 및 레이블 재생성 중...", bLines=True)
        self.preJob(annotation_dir, image_dir, yaml_file, yaml_label_dir)

        # YOLOv8 학습 시작 (YOLOv8는 내부적으로 전체 에포크를 학습하고 검증까지 자동 수행)
        data_root = os.path.dirname(image_dir)
        results = self.model.train(
            data=yaml_file,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=DEVICE_TYPE,
            project=os.path.join(data_root, "yolo_results"),
            name=f"yolov8{self.model_size}_train",
            exist_ok=True,
            patience=patience,  # Early stopping (과적합시 훈련 종료)
            save=True,
            plots=True,
        )

        OpLog(f"YOLOv8 학습 완료!", bLines=True)

        # YOLOv8 best/last 모델 파일 복사
        try:
            results_dir = os.path.join(data_root, "yolo_results", f"yolov8{self.model_size}_train")
            weights_dir = os.path.join(results_dir, "weights")
            best_pt = os.path.join(weights_dir, "best.pt")
            last_pt = os.path.join(weights_dir, "last.pt")
            
            makedirs(MODEL_FILES)
            if os.path.exists(best_pt):
                shutil.copy(best_pt, os.path.join(MODEL_FILES, "yolobest.pt"))
                OpLog(f"YOLOv8 best 모델 복사 완료: yolobest.pt", bLines=True)
            
            if os.path.exists(last_pt):
                shutil.copy(last_pt, os.path.join(MODEL_FILES, "yololast.pt"))
                OpLog(f"YOLOv8 last 모델 복사 완료: yololast.pt", bLines=True)
        except Exception as e:
            OpLog(f"YOLOv8 모델 파일 복사 중 오류: {e}", bLines=True)

        # YOLOv8 학습 결과에서 각 에폭의 메트릭을 CSV에 저장
        try:
            # YOLOv8 results.csv 파일에서 메트릭 읽기
            results_dir = os.path.join(data_root, "yolo_results", f"yolov8{self.model_size}_train")
            results_csv = os.path.join(results_dir, "results.csv")
            
            if os.path.exists(results_csv):
                import pandas as pd
                yolo_results = pd.read_csv(results_csv)
                
                # 각 에폭의 메트릭을 RESULT_CSV에 저장
                for idx, row in yolo_results.iterrows():
                    epoch_num = idx + 1
                    
                    # YOLOv8 결과에서 메트릭 추출 (컬럼명은 YOLOv8 버전에 따라 다를 수 있음)
                    train_loss = row.get('train/box_loss', 0) + row.get('train/cls_loss', 0) + row.get('train/dfl_loss', 0)
                    val_loss = row.get('val/box_loss', 0) + row.get('val/cls_loss', 0) + row.get('val/dfl_loss', 0)
                    mAP50 = row.get('metrics/mAP50(B)', None)
                    mAP50_95 = row.get('metrics/mAP50-95(B)', None)
                    precision = row.get('metrics/precision(B)', None)
                    recall = row.get('metrics/recall(B)', None)
                    
                    # 학습 메트릭 저장
                    self.save_metrics_to_csv(
                        model_name=self.getMyName(),
                        epoch_index=epoch_num,
                        max_epochs=epochs,
                        train_loss=train_loss if train_loss > 0 else None,
                        current_lr=lr,
                        val_loss=val_loss if val_loss > 0 else None,
                        mAP50=mAP50,
                        mAP50_95=mAP50_95,
                        precision=precision,
                        recall=recall,
                        mode="train",
                    )
                
                OpLog(f"YOLOv8 학습 메트릭 {len(yolo_results)}개 에폭 저장 완료", bLines=True)
        except Exception as e:
            OpLog(f"YOLOv8 결과 파싱 중 오류: {e}", bLines=True)

        # 학습 완료 후 최종 검증 및 테스트 (1회만 수행)
        if val_loader is not None:
            OpLog("최종 검증 수행 중...", bLines=True)
            self.evalModel(yaml_file, val_loader, epochs, epochs)

        if test_loader is not None:
            OpLog("최종 테스트 수행 중...", bLines=True)
            self.testModel(test_img_dir, test_loader, epochs, epochs)

        # 학습 결과 시각화
        self.plot_results()

        return results

    def evalModel(self, yaml_file, val_loader, epoch, max_epochs):
        """
        검증 데이터셋에 대한 모델 평가 (BaseModel 인터페이스 구현)

        Args:
            yaml_file: YAML 파일 경로
            val_loader: 검증 데이터로더
            epoch: 현재 에포크 번호
            max_epochs: 전체 에포크 수
        """
        OpLog(f"[Epoch {epoch}/{max_epochs}] Validation 시작", bLines=True)

        # 모델 검증
        metrics = self.model.val(
            data=yaml_file,
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
            bLines=True,
        )

        # 메트릭 딕셔너리 생성
        metrics_dict = {
            "mAP50": mAP50,
            "mAP50_95": mAP50_95,
            "precision": precision,
            "recall": recall,
        }
        
        # 공통 헬퍼로 저장 및 시각화
        self._save_eval_metrics(
            epoch=epoch,
            max_epochs=max_epochs,
            metrics_dict=metrics_dict,
            train_loss=0.0,
            val_loss=val_loss,
            current_lr=0.0,
        )

        return val_loss

    def testModel(self, test_img_dir, test_loader, epoch, max_epochs):
        """
        테스트 데이터셋에 대한 모델 평가 (BaseModel 인터페이스 구현)

        Args:
            test_img_dir: 테스트 이미지 디렉토리
            test_loader: 테스트 데이터로더
            epoch: 현재 에포크 번호
            max_epochs: 전체 에포크 수
        """
        OpLog(f"[Epoch {epoch}/{max_epochs}] Test 시작", bLines=True)

        # 테스트 이미지에 대한 예측 수행
        results = self.model.predict(
            source=test_img_dir,
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

        OpLog(f"테스트 이미지 {len(predictions)}개 예측 완료", bLines=True)

        # 공통 헬퍼로 통계 계산, 저장 및 시각화
        self._save_test_metrics(
            epoch=epoch,
            max_epochs=max_epochs,
            predictions=predictions,
            test_img_dir=test_img_dir,
            train_loss=0.0,
            current_lr=0.0,
        )

        # test_loss 계산 후 반환
        total_detections = sum(len(p["boxes"]) for p in predictions)
        avg_conf = sum(
            p["scores"].mean() if len(p["scores"]) > 0 else 0.0 for p in predictions
        ) / max(len(predictions), 1)
        test_loss = 1.0 - avg_conf
        
        # Submission 파일 생성 (매 테스트마다 수행)
        # 클래스 이름 가져오기 - yaml_file은 test_img_dir의 부모의 하위 디렉토리에 있음
        # test_img_dir = data/oraldrug/test_images
        # data_dir = data/oraldrug/1.drug_Image_annotation_allOK (또는 유사)
        # yaml_file = data_dir/yolo_yaml.yaml
        
        parent_dir = os.path.dirname(test_img_dir)  # data/oraldrug
        yaml_file = None
        
        # 부모 디렉토리의 하위 디렉토리에서 yolo_yaml.yaml 찾기
        for item in os.listdir(parent_dir):
            item_path = os.path.join(parent_dir, item)
            if os.path.isdir(item_path):
                yaml_candidate = os.path.join(item_path, "yolo_yaml.yaml")
                if os.path.exists(yaml_candidate):
                    yaml_file = yaml_candidate
                    break
        
        if yaml_file and os.path.exists(yaml_file):
            import yaml
            with open(yaml_file, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
                class_names = yaml_data.get('names', [])
            OpLog(f"YAML 파일 로드: {yaml_file}", bLines=True)
        else:
            OpLog(f"Warning: yolo_yaml.yaml을 찾을 수 없습니다. 기본 클래스 이름 사용", bLines=True)
            class_names = list(range(self.num_classes)) if self.num_classes else []
        
        self.CreateSubmission(
            predictions=predictions,
            test_img_dir=test_img_dir,
            class_names=class_names,
            save_image_num=10
        )
        
        return test_loss
   

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

    def TestModelByBest(self, pt_file, test_img_dir, test_loader=None):
        """Best 모델 파일로 테스트 및 Submission 생성
        
        Args:
            pt_file: 로드할 .pt 모델 파일 경로
            test_img_dir: 테스트 이미지 디렉토리
            test_loader: 테스트 데이터 로더 (선택, 사용하지 않음)
        """
        OpLog(f"Best 모델로 테스트 시작: {pt_file}", bLines=True)
        
        # 모델 로드
        if not self.load_yolo_model(pt_file):
            OpLog(f"모델 로드 실패: {pt_file}", bLines=True)
            return False
        
        # testModel 호출 (epoch=1, max_epochs=1로 설정)
        self.testModel(test_img_dir, test_loader, epoch=1, max_epochs=1)
        
        OpLog(f"Best 모델 테스트 및 Submission 생성 완료", bLines=True)
        return True

    

    def predict_single_image(self, image_file, conf=0.25):
        """YOLOv8 단일 이미지 예측
        
        Args:
            image_file: 예측할 이미지 파일 경로
            conf: 신뢰도 임계값
            
        Returns:
            dict: {'boxes': np.array, 'scores': np.array, 'labels': np.array}
        """
        results = self.model.predict(
            source=image_file,
            conf=conf,
            save=False,
            device=DEVICE_TYPE,
            verbose=False,
        )
        
        result = results[0]
        return {
            'boxes': result.boxes.xyxy.cpu().numpy() if result.boxes else np.array([]),
            'scores': result.boxes.conf.cpu().numpy() if result.boxes else np.array([]),
            'labels': result.boxes.cls.cpu().numpy().astype(int) if result.boxes else np.array([])
        }

    def plot_results(self):
        """학습 결과 시각화"""
        results_dir = os.path.join(
            BASE_DIR, "yolo_results", f"yolov8{self.model_size}_train"
        )
        results_file = os.path.join(results_dir, "results.png")

        if os.path.exists(results_file):
            OpLog(f"학습 결과 그래프: {results_file}", bLines=True)
        else:
            OpLog("학습 결과 파일을 찾을 수 없습니다.", bLines=True)
        plt.close()


# ════════════════════════════════════════
# ▣ 06. Faster R-CNN 모델 정의
# ════════════════════════════════════════
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNNModel(BaseModel):
    def save_model(self, epoch_index, is_best=False, **kwargs):
        """현재 모델 상태를 저장 (FasterRCNN 전용 파일명)

        Args:
            epoch_index: 현재 에포크 번호
            is_best: Best 모델인지 여부
            **kwargs: 추가로 저장할 데이터 (model_state_dict, train_losses 등)
        """
        save_dir = MODEL_FILES
        makedirs(save_dir)

        # FasterRCNN 전용 파일명
        if is_best:
            filename = os.path.join(save_dir, "fasterbest.pt")
        else:
            filename = os.path.join(save_dir, "fasterlast.pt")

        # 기본 저장 데이터
        checkpoint = {
            "epoch": epoch_index,
            "is_best": is_best,
            "model_name": self.getMyName(),
        }

        # kwargs로 전달된 추가 데이터 저장
        checkpoint.update(kwargs)

        torch.save(checkpoint, filename)

        if is_best:
            print(f"  Best 모델 저장됨: {filename}")
            OpLog(f"Best model saved: {filename}")
        else:
            OpLog(f"모델 저장됨: {filename}", bLines=True)

    """
    Faster R-CNN 기반 객체 탐지 모델
    - torchvision의 사전 학습된 Faster R-CNN 사용
    - ResNet50-FPN 백본 활용
    - 객체 탐지 및 분류 동시 수행
    """

    def __init__(self, num_classes, backbone="resnet50"):
        """
        Args:
            num_classes: 클래스 수 (필수, +1은 배경 클래스)
            backbone: 백본 네트워크 ('resnet50', 'mobilenet' 등)
        """
        super(FasterRCNNModel, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes + 1  # +1 for background

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
            bLines=True,
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
                bLines=True,
            )
        elif gubun == "freeze":
            # Backbone을 고정하고 head만 학습
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            params = self.model.roi_heads.parameters()
            OpLog(f"Optimizer 생성: freeze mode (only head, lr={lr:.6f})", bLines=True)
        else:  # 'all' or default
            # 전체 모델 학습
            params = self.model.parameters()
            OpLog(f"Optimizer 생성: all mode (lr={lr:.6f})", bLines=True)

        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
        return optimizer

    def fit(
        self,
        test_img_dir,
        gubun="freeze",
        train_loader=None,
        val_loader=None,
        test_loader=None,
        epochs=50,
        imgsz=640,
        batch_size=16,
        lr=0.005,
        patience=10,
    ):
        """
        Faster R-CNN 모델 학습 (BaseModel 인터페이스 준수)

        Args:
            test_img_dir: 테스트 이미지 디렉토리
            gubun: 최적화 방식 ('freeze', 'partial', 'all')
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더 (선택)
            test_loader: 테스트 데이터 로더 (선택)
            epochs: 학습 에포크 수
            imgsz: 입력 이미지 크기 (FasterRCNN은 사용하지 않음, 인터페이스 통일용)
            batch_size: 배치 크기 (FasterRCNN은 사용하지 않음, 인터페이스 통일용)
            lr: 학습률
            patience: Early stopping patience (과적합 방지)
        """
        OpLog(f"Faster R-CNN 모델 학습 시작 (Epochs: {epochs}, LR: {lr}, Patience: {patience})", bLines=True)

        self.model.to(DEVICE_TYPE)

        # 옵티마이저 및 스케줄러 설정
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = self.getOptimizer(lr=lr, gubun="partial")
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=3, gamma=0.1
        )

        # Early stopping 변수 초기화
        patience_counter = 0
        best_val_loss = float('inf')

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
                msg =  f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {losses.item():.4f}"
                print(msg,end="\r")
              
                if batch_idx % 10 == 0:
                    OpLog(
                        f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {losses.item():.4f}",
                        bLines=True,
                    )

            # 에포크 평균 손실
            avg_train_loss = epoch_loss / batch_count
            self.train_losses.append(avg_train_loss)

            # 현재 학습률 가져오기
            current_lr = self.optimizer.param_groups[0]["lr"]

            # 매 epoch마다 학습 메트릭 저장
            self.save_metrics_to_csv(
                model_name=self.getMyName(),
                epoch_index=epoch + 1,
                max_epochs=epochs,
                train_loss=avg_train_loss,
                current_lr=current_lr,
                mode="train",
            )

            # 매 epoch 검증
            if val_loader is not None:
                self.evalModel(val_loader, epoch + 1, epochs)

                # Best 모델 저장 및 early stopping 검사
                current_val_loss = self.val_losses[-1]
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    self.best_val_loss = best_val_loss
                    patience_counter = 0
                    OpLog(f"Best 모델 업데이트! Val Loss: {best_val_loss:.4f}", bLines=True)
                else:
                    patience_counter += 1
                    OpLog(f"Patience counter: {patience_counter}/{patience}", bLines=True)
                    
                    if patience_counter >= patience:
                        OpLog(f"Early stopping triggered! {patience} epochs without improvement", bLines=True)
                        break
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
                self.testModel(test_img_dir, test_loader, epoch + 1, epochs)

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

        # 학습 완료 후 최종 테스트 (early stopping으로 중간에 종료되었을 수도 있으므로)
        if test_loader is not None:
            OpLog("최종 테스트 수행 중...", bLines=True)
            final_epoch = epoch + 1  # 실제 학습이 완료된 에포크
            self.testModel(test_img_dir, test_loader, final_epoch, epochs)

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

        # 예측 통계 계산
        total_detections = sum(len(p["boxes"]) for p in predictions_all)
        avg_confidence = sum(
            p["scores"].mean().item() if len(p["scores"]) > 0 else 0.0 
            for p in predictions_all
        ) / max(len(predictions_all), 1)
        
        # 정밀도/재현율 근사 계산 (신뢰도 기반)
        precision = avg_confidence if avg_confidence > 0 else None
        recall = avg_confidence * 0.9 if avg_confidence > 0 else None  # 근사값

        OpLog(
            f"Epoch [{epoch}/{max_epochs}] - Val Loss: {avg_val_loss:.4f}, Detections: {total_detections}, Avg Conf: {avg_confidence:.4f}", bLines=True
        )

        # 메트릭 딕셔너리 생성
        current_lr = self.optimizer.param_groups[0]["lr"]
        metrics_dict = {
            'mAP50': None,  # FasterRCNN은 mAP 계산 안 함
            'mAP50_95': None,
            'precision': precision,
            'recall': recall,
            'total_detections': total_detections,
            'avg_confidence': avg_confidence,
        }
        
        # 공통 헬퍼로 저장 및 시각화
        self._save_eval_metrics(
            epoch=epoch,
            max_epochs=max_epochs,
            metrics_dict=metrics_dict,
            train_loss=self.train_losses[-1] if len(self.train_losses) > 0 else None,
            val_loss=avg_val_loss,
            current_lr=current_lr,
        )

    def testModel(self, test_img_dir, test_loader, epoch, max_epochs):
        """테스트 모드
        
        Args:
            test_img_dir: 테스트 이미지 디렉토리
            test_loader: 테스트 데이터 로더 (사용하지 않음, 인터페이스 통일용)
            epoch: 현재 에포크
            max_epochs: 최대 에포크
        """
        self.model.eval()
        predictions_all = []
        
        # test_img_dir에서 직접 이미지 파일 리스트 가져오기
        if not os.path.exists(test_img_dir):
            OpLog(f"테스트 이미지 디렉토리가 없습니다: {test_img_dir}", bLines=True)
            return
        
        image_files = [
            f for f in os.listdir(test_img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        
        if len(image_files) == 0:
            OpLog(f"테스트 이미지가 없습니다: {test_img_dir}", bLines=True)
            return

        # 전처리 변환 (FasterRCNN 표준)
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        with torch.no_grad():
            for img_file in image_files:
                img_path = os.path.join(test_img_dir, img_file)
                
                # 이미지 로드 및 전처리
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).to(DEVICE_TYPE)
                
                # 예측 (배치 형태로 전달)
                preds = self.model([img_tensor])
                
                # 예측 결과에 파일명 추가
                pred_dict = {
                    "boxes": preds[0]["boxes"].cpu().numpy(),
                    "scores": preds[0]["scores"].cpu().numpy(),
                    "labels": preds[0]["labels"].cpu().numpy(),
                    "filename": img_file,
                }
                predictions_all.append(pred_dict)

        # 공통 헬퍼로 통계 계산, 저장 및 시각화
        current_lr = self.optimizer.param_groups[0]["lr"]
        self._save_test_metrics(
            epoch=epoch,
            max_epochs=max_epochs,
            predictions=predictions_all,
            test_img_dir=test_img_dir,
            train_loss=self.train_losses[-1] if len(self.train_losses) > 0 else None,
            current_lr=current_lr,
        )
        
        # Submission 파일 생성 (최종 에포크일 때만)
        # 클래스 이름 가져오기 (0-based index이므로 num_classes-1개)
        class_names = list(range(self.num_classes - 1))  # -1은 배경 클래스 제외
        self.CreateSubmission(
            predictions=predictions_all,
            test_img_dir=test_img_dir,
            class_names=class_names,
            save_image_num=10
        )

    def TestModelByBest(self, pt_file, test_img_dir, test_loader=None):
        """Best 모델 파일로 테스트 및 Submission 생성
        
        Args:
            pt_file: 로드할 .pt 모델 파일 경로
            test_img_dir: 테스트 이미지 디렉토리
            test_loader: 테스트 데이터 로더 (선택, 사용하지 않음)
        """
        OpLog(f"Best 모델로 테스트 시작: {pt_file}", bLines=True)
        
        # 모델 로드
        checkpoint = self.load_model(pt_file)
        if checkpoint is None:
            OpLog(f"모델 로드 실패: {pt_file}", bLines=True)
            return False
        
        # 모델 state_dict 로드
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(DEVICE_TYPE)
            OpLog(f"FasterRCNN 모델 로드 완료: {pt_file}", bLines=True)
        else:
            OpLog(f"model_state_dict를 찾을 수 없습니다: {pt_file}", bLines=True)
            return False
        
        # testModel 호출 (epoch=1, max_epochs=1로 설정)
        self.testModel(test_img_dir, test_loader, epoch=1, max_epochs=1)
        
        OpLog(f"Best 모델 테스트 및 Submission 생성 완료", bLines=True)
        return True

    

    def predict_single_image(self, image_file, conf=0.25):
        """FasterRCNN 단일 이미지 예측
        
        Args:
            image_file: 예측할 이미지 파일 경로
            conf: 신뢰도 임계값
            
        Returns:
            dict: {'boxes': np.array, 'scores': np.array, 'labels': np.array}
        """
        self.model.eval()
        self.model.to(DEVICE_TYPE)
        
        img = Image.open(image_file).convert('RGB')
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE_TYPE)
        
        with torch.no_grad():
            predictions = self.model(img_tensor)
        
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        # 신뢰도 필터링
        mask = scores >= conf
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        # FasterRCNN은 1-based label이므로 0-based로 변환
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels - 1  # 0-based로 변환
        }

    def plot_results(self):
        """학습 결과 시각화 (FasterRCNN용)"""
        if len(self.train_losses) == 0:
            OpLog("학습 이력이 없습니다.", bLines=True)
            return

    def plot_results(self):
        """학습 결과 시각화"""
        if len(self.train_losses) == 0:
            OpLog("학습 이력이 없습니다.", bLines=True)
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
        OpLog(f"학습 결과 그래프 저장: {save_path}", bLines=True)
        plt.close()


# ════════════════════════════════════════
# ▣ 07. 모델 생성 및 학습 실행
# ════════════════════════════════════════
def MakeModel(model_type, num_classes, model_size="n", backbone="resnet50", bBestLoad=False):
    """
    모델 생성 함수
    Args:
        model_type: 모델 유형 ("yolov8" 또는 "faster")
        num_classes: 클래스 수
        model_size: YOLOv8 모델 크기 ('n', 's', 'm', 'l', 'x'), 기본값 'n'
        backbone: FasterRCNN 백본 ('resnet50', 'mobilenet'), 기본값 'resnet50'
        bBestLoad: True이면 best 모델 파일 로드 (yolobest.pt 또는 fasterbest.pt), 기본값 False
    Returns:
        model: 생성된 모델 객체
    """
    if model_type == "faster":
        OpLog(f"FasterRCNN 모델 생성 중... (backbone={backbone}, num_classes={num_classes})", bLines=True)
        model = FasterRCNNModel(num_classes=num_classes, backbone=backbone)
        
        # bBestLoad가 True이면 best 모델 로드
        if bBestLoad:
            best_model_path = os.path.join(MODEL_FILES, "fasterbest.pt")
            if os.path.exists(best_model_path):
                OpLog(f"Best FasterRCNN 모델 로드 중: {best_model_path}", bLines=True)
                model.load_model(best_model_path)
            else:
                OpLog(f"Best 모델 파일이 없습니다: {best_model_path}. 새로운 모델로 시작합니다.", bLines=True)
        
        return model
    elif model_type == "yolov8":
        OpLog(f"YOLOv8 모델 생성 중... (model_size={model_size}, num_classes={num_classes})", bLines=True)
        model = YOLOv8Model(model_size=model_size, num_classes=num_classes)
        
        # bBestLoad가 True이면 best 모델 로드
        if bBestLoad:
            best_model_path = os.path.join(MODEL_FILES, "yolobest.pt")
            if os.path.exists(best_model_path):
                OpLog(f"Best YOLOv8 모델 로드 중: {best_model_path}", bLines=True)
                model.load_model(best_model_path)
            else:
                OpLog(f"Best 모델 파일이 없습니다: {best_model_path}. 새로운 모델로 시작합니다.", bLines=True)
        
        return model
    else:
        raise ValueError(f"지원되지 않는 모델 타입: {model_type}. 'yolov8' 또는 'faster'를 사용하세요.")

def Execute_Train(model_type, data_dir, model_size="n", backbone="resnet50", epochs=50, batch_size=16, lr=0.001, bBestLoad=False, **kwargs):
    """
    모델 생성 및 학습 실행 함수
    
    Args:
        model_type: 모델 유형 ("yolov8" 또는 "faster")
        data_dir: 데이터 디렉토리 경로
        model_size: YOLOv8 모델 크기 ('n', 's', 'm', 'l', 'x'), 기본값 'n'
        backbone: FasterRCNN 백본 ('resnet50', 'mobilenet'), 기본값 'resnet50'
        epochs: 학습 에포크 수, 기본값 50
        batch_size: 배치 크기, 기본값 16
        lr: 학습률, 기본값 0.001 (FasterRCNN은 자동으로 0.005 사용)
        bBestLoad: True이면 best 모델 로드 (yolobest.pt/fasterbest.pt), 기본값 False
        **kwargs: fit 메서드에 전달할 추가 파라미터
            - imgsz: 이미지 크기 (기본값 640)
            - patience: Early stopping patience (YOLOv8, 기본값 10)
            - gubun: 최적화 방식 (FasterRCNN, 기본값 'partial')
            - train_ratio: 학습/검증 분할 비율 (기본값 0.8)
            - num_workers: 데이터 로더 워커 수 (기본값 2)
            - transform_type: 데이터 증강 타입 (기본값 'A')
    """
    # kwargs에서 공통 파라미터 추출
    imgsz = kwargs.pop('imgsz', 640)
    patience = kwargs.pop('patience', 10)
    gubun = kwargs.pop('gubun', 'partial')
    train_ratio = kwargs.pop('train_ratio', 0.8)
    num_workers = kwargs.pop('num_workers', 2)
    transform_type = kwargs.pop('transform_type', 'A')
    
    # 경로 설정
    image_dir, annotation_dir, yaml_file, yaml_label_dir, test_img_dir = GetConfig(data_dir)
    
    # 클래스 수 계산
    num_classes = count_classes(annotation_dir)
    OpLog(f"총 클래스 수: {num_classes}", bLines=True)

    # 모델 생성 (bBestLoad 전달)
    model = MakeModel(model_type, num_classes=num_classes, model_size=model_size, backbone=backbone, bBestLoad=bBestLoad)
    
    # 데이터 로더 생성
    train_loader, val_loader, test_loader = GetLoaders(
        annotation_dir, transform_type, image_dir, test_img_dir, 
        batch_size=batch_size, train_ratio=train_ratio, num_workers=num_workers
    )
    
    # 모델 타입에 따라 fit 호출 방식 구분
    if isinstance(model, YOLOv8Model):
        # YOLOv8 모델 학습
        model.fit(
            annotation_dir=annotation_dir,
            image_dir=image_dir,
            yaml_file=yaml_file,
            yaml_label_dir=yaml_label_dir,
            test_img_dir=test_img_dir,
            epochs=epochs,
            imgsz=imgsz,
            batch_size=batch_size,
            lr=lr,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            patience=patience,
            **kwargs  # 추가 파라미터 전달
        )
    elif isinstance(model, FasterRCNNModel):
        # Faster R-CNN 모델 학습
        # FasterRCNN은 기본 lr이 더 높음
        actual_lr = lr if lr > 0.001 else 0.005
        model.fit(
            test_img_dir, 
            gubun=gubun, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            test_loader=test_loader,
            epochs=epochs, 
            imgsz=imgsz, 
            batch_size=batch_size, 
            lr=actual_lr,
            **kwargs  # 추가 파라미터 전달
        )
    else:
        raise ValueError(f"지원되지 않는 모델 타입: {type(model)}")


def testbest():
    data_dir = r"D:\01.project\EntryPrj\data\oraldrug\1.drug_Image_annotation_allOK"
    image_dir, annotation_dir, yaml_file, yaml_label_dir, test_img_dir = GetConfig(data_dir)
    # 클래스 수 계산
    num_classes = count_classes(annotation_dir)
    OpLog(f"총 클래스 수: {num_classes}", bLines=True)

    # 모델 생성 (bBestLoad 전달)
    model = MakeModel("yolov8", 74)
    model.TestModelByBest(MODEL_FILES + "/yolobest.pt", test_img_dir,20)


def testOne():
    
    image_file = r"D:\01.project\EntryPrj\data\oraldrug\test_images\1.png"
    data_dir = r"D:\01.project\EntryPrj\data\oraldrug\1.drug_Image_annotation_allOK"
    image_dir, annotation_dir, yaml_file, yaml_label_dir, test_img_dir = GetConfig(data_dir)
    # 클래스 수 계산
    num_classes = 74
    OpLog(f"총 클래스 수: {num_classes}", bLines=True)
    
    # GetIndexCategoryName은 (list, dict) 튜플을 반환합니다
    # list: [[index, category_id, dl_name], ...] (YAML names 순서)
    # dict: {index: {'category_id': int, 'dl_name': str}, ...}
    class_info_list, class_info_dict = GetIndexCategoryName(
        yaml_file=r'D:\01.project\EntryPrj\data\oraldrug\yolo_yaml.yaml',
        annotation_dir=r'D:\01.project\EntryPrj\data\oraldrug\1.drug_Image_annotation_allOK\train_annotations'
    )
    
    OpLog(f"클래스 정보 로드 완료: {len(class_info_dict)}개 클래스", bLines=True)
    
    # 클래스 정보 확인 (안전하게)
    if len(class_info_dict) > 0 and 0 in class_info_dict:
        OpLog(f"예시 - 클래스[0]: category_id={class_info_dict[0]['category_id']}, dl_name={class_info_dict[0]['dl_name']}", bLines=True)
    else:
        OpLog(f"경고: class_info_dict가 비어있거나 키 0이 없습니다. 키 목록: {list(class_info_dict.keys())[:5]}", bLines=True)
    
    # 모델 생성 (bBestLoad 전달)
    model = MakeModel("yolov8", 74, bBestLoad=True)
    model.load_model(MODEL_FILES + "/yolobest.pt")
 


    
