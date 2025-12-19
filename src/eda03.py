
from dataclasses import dataclass
import os
import sys
import datetime
import torch
import json
import hashlib
import csv
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter
from filelock import FileLock
from PIL import Image, ImageDraw, ImageFont, ImageTk
import numpy as np
import shutil
import tkinter as tk
from tkinter import simpledialog, messagebox

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 한글 폰트
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
plt.style.use('fivethirtyeight')
LOG_FILE = r".\logs\eda_operation.log"


# ════════════════════════════════════════
# ▣ 01. 유틸 함수 설정 
# ════════════════════════════════════════

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

makedirs(os.path.dirname(LOG_FILE))

## 운영 로그 함수
def OpLog(log):
    try:
        caller_name = sys._getframe(1).f_code.co_name
    except Exception:
        caller_name = "UnknownFunction"
        
    Lines(f"[{now_str()}]{caller_name}: {log}")
    log_filename = LOG_FILE
    log_lock_filename = log_filename + ".lock"
    log_content = f"[{now_str()}] {caller_name}: {log}\n"
    try:
        lock = FileLock(log_lock_filename, timeout=10)
        with lock:
            with open(log_filename, 'a', encoding='utf-8') as f:
                f.write(log_content)
    except Exception as e:
        print(f"Log write error: {e}")

# ════════════════════════════════════════
# ▣ 01.데이터 구조체.
# ════════════════════════════════════════

@dataclass
class AnnotationItem:
    image_file: str
    annotation_file: str
    bbox: list
    dl_name: str
    category_id: int
    dl_idx: str
  

def load_anotation(image_dir, annotation_dir):
    items = []
    if not os.path.exists(image_dir) or not os.path.exists(annotation_dir):
      return items
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]
    for img_file in image_files:
      base_name = os.path.splitext(img_file)[0]
      img_rel_path = img_file  # image_dir 기준 상대경로 (여기선 1단계)
      ann_paths = []
      for root, _, files in os.walk(annotation_dir):
        for f in files:
          if f.endswith('.json') and base_name in f:
            ann_abs_path = os.path.join(root, f)
            ann_rel_path = os.path.relpath(ann_abs_path, annotation_dir)
            ann_paths.append(ann_rel_path)
      for ann_rel_path in ann_paths:
        ann_abs_path = os.path.join(annotation_dir, ann_rel_path)
        try:
          with open(ann_abs_path, encoding='utf-8') as f:
            ann_data = json.load(f)
          # COCO 스타일인지 판별: 'images'와 'annotations'가 모두 있고, 'images'가 list
          if 'images' in ann_data and isinstance(ann_data['images'], list) and 'annotations' in ann_data and isinstance(ann_data['annotations'], list):
            # COCO 스타일 (id 없이도 file_name으로 매칭)
            for imginfo in ann_data['images']:
              if os.path.splitext(os.path.basename(imginfo.get('file_name','')))[0] == base_name:
                dl_name = imginfo.get('dl_name', ann_data.get('dl_name', ''))
                dl_idx = imginfo.get('dl_idx', "0")
                # annotation에서 image_id가 없으면 첫번째 annotation 사용
                found = False
                for ann in ann_data['annotations']:
                  # image_id가 있으면 매칭, 없으면 그냥 사용
                  if ('image_id' not in ann) or (imginfo.get('id') is not None and ann.get('image_id') == imginfo.get('id')):
                    bbox = ann.get('bbox', [])
                    category_id = ann.get('category_id', -1)
                    # dl_idx: annotation에 없으면 images[0]에서 가져오고, 그래도 없으면 "0"
                    ann_dl_idx = ann.get('dl_idx', None)
                    if ann_dl_idx is None:
                        if 'images' in ann_data and isinstance(ann_data['images'], list) and len(ann_data['images']) > 0:
                            ann_dl_idx = ann_data['images'][0].get('dl_idx', "0")
                        else:
                            ann_dl_idx = "0"
                    ann_dl_idx_str = str(ann_dl_idx) if ann_dl_idx is not None else "0"
                    item = AnnotationItem(
                      image_file=img_rel_path,
                      annotation_file=ann_rel_path,
                      bbox=bbox,
                      dl_name=dl_name,
                      category_id=category_id,
                      dl_idx=ann_dl_idx_str
                    )
                    items.append(item)
                    found = True
                    break
                if not found and ann_data['annotations']:
                  # image_id 매칭 실패시 첫 annotation 사용
                  ann = ann_data['annotations'][0]
                  bbox = ann.get('bbox', [])
                  category_id = ann.get('category_id', -1)
                  ann_dl_idx = ann.get('dl_idx', None)
                  if ann_dl_idx is None:
                      if 'images' in ann_data and isinstance(ann_data['images'], list) and len(ann_data['images']) > 0:
                          ann_dl_idx = ann_data['images'][0].get('dl_idx', "0")
                      else:
                          ann_dl_idx = "0"
                  ann_dl_idx_str = str(ann_dl_idx) if ann_dl_idx is not None else "0"
                  item = AnnotationItem(
                    image_file=img_rel_path,
                    annotation_file=ann_rel_path,
                    bbox=bbox,
                    dl_name=dl_name,
                    category_id=category_id,
                    dl_idx=ann_dl_idx_str
                  )
                  items.append(item)
          else:
            dl_name = ann_data.get('dl_name', '')
            category_id = -1
            dl_idx = "0"
            bbox = []
            if 'annotations' in ann_data and isinstance(ann_data['annotations'], list) and ann_data['annotations']:
              ann0 = ann_data['annotations'][0]
              bbox = ann0.get('bbox', [])
              category_id = ann0.get('category_id', -1)
              ann_dl_idx = ann0.get('dl_idx', None)
              if ann_dl_idx is None:
                  if 'images' in ann_data and isinstance(ann_data['images'], list) and len(ann_data['images']) > 0:
                      ann_dl_idx = ann_data['images'][0].get('dl_idx', "0")
                  else:
                      ann_dl_idx = "0"
              dl_idx = str(ann_dl_idx) if ann_dl_idx is not None else "0"
            item = AnnotationItem(
              image_file=img_rel_path,
              annotation_file=ann_rel_path,
              bbox=bbox,
              dl_name=dl_name,
              category_id=category_id,
              dl_idx=dl_idx
            )
            items.append(item)
        except Exception as e:
          print(f"[load_anotation] Error reading {ann_rel_path}: {e}")
          pass
    print(f"Loaded {len(items)} annotation items.")
    return items
    
# AnnotationItem 리스트를 이미지 파일별로 딕셔너리로 변환
def change_Annotation2dict(annotation_items):
    annotation_dict = {}
    for item in annotation_items:
        if item.image_file not in annotation_dict:
            annotation_dict[item.image_file] = []
        annotation_dict[item.image_file].append(item)
    return annotation_dict

def save_anntation_items_csv(annotation_items, csv_file):
    
    # 항상 덮어쓰기 모드('w')로 동작합니다. 기존 파일이 있으면 새로 덮어씁니다.
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_file', 'annotation_file', 'bbox', 'dl_name', 'category_id', 'dl_idx'])
            for item in annotation_items:
                writer.writerow([item.image_file, item.annotation_file, item.bbox, item.dl_name, item.category_id, item.dl_idx])
        OpLog(f"Annotation items saved to CSV (overwrite): {csv_file}")
    except Exception as e:
        OpLog(f"Error saving annotation items to CSV {csv_file}: {e}")

def read_anntation_items_csv(csv_file):
    annotation_items = []
    try:
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                bbox = eval(row['bbox']) if row['bbox'] else []
                item = AnnotationItem(
                    image_file=row['image_file'],
                    annotation_file=row['annotation_file'],
                    bbox=bbox,
                    dl_name=row['dl_name'],
                    category_id=int(row['category_id']),
                    dl_idx=row['dl_idx']
                )
                annotation_items.append(item)
        OpLog(f"Annotation items loaded from CSV: {csv_file}")
    except Exception as e:
        OpLog(f"Error loading annotation items from CSV {csv_file}: {e}")
    return annotation_items

def Test():

    annotation_dir = "./data/eda/hap/train_annotations"
    image_dir = "./data/eda/hap/train_images"
    csv_file = "./data/eda/hap/annotation_items.csv"
    
   # annotation_items = load_anotation(image_dir, annotation_dir)
    
   # save_anntation_items_csv(annotation_items, csv_file)
    annotation_items = read_anntation_items_csv(csv_file)
    
    annotation_dict = change_Annotation2dict(annotation_items)
    nCount = 0
    nDiffCount = 0
    for item in annotation_items:
        if item.category_id == 1:
            nCount += 1
        if item.dl_idx !=  str(item.category_id) :
            nDiffCount += 1
    print(f"category_id ==1 인 항목 수: {nCount}")
    print(f"dl_idx와 category_id가 다른 항목 수: {nDiffCount}")

    dl_idx_bbox_count = 0
    files_to_remove = []
    for image_file, items in annotation_dict.items():
        # (dl_idx, bbox) 쌍의 등장 횟수 세기 및 파일 삭제 대상 수집
        key_to_items = {}
        for item in items:
            key = (item.dl_idx, tuple(item.bbox))
            if key not in key_to_items:
                key_to_items[key] = []
            key_to_items[key].append(item)
        for key, item_list in key_to_items.items():
            if len(item_list) > 1:
                # 2번째(이상) annotation_file만 삭제 대상으로 추가
                for dup_item in item_list[1:]:
                    files_to_remove.append(dup_item.annotation_file)
                    dl_idx_bbox_count += 1
    print(f"동일한 dl_idx와 bbox를 가진 항목 수: {dl_idx_bbox_count}")

    # 파일 삭제 및 빈 디렉토리 삭제
    for annotation_file in files_to_remove:
        abs_path = os.path.join(annotation_dir, annotation_file)
        try:
            if os.path.exists(abs_path):
                os.remove(abs_path)
                print(f"삭제: {abs_path}")
                # 상위 디렉토리가 비었으면 삭제
                dir_path = os.path.dirname(abs_path)
                if os.path.isdir(dir_path) and len(os.listdir(dir_path)) == 0:
                    os.rmdir(dir_path)
                    print(f"빈 디렉토리 삭제: {dir_path}")
        except Exception as e:
            print(f"파일/디렉토리 삭제 오류: {abs_path}, {e}")

if __name__ == "__main__":
    Test()


