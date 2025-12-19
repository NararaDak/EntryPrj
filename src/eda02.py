
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


@dataclass
class SubmissionItem:
    image_id : int
    category_id : int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    score: float
    dl_name : str

@dataclass
class Image_SubmissionItem:
    SubmissionItem : list

# CSV 파일에서 SubmissionItem 리스트 읽기
def read_submission_csv(file_path,dl_mapping):
    def get_dl_name(category_id):
        if dl_mapping and str(category_id) in dl_mapping:
            return dl_mapping[str(category_id)]
        return "Unknown"

    submission_items = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                item = SubmissionItem(
                    image_id=int(row['image_id']),
                    category_id=int(row['category_id']),
                    bbox_x=int(row['bbox_x']),
                    bbox_y=int(row['bbox_y']),
                    bbox_w=int(row['bbox_w']),
                    bbox_h=int(row['bbox_h']),
                    score=float(row['score']),
                    dl_name=get_dl_name(int(row['category_id']))
                )
                submission_items.append(item)
        OpLog(f"Submission CSV 읽기 완료: {file_path} ({len(submission_items)}개 항목)")
    except Exception as e:
        OpLog(f"Submission CSV 읽기 오류 {file_path}: {e}")
    return submission_items
  
# SubmissionItem 리스트를 이미지 ID별로 딕셔너리로 변환
def change_submission2dict(submission_items):
    submission_dict = {}
    for item in submission_items:
        if item.image_id not in submission_dict:
            submission_dict[item.image_id] = []
        submission_dict[item.image_id].append(item)
    return submission_dict

# 이미지에 바운딩 박스 그리기
def loop_submissionDict( submission_dict, image_path,save_path,score_bound =0.75):
    for image_id, items in submission_dict.items():
        draw_submissionBox(image_id,items, image_path, save_path, score_bound)
# 이미지에 바운딩 박스 그리기
def draw_submissionBox(image_id,SubmissionItems, image_path, save_path, score_bound=0.75):
    bDraw = False
    for item in SubmissionItems:
        if item.score < score_bound:
            bDraw = True
            break
    # score 0.75 미만인 경우에만 박스 그리기
    if bDraw:
        draw_box_one_image(image_id,SubmissionItems, image_path, save_path)

def draw_box_one_image(image_id,SubmissionItems, image_path, save_path):
    try:
        img = Image.open(os.path.join(image_path, f"{image_id}.png")).convert("RGB")
        draw = ImageDraw.Draw(img)

        # 한글 폰트 경로 지정 (Windows 기준)
        import matplotlib.font_manager as fm
        font_path = fm.findfont('Malgun Gothic', fallback_to_default=True)
        font = ImageFont.truetype(font_path, 18)
        for item in SubmissionItems:
            x, y, w, h = item.bbox_x, item.bbox_y, item.bbox_w, item.bbox_h
            draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
            # category_id와 dl_name을 박스 위에 표시
            label = f"Cat:{item.category_id}-{item.dl_name}"
            draw.text((x, y - 18), label, fill="red", font=font)
            # score는 박스 아래에 별도로 표시
            score_label = f"Score: {item.score:.2f}"
            draw.text((x, y + h + 2), score_label, fill="blue", font=font)
        makedirs(save_path)
        img.save(os.path.join(save_path, f"{image_id}_boxed.png"))
        OpLog(f"이미지 박스 그리기 완료: {image_id}.png")
    except Exception as e:
        OpLog(f"이미지 박스 그리기 오류 {image_id}.png: {e}")


def make_dl_idx2dlname_mapping(annotation_dir):
    """
    어노테이션 디렉토리(annotation_dir) 하위의 모든 JSON 파일을 읽어
    dl_idx와 dl_name의 매핑 딕셔너리를 생성하여 반환
    Returns: {dl_idx: dl_name, ...}
    """
    # COCO 포맷의 JSON 파일에서 dl_idx/dl_name 또는 categories(id-name) 매핑 생성
    dl_mapping = {}
    if os.path.exists(annotation_dir):
        for subdir in os.listdir(annotation_dir):
            subdir_path = os.path.join(annotation_dir, subdir)
            if os.path.isdir(subdir_path):
                for class_dir in os.listdir(subdir_path):
                    class_dir_path = os.path.join(subdir_path, class_dir)
                    if os.path.isdir(class_dir_path):
                        for json_file in os.listdir(class_dir_path):
                            if json_file.endswith('.json'):
                                json_path = os.path.join(class_dir_path, json_file)
                                try:
                                    with open(json_path, 'r', encoding='utf-8') as f:
                                        data = json.load(f)
                                    # 기존 방식: images 필드에서 dl_idx, dl_name 추출
                                    if 'images' in data:
                                        for img_info in data['images']:
                                            dl_idx = img_info.get('dl_idx', None)
                                            dl_name = img_info.get('dl_name', None)
                                            if dl_idx is not None and dl_name is not None:
                                                dl_mapping[dl_idx] = dl_name
                                    # COCO categories 지원: categories 필드에서 id, name 추출
                                    elif 'categories' in data:
                                        for cat in data['categories']:
                                            cat_id = cat.get('id', None)
                                            cat_name = cat.get('name', None)
                                            if cat_id is not None and cat_name is not None:
                                                dl_mapping[cat_id] = cat_name
                                except Exception as e:
                                    OpLog(f"JSON 파일 읽기 오류 {json_path}: {e}")
    return dl_mapping

def get_submission_item_by_imageid(submission_items, image_id):
    return [item for item in submission_items if item.image_id == image_id]

def Save_Submission_box():
    submission_file = "./data/eda/submission20251212112431_ckeck.csv"
    image_dir = "./data/oraldrug/test_images"
    save_path = "./data/eda/submission0.75"
    anntation_dir = "./data/eda/hap/train_annotations"
    #
    dl_mapping = make_dl_idx2dlname_mapping(anntation_dir)
    submission_items = read_submission_csv(submission_file, dl_mapping)

    submission_dict = change_submission2dict(submission_items)
    loop_submissionDict(submission_dict, image_dir, save_path, score_bound=0.75)
   # 409, 107
    items = get_submission_item_by_imageid(submission_items, 409)
    draw_box_one_image(409,items, image_dir, save_path)
    
    items = get_submission_item_by_imageid(submission_items, 1076 )
    draw_box_one_image(1076, items, image_dir, save_path)
        

if __name__ == "__main__":
    Save_Submission_box()

