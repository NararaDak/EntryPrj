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

from dataclasses import dataclass


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
        if dl_mapping and category_id in dl_mapping:
            return dl_mapping[category_id]
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
    try:
        img = Image.open(os.path.join(image_path, f"{image_id}.png")).convert("RGB")
        draw = ImageDraw.Draw(img)
        bDraw = False
        for item in SubmissionItems:
            if item.score < score_bound:
                bDraw = True
                break
        # score 0.75 미만인 항목이 없으면 그리지 않음
        if not bDraw:
            return 
        
        for item in SubmissionItems:
            x, y, w, h = item.bbox_x, item.bbox_y, item.bbox_w, item.bbox_h
            draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
            # category_id와 dl_name을 함께 표시
            label = f"Cat:{item.category_id} {item.dl_name} ({item.score:.2f})"
            draw.text((x, y - 18), label, fill="red")
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

def Save_Submission_box():
    submission_file = "./data/eda/submission20251212112431_ckeck.csv"
    image_dir = "./data/oraldrug/test_images"
    save_path = "./data/eda/submission0.75"
    anntation_dir = "./data/eda/hap/train_annotations"
    dl_mapping = make_dl_idx2dlname_mapping(anntation_dir)

    submission_items = read_submission_csv(submission_file, dl_mapping)
    submission_dict = change_submission2dict(submission_items)
    loop_submissionDict(submission_dict, image_dir, save_path, score_bound=0.75)


class DatasetInfo:
    """데이터셋 디렉토리 및 파일 경로 관리 클래스"""
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.eda_dir = os.path.join(base_dir, "eda")
        makedirs(self.eda_dir)
        # org_data
        self.org_data_imges_dir = os.path.join(base_dir, "org_data", "train_images")
        self.org_data_annotations_dir = os.path.join(base_dir, "org_data", "train_annotations")
        # okImage_okAnno
        self.okImage_okAnno_imges_dir = os.path.join(base_dir, "okImage_okAnno", "train_images")
        self.okImage_okAnno_annotations_dir = os.path.join(base_dir, "okImage_okAnno", "train_annotations")
        # noImage_okAnno
        self.noImage_okAnno_imges_dir = os.path.join(base_dir, "noImage_okAnno", "train_images")
        self.noImage_okAnno_annotations_dir = os.path.join(base_dir, "noImage_okAnno", "train_annotations")
        # okImage_noAnno
        self.okImage_noAnno_imges_dir = os.path.join(base_dir, "okImage_noAnno", "train_images")
        self.okImage_noAnno_annotations_dir = os.path.join(base_dir, "okImage_noAnno", "train_annotations")
        # eda csv
        self.image2Json = os.path.join(self.eda_dir, "img_to_json.csv")
        self.json2Image = os.path.join(self.eda_dir, "json_to_img.csv")
        self.duplication__result = os.path.join(self.eda_dir, "duplicate_images.csv")
        self.addJson_result = os.path.join(self.eda_dir, "addJason.csv")
        
       
        #duplicate_images.csv
        # 디렉토리 생성
        makedirs(self.okImage_okAnno_imges_dir)
        makedirs(self.okImage_okAnno_annotations_dir)
        makedirs(self.noImage_okAnno_imges_dir)
        makedirs(self.noImage_okAnno_annotations_dir)
        makedirs(self.okImage_noAnno_imges_dir)
        makedirs(self.okImage_noAnno_annotations_dir)
        self.AI_HUB_IMAGE_DIR = r"D:\temp\AI_HUB\01.데이터\1.Training\원천데이터\경구약제조합 5000종"
        self.AI_HUB_ANNOTATION_DIR = r"D:\temp\AI_HUB\01.데이터\1.Training\라벨링데이터\경구약제조합 5000종"

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

    def  get_duplication_result(self):
        return self.duplication__result

    def get_addJson_result(self):
        add_json_result_path = os.path.join(self.base_dir, "eda", "add_json_result.txt")
        makedirs(os.path.dirname(add_json_result_path))
        return add_json_result_path

    def get_all(self):
        return {
            'org_data': (self.org_data_imges_dir, self.org_data_annotations_dir),
            'okImage_okAnno': (self.okImage_okAnno_imges_dir, self.okImage_okAnno_annotations_dir),
            'noImage_okAnno': (self.noImage_okAnno_imges_dir, self.noImage_okAnno_annotations_dir),
            'okImage_noAnno': (self.okImage_noAnno_imges_dir, self.okImage_noAnno_annotations_dir),
            'eda_csv': (self.image2Json, self.json2Image)
        }



def GetAllAnnotation(anno_dir):
    """
    어노테이션 디렉토리(anno_dir) 하위의 모든 JSON 파일 경로와 해당 JSON이 포함하는 이미지 파일명을 리스트로 반환
    Returns: [(json_path, [img_filename, ...]), ...]
    """
    json_files = []
    if os.path.exists(anno_dir):
        for subdir in os.listdir(anno_dir):
            subdir_path = os.path.join(anno_dir, subdir)
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
                                    images_in_json = []
                                    if 'images' in data:
                                        for img_info in data['images']:
                                            img_filename = img_info.get('file_name', '')
                                            if img_filename:
                                                images_in_json.append(img_filename)
                                    json_files.append((json_path, images_in_json))
                                except Exception as e:
                                    OpLog(f"JSON 파일 읽기 오류 {json_path}: {e}")
    return json_files

# ════════════════════════════════════════
# ▣ 01. 이미지-JSON 매핑 분석 함수 
# ════════════════════════════════════════
def analyze_image_json_mapping(dataset_info):
    """
    이미지와 JSON 파일 간의 매핑 관계를 분석하여 CSV 파일로 저장
    
    생성 파일:
        - image2Json: 이미지 -> JSON 매핑 (이미지명, JSON경로)
        -  json2Image: JSON -> 이미지 매핑 (JSON경로, 이미지명)
    
    Returns:
        tuple: (img_to_json_mapping, json_to_img_mapping)
    """
    OpLog("이미지-JSON 매핑 분석 시작")
    train_img_dir, train_annotation_dir = dataset_info.get_org_data()
    image2Json, json2Image = dataset_info.get_eda_csv()
    # 1. train_imge_dir 밑의 모든 이미지 파일 수집
    image_files = {}  # {filename: full_path}
    if os.path.exists(train_img_dir):
        for img_file in os.listdir(train_img_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_files[img_file] = os.path.join(train_img_dir, img_file)
    
    OpLog(f"이미지 파일 수: {len(image_files)}")
    
    # 2. train_annotation_dir 밑의 모든 JSON 파일 수집
    json_files = GetAllAnnotation(train_annotation_dir)
    
    OpLog(f"JSON 파일 수: {len(json_files)}")
    
    # 3. image2Json 매핑 생성
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
    
    # 4.  json2Image 매핑 생성
    json_to_img_mapping = []  # [(json_path, img_name), ...]
    
    for json_path, images_in_json in json_files:
        if images_in_json:
            for img_name in images_in_json:
                # 실제 이미지 파일이 존재하는지 확인
                if img_name in image_files:
                    json_to_img_mapping.append((json_path, img_name))
                else:
                    # 이미지가 없을 때 JSON에서 dl_name 추출
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        dl_name = ""
                        if 'images' in data and len(data['images']) > 0:
                            dl_name = data['images'][0].get('dl_name', '')
                        
                        if dl_name:
                            json_to_img_mapping.append((json_path, f"NONE({dl_name})"))
                        else:
                            json_to_img_mapping.append((json_path, "NONE"))
                    except Exception as e:
                        json_to_img_mapping.append((json_path, "NONE"))
        else:
            # 이미지 정보가 없는 JSON - dl_name 추출
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                dl_name = ""
                if 'images' in data and len(data['images']) > 0:
                    dl_name = data['images'][0].get('dl_name', '')
                
                if dl_name:
                    json_to_img_mapping.append((json_path, f"NONE({dl_name})"))
                else:
                    json_to_img_mapping.append((json_path, "NONE"))
            except Exception as e:
                json_to_img_mapping.append((json_path, "NONE"))
    
    # 5. image2Json CSV 파일 저장
    makedirs(os.path.dirname(image2Json))
    with open(image2Json, 'w', encoding='utf-8') as f:
        f.write("Image,JSON\n")
        for img_name, json_path in img_to_json_mapping:
            f.write(f"{img_name},{json_path}\n")
    
    OpLog(f"image2Json 저장 완료: {image2Json} ({len(img_to_json_mapping)}개 매핑)")
    
    # 6.  json2Image CSV 파일 저장
    with open( json2Image, 'w', encoding='utf-8') as f:
        f.write("JSON,Image\n")
        for json_path, img_name in json_to_img_mapping:
            f.write(f"{json_path},{img_name}\n")
    
    OpLog(f" json2Image 저장 완료: { json2Image} ({len(json_to_img_mapping)}개 매핑)")
    
    # 7. 통계 정보 출력
    img_without_json = sum(1 for _, json_path in img_to_json_mapping if json_path == "NONE")
    json_without_img = sum(1 for _, img_name in json_to_img_mapping if img_name.startswith("NONE"))
    
    OpLog(f"매핑 분석 완료:")
    OpLog(f"전체 이미지: {len(image_files)}개")
    OpLog(f"전체 JSON: {len(json_files)}개")
    OpLog(f"JSON 없는 이미지: {img_without_json}개")
    OpLog(f"이미지 없는 JSON: {json_without_img}개")
    return img_to_json_mapping, json_to_img_mapping


# ════════════════════════════════════════
# ▣ 02. 이미지-JSON 매핑 통계 시각화 함수 
# ════════════════════════════════════════
def visualize_mapping_statistics(dataset_info, img_to_json_mapping, json_to_img_mapping):
    """
    이미지-JSON 매핑 통계를 시각화
    
    Args:
        dataset_info: DatasetInfo 객체
        img_to_json_mapping: analyze_image_json_mapping()의 첫 번째 반환값
        json_to_img_mapping: analyze_image_json_mapping()의 두 번째 반환값
    """
    OpLog("매핑 통계 시각화 시작")
    # 1. 데이터 분석
    # 이미지당 JSON 개수 분석
    img_json_count = Counter()
    for img_name, json_path in img_to_json_mapping:
        if json_path == "NONE":
            img_json_count[img_name] = 0
        else:
            img_json_count[img_name] += 1
    
    # JSON당 이미지 개수 분석
    json_img_count = Counter()
    for json_path, img_name in json_to_img_mapping:
        if img_name.startswith("NONE"):
            json_img_count[json_path] = 0
        else:
            json_img_count[json_path] += 1
    
    # 통계 계산
    json_per_img_distribution = Counter(img_json_count.values())
    img_per_json_distribution = Counter(json_img_count.values())
    
    total_images = len(img_json_count)
    total_jsons = len(json_img_count)
    images_with_no_json = sum(1 for count in img_json_count.values() if count == 0)
    jsons_with_no_img = sum(1 for count in json_img_count.values() if count == 0)
    images_with_json = total_images - images_with_no_json
    jsons_with_img = total_jsons - jsons_with_no_img
    
    # 2. 시각화
    fig = plt.figure(figsize=(16, 12))
    
    # 2-1. 전체 통계 요약
    ax1 = plt.subplot(3, 3, 1)
    summary_text = f"""▣전체 통계 요약
    
* 총 이미지: {total_images:,}개
* JSON 있는 이미지: {images_with_json:,}개
* JSON 없는 이미지: {images_with_no_json:,}개

총 JSON: {total_jsons:,}개
이미지 있는 JSON: {jsons_with_img:,}개
이미지 없는 JSON: {jsons_with_no_img:,}개

총 매핑 수: {len(img_to_json_mapping):,}개"""
    
    ax1.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center')
    ax1.axis('off')
    ax1.set_title('▶ 전체 통계', fontsize=14, fontweight='bold', pad=20)
    
    # 2-2. 이미지-JSON 매핑 비율 (파이 차트)
    ax2 = plt.subplot(3, 3, 2)
    labels = ['JSON 있음', 'JSON 없음']
    sizes = [images_with_json, images_with_no_json]
    colors = ['#4CAF50', '#f44336']
    explode = (0.05, 0)
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10})
    ax2.set_title('▶ 이미지별 JSON 매핑 비율', fontsize=12, fontweight='bold', pad=10)
    
    # 2-3. JSON-이미지 매핑 비율 (파이 차트)
    ax3 = plt.subplot(3, 3, 3)
    labels = ['이미지 있음', '이미지 없음']
    sizes = [jsons_with_img, jsons_with_no_img]
    colors = ['#2196F3', '#FF9800']
    explode = (0.05, 0)
    
    ax3.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10})
    ax3.set_title('▶ JSON별 이미지 매핑 비율', fontsize=12, fontweight='bold', pad=10)
    
    # 2-4. 이미지당 JSON 개수 분포 (막대 그래프)
    ax4 = plt.subplot(3, 3, 4)
    counts = sorted(json_per_img_distribution.keys())
    values = [json_per_img_distribution[c] for c in counts]
    bars = ax4.bar(counts, values, color='#4CAF50', alpha=0.7, edgecolor='black')
    
    # 막대 내부에 값 표시 (높이가 충분한 경우만)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        value_text = f'{int(height):,}'
        
        # 막대 높이가 50 이상이면 내부에, 아니면 위에 표시
        if height > 50:
            # 막대 내부 (가운데)
            y_pos = height / 2
            ax4.text(bar.get_x() + bar.get_width()/2., y_pos,
                    value_text, ha='center', va='center', 
                    fontsize=9, fontweight='bold', color='white')
        else:
            # 막대 위
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    value_text, ha='center', va='bottom', fontsize=8)
    
    ax4.set_xlabel('* 이미지당 JSON 개수', fontsize=10)
    ax4.set_ylabel('* 이미지 수', fontsize=10)
    ax4.set_title('▶ 이미지당 JSON 개수 분포', fontsize=12, fontweight='bold', pad=10)
    ax4.grid(axis='y', alpha=0.3)
    
    # 2-5. JSON당 이미지 개수 분포 (막대 그래프)
    ax5 = plt.subplot(3, 3, 5)
    counts = sorted(img_per_json_distribution.keys())
    values = [img_per_json_distribution[c] for c in counts]
    bars = ax5.bar(counts, values, color='#2196F3', alpha=0.7, edgecolor='black')
    
    # 막대 내부에 값 표시 (높이가 충분한 경우만)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        value_text = f'{int(height):,}'
        
        # 막대 높이가 50 이상이면 내부에, 아니면 위에 표시
        if height > 50:
            # 막대 내부 (가운데)
            y_pos = height / 2
            ax5.text(bar.get_x() + bar.get_width()/2., y_pos,
                    value_text, ha='center', va='center', 
                    fontsize=9, fontweight='bold', color='white')
        else:
            # 막대 위
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    value_text, ha='center', va='bottom', fontsize=8)
    
    ax5.set_xlabel('* JSON당 이미지 개수', fontsize=10)
    ax5.set_ylabel('* JSON 수', fontsize=10)
    ax5.set_title('▶ JSON당 이미지 개수 분포', fontsize=12, fontweight='bold', pad=10)
    ax5.grid(axis='y', alpha=0.3)
    
    # 2-6. 이미지당 JSON 개수 상세 통계
    ax6 = plt.subplot(3, 3, 6)
    img_json_counts = list(img_json_count.values())
    stats_text = f"""▣ 이미지당 JSON 개수 통계

* 최소: {min(img_json_counts)}개
* 최대: {max(img_json_counts)}개
* 평균: {sum(img_json_counts)/len(img_json_counts):.2f}개
* 중앙값: {sorted(img_json_counts)[len(img_json_counts)//2]}개

분포:"""
    
    for count in sorted(json_per_img_distribution.keys())[:10]:  # 상위 10개만 표시
        percentage = (json_per_img_distribution[count] / total_images) * 100
        stats_text += f"\n  {count}개: {json_per_img_distribution[count]:,}개 ({percentage:.1f}%)"
    
    ax6.text(0.1, 0.5, stats_text, fontsize=9, verticalalignment='center')
    ax6.axis('off')
    ax6.set_title('▶ 이미지당 JSON 통계', fontsize=12, fontweight='bold', pad=10)
    
    # 2-7. JSON당 이미지 개수 상세 통계
    ax7 = plt.subplot(3, 3, 7)
    json_img_counts = list(json_img_count.values())
    stats_text = f"""▣ JSON당 이미지 개수 통계

* 최소: {min(json_img_counts)}개
* 최대: {max(json_img_counts)}개
* 평균: {sum(json_img_counts)/len(json_img_counts):.2f}개
* 중앙값: {sorted(json_img_counts)[len(json_img_counts)//2]}개

분포 (상위 10개):"""
    
    for count in sorted(img_per_json_distribution.keys())[:10]:  # 상위 10개만 표시
        percentage = (img_per_json_distribution[count] / total_jsons) * 100
        stats_text += f"\n  {count}개: {img_per_json_distribution[count]:,}개 ({percentage:.1f}%)"
    
    ax7.text(0.05, 0.5, stats_text, fontsize=8, verticalalignment='center')
    ax7.axis('off')
    ax7.set_title('▶ JSON당 이미지 통계', fontsize=12, fontweight='bold', pad=10)
    
    plt.suptitle('▦ 이미지-JSON 매핑 통계 분석', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    
    # 저장
    save_path = dataset_info.get_visualize_path("mapping_statistics.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    OpLog(f"통계 그래프 저장: {save_path}")
    plt.show()
    plt.close()


# 이미지 비교 방법: 체크섬 vs 픽셀 단위

def calculate_image_checksum(image_path):
    """
    이미지 파일의 체크섬(MD5 해시) 계산
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        str: MD5 해시 값
    """
    md5_hash = hashlib.md5()
    try:
        with open(image_path, 'rb') as f:
            # 파일을 청크 단위로 읽어서 해시 계산 (메모리 효율적)
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except Exception as e:
        OpLog(f"체크섬 계산 오류 {image_path}: {e}")
        return None

# 픽셀 단위 비교 함수
def compare_images_pixel(img1_path, img2_path):
    """
    두 이미지를 픽셀 단위로 비교
    
    Args:
        img1_path: 첫 번째 이미지 경로
        img2_path: 두 번째 이미지 경로
        
    Returns:
        bool: 이미지가 동일하면 True, 아니면 False
    """
    try:
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        # 크기가 다르면 다른 이미지
        if img1.size != img2.size:
            return False
        
        # numpy 배열로 변환하여 비교
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        
        # 배열이 완전히 동일한지 확인
        return np.array_equal(arr1, arr2)
    except Exception as e:
        OpLog(f"이미지 비교 오류 {img1_path} vs {img2_path}: {e}")
        return False

def find_duplicate_images(dataset_info, method='checksum'):
    """
    JSON 파일이 없는 이미지를 찾아서 AI_HUB_500_DIR에 중복이 있는지 검사
    
    Args:
        method: 비교 방법 ('checksum' 또는 'pixel')
            - 'checksum': MD5 해시 기반 빠른 비교
            - 'pixel': 픽셀 단위 정확한 비교 (느림)
    
    Returns:
        list: [(train_img, aihub_img, match_method), ...] 중복 이미지 리스트
    """
    OpLog(f"중복 이미지 검사 시작 (방법: {method})")
    
    # 1. JSON 파일이 없는 이미지 찾기
    OpLog("JSON 없는 이미지 검색 중...")
    
    # TRAIN_IMG_DIR의 모든 이미지 수집
    train_imge_dir,train_annotation_dir = dataset_info.get_org_data()
    train_images = {}
    if os.path.exists(train_imge_dir):
        for img_file in os.listdir(train_imge_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                train_images[img_file] = os.path.join(train_imge_dir, img_file)
    
    # ANNOTATION_DIR의 JSON 파일에 있는 이미지 수집
    images_with_json = set()
    if os.path.exists(train_annotation_dir):
        for subdir in os.listdir(train_annotation_dir):
            subdir_path = os.path.join(train_annotation_dir, subdir)
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
                                    if 'images' in data:
                                        for img_info in data['images']:
                                            img_filename = img_info.get('file_name', '')
                                            if img_filename:
                                                images_with_json.add(img_filename)
                                except Exception as e:
                                    pass
    
    # JSON 없는 이미지 목록
    images_without_json = {name: path for name, path in train_images.items() 
                          if name not in images_with_json}
    
    OpLog(f"JSON 없는 이미지: {len(images_without_json)}개")
    
    if len(images_without_json) == 0:
        OpLog("JSON 없는 이미지가 없습니다.")
        return []
    dataset_info.GetAiHubDir()
    AI_HUB_500_DIR, _ = dataset_info.GetAiHubDir()
    DUPLICATE_RESULT = dataset_info.get_duplication_result()
    # 2. AI_HUB_500_DIR의 이미지 수집 (서브 디렉토리 포함)
    if not os.path.exists(AI_HUB_500_DIR):
        OpLog(f"AI Hub 디렉토리가 존재하지 않습니다: {AI_HUB_500_DIR}")
        return []
    
    aihub_images = {}
    # os.walk()를 사용하여 모든 서브 디렉토리 탐색
    for root, dirs, files in os.walk(AI_HUB_500_DIR):
        for img_file in files:
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                full_path = os.path.join(root, img_file)
                # 파일명이 중복될 수 있으므로 상대 경로를 키로 사용
                rel_path = os.path.relpath(full_path, AI_HUB_500_DIR)
                aihub_images[rel_path] = full_path
    
    OpLog(f"AI Hub 이미지: {len(aihub_images)}개 (서브 디렉토리 포함)")
    
    # 3. 중복 검사
    duplicates = []
    
    if method == 'checksum':
        OpLog("체크섬 기반 중복 검사 중...")
        
        # AI Hub 이미지의 체크섬 미리 계산
        aihub_checksums = {}
        for idx, (aihub_name, aihub_path) in enumerate(aihub_images.items(), 1):
            if idx % 100 == 0:
                OpLog(f"AI Hub 체크섬 계산 중: {idx}/{len(aihub_images)}")
            checksum = calculate_image_checksum(aihub_path)
            if checksum:
                if checksum not in aihub_checksums:
                    aihub_checksums[checksum] = []
                aihub_checksums[checksum].append((aihub_name, aihub_path))
        
        # Train 이미지와 비교
        for idx, (train_name, train_path) in enumerate(images_without_json.items(), 1):
            if idx % 10 == 0:
                OpLog(f"중복 검사 중: {idx}/{len(images_without_json)}")
            
            train_checksum = calculate_image_checksum(train_path)
            if train_checksum and train_checksum in aihub_checksums:
                for aihub_name, aihub_path in aihub_checksums[train_checksum]:
                    duplicates.append((train_name, aihub_name, 'checksum'))
                    OpLog(f"중복 발견 (체크섬): {train_name} == {aihub_name}")
    
    elif method == 'pixel':
        OpLog("픽셀 기반 중복 검사 중...")
        
        total_comparisons = len(images_without_json) * len(aihub_images)
        current = 0
        
        for train_name, train_path in images_without_json.items():
            for aihub_name, aihub_path in aihub_images.items():
                current += 1
                if current % 1000 == 0:
                    OpLog(f"픽셀 비교 중: {current}/{total_comparisons} ({current*100/total_comparisons:.1f}%)")
                
                if compare_images_pixel(train_path, aihub_path):
                    duplicates.append((train_name, aihub_name, 'pixel'))
                    OpLog(f"중복 발견 (픽셀): {train_name} == {aihub_name}")
    
    else:
        OpLog(f"알 수 없는 비교 방법: {method}")
        return []
    
    # 4. 결과 저장
    makedirs(os.path.dirname(DUPLICATE_RESULT))
    try:
        with open(DUPLICATE_RESULT, 'w', encoding='utf-8') as f:
            f.write("Train_Image,AIHub_Image,Match_Method\n")
            for train_img, aihub_img, match_method in duplicates:
                f.write(f"{train_img},{aihub_img},{match_method}\n")
        OpLog(f"결과 저장: {DUPLICATE_RESULT}")
    except PermissionError:
        OpLog(f"CSV 파일 쓰기 권한 오류: {DUPLICATE_RESULT}")
        OpLog("파일이 다른 프로그램에서 열려있는지 확인하세요.")
    except Exception as e:
        OpLog(f"결과 저장 오류: {e}")
    
    OpLog(f"중복 이미지 검사 완료: {len(duplicates)}개 발견")
    
    return duplicates

def copy_missing_annoations(duplicates,dataset_info):
    """
    중복 이미지와 관련 annotation을 복사
    
    Args:
        duplicates: find_duplicate_images()의 반환값 [(train_img, aihub_img, method), ...]
    """
    import shutil
    
    OpLog(f"누락 데이터 복사 시작: {len(duplicates)}개")
    
    # 디렉토리 생성
    
    train_imge_dir, train_annotation_dir = dataset_info.get_org_data()

    MISSING_TRAIN_IMAGES,MISSING_ANNOTATIONS = dataset_info.get_okImage_noAnno()

    AI_HUB_LABELING_DIR,_= dataset_info.GetAiHubDir()

    dataset_info.get.get  .GetAiHubDir()
    aihub_img_path, aihub_anno_path = dataset_info.GetAiHubDir()

    copied_jsons = 0
    
    for train_img_name, aihub_img_rel_path, match_method in duplicates:
        # 1. 이미지 복사
        # TRAIN_IMG_DIR의 원본 이미지
        train_img_path = os.path.join(train_imge_dir, train_img_name)
        
        # AI_HUB의 이미지 (상대 경로를 전체 경로로 변환)

        aihub_img_path = os.path.join(aihub_img_path, aihub_img_rel_path)
        # 복사 대상 확인
        if os.path.exists(train_img_path):
            # TRAIN_IMG_DIR의 이미지를 그대로 이름으로 복사
            dest_train = os.path.join(MISSING_TRAIN_IMAGES, train_img_name)
            try:
                shutil.copy2(train_img_path, dest_train)
                OpLog(f"이미지 복사: {train_img_name}")
                copied_images += 1
            except Exception as e:
                OpLog(f"이미지 복사 오류 {train_img_name}: {e}")
        
        if os.path.exists(aihub_img_path):
            # AI_HUB의 이미지를 .AI_HUB 접미사로 복사
            base_name = os.path.splitext(train_img_name)[0]
            ext = os.path.splitext(train_img_name)[1]
            aihub_dest_name = f"{base_name}.AI_HUB{ext}"
            dest_aihub = os.path.join(MISSING_TRAIN_IMAGES, aihub_dest_name)
            try:
                shutil.copy2(aihub_img_path, dest_aihub)
                OpLog(f"AI Hub 이미지 복사: {aihub_dest_name}")
                copied_images += 1
            except Exception as e:
                OpLog(f"AI Hub 이미지 복사 오류 {aihub_img_rel_path}: {e}")
        
        # 2. JSON annotation 찾기 및 복사
        # A.png -> A.json을 AI_HUB_LABELING_DIR에서 찾기
        base_name = os.path.splitext(train_img_name)[0]
        json_name = f"{base_name}.json"
        
        # MISSING_ANNOTATIONS 밑에 이미지 이름의 디렉토리 생성
        img_anno_dir = os.path.join(MISSING_ANNOTATIONS, base_name)
        makedirs(img_anno_dir)
        
        # AI_HUB_LABELING_DIR에서 재귀적으로 JSON 파일 찾기
        found_jsons = []
        if os.path.exists(AI_HUB_LABELING_DIR):
            for root, dirs, files in os.walk(AI_HUB_LABELING_DIR):
                if json_name in files:
                    json_path = os.path.join(root, json_name)
                    found_jsons.append(json_path)
        
        # 찾은 JSON 파일들을 01, 02, 03... 디렉토리에 복사
        if found_jsons:
            OpLog(f"{train_img_name}에 대한 JSON {len(found_jsons)}개 발견")
            for idx, json_path in enumerate(found_jsons, 1):
                # 01, 02, 03 형식의 디렉토리 생성
                json_subdir = os.path.join(img_anno_dir, f"{idx:02d}")
                makedirs(json_subdir)
                
                # JSON 파일 복사
                dest_json = os.path.join(json_subdir, json_name)
                try:
                    shutil.copy2(json_path, dest_json)
                    OpLog(f"  JSON 복사: {json_name} -> {base_name}/{idx:02d}/")
                    copied_jsons += 1
                except Exception as e:
                    OpLog(f"  JSON 복사 오류 {json_path}: {e}")
        else:
            OpLog(f"{train_img_name}에 대한 JSON을 찾을 수 없음")
    
    OpLog(f"복사 완료: 이미지 {copied_images}개, JSON {copied_jsons}개")
    OpLog(f"이미지 저장 위치: {MISSING_TRAIN_IMAGES}")
    OpLog(f"Annotation 저장 위치: {MISSING_ANNOTATIONS}")
    
    return copied_images, copied_jsons


# ════════════════════════════════════════
# ▣ 03. 통계 시각화 실행 함수
# ════════════════════════════════════════
def MakeStatistic(dataset_info):
    img_to_json, json_to_img = analyze_image_json_mapping(dataset_info) # 매핑 분석
    visualize_mapping_statistics(dataset_info, img_to_json, json_to_img) # 시각화
    return img_to_json,json_to_img

def CopyOkImageOkJson(dataset_info, img_to_json_mapping):
    """
    이미지와 어노테이션이 모두 존재하는 경우만 타겟 디렉토리로 동일 구조로 복사
    - img_to_json_mapping: [(img_name, json_path), ...] (MakeStatistic에서 리턴받은 값)
    - org_images_dir: 원본 이미지 디렉토리
    - org_annos_dir: 원본 어노테이션 디렉토리
    - target_images_dir: 복사할 이미지 디렉토리
    - target_annos_dir: 복사할 어노테이션 디렉토리
    """
    org_images_dir, org_annos_dir =dataset_info.get_org_data()
    target_images_dir, target_annos_dir = dataset_info.get_okImage_okAnno()
    
    for img_name, json_path in img_to_json_mapping:
        if json_path == "NONE":
            continue  # 어노테이션 없는 이미지 skip
        # 원본 이미지 경로
        src_img_path = os.path.join(org_images_dir, img_name)
        # 어노테이션 상대경로 (org_annos_dir 하위)
        rel_json_path = os.path.relpath(json_path, org_annos_dir)
        src_json_path = json_path
        # 타겟 경로
        dst_img_path = os.path.join(target_images_dir, img_name)
        dst_json_path = os.path.join(target_annos_dir, rel_json_path)
        # 디렉토리 생성
        makedirs(os.path.dirname(dst_img_path))
        makedirs(os.path.dirname(dst_json_path))
        # 파일 복사
        try:
            if os.path.exists(src_img_path):
                shutil.copy2(src_img_path, dst_img_path)
            if os.path.exists(src_json_path):
                shutil.copy2(src_json_path, dst_json_path)
            OpLog(f"복사 완료: {img_name} / {rel_json_path}")
        except Exception as e:
            OpLog(f"복사 실패: {img_name} / {rel_json_path} - {e}")


# ════════════════════════════════════════
# ▣ 03-2. 어노테이션만 존재하는 경우 이미지 복사함수 
# ════════════════════════════════════════
def CopyImageWithOnlyAnno(dataset_info):
    """
    ANNOTATION_DIR에는 JSON 파일이 있지만 TRAIN_IMG_DIR에는 이미지가 없는 경우를 찾아서
    AI_HUB_500_DIR에서 이미지를 찾아 MISSING_ONLY_ANNOTAIIONS_IMG 복사하고 바운딩 박스 시각화
    AI_HUB_LABELING_DIR에서 추가 JSON 파일을 찾아 MISSING_ONLY_ANNOTATIONS_ADD에 복사
    """
    OpLog("JSON은 있지만 이미지가 없는 경우 검색 시작")

    train_imge_dir, train_annotation_dir = dataset_info.get_org_data()
    AI_HUB_500_DIR, AI_HUB_LABELING_DIR = dataset_info.GetAiHubDir()
     
    MISSING_ONLY_ANNOTAIIONS_IMG, MISSING_ONLY_ANNOTATIONS_ANNO = dataset_info.get_noImage_okAnno()
    add_json_result_path = dataset_info.get_addJson_result()
    # 1. TRAIN_IMG_DIR의 모든 이미지 파일 수집
    train_images = set()
    if os.path.exists(train_imge_dir):
        for img_file in os.listdir(train_imge_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                train_images.add(img_file)
    OpLog(f"Train 이미지: {len(train_images)}개")
    # 2. ANNOTATION_DIR의 모든 JSON 파일에서 이미지 정보 수집
    json_image_mapping = []  # [(json_path, img_filename, dl_name), ...]
    if os.path.exists(train_annotation_dir):
        for root, dirs, files in os.walk(train_annotation_dir):
            for json_file in files:
                if json_file.endswith('.json'):
                    json_path = os.path.join(root, json_file)
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if 'images' in data and len(data['images']) > 0:
                            img_info = data['images'][0]
                            img_filename = img_info.get('file_name', '')
                            dl_name = img_info.get('dl_name', 'Unknown')
                            if img_filename:
                                json_image_mapping.append((json_path, img_filename, dl_name))
                    except Exception as e:
                        OpLog(f"JSON 파일 읽기 오류 {json_path}: {e}")
    OpLog(f"JSON 파일: {len(json_image_mapping)}개")
    # 3. Train에 없는 이미지 찾기 (고유 이미지명 기준)
    unique_missing = {}
    for json_path, img_filename, dl_name in json_image_mapping:
        if img_filename not in train_images and img_filename not in unique_missing:
            unique_missing[img_filename] = (json_path, dl_name)
    OpLog(f"Train에 없는 고유 이미지: {len(unique_missing)}개")
    if len(unique_missing) == 0:
        OpLog("모든 JSON에 대응하는 이미지가 Train 디렉토리에 있습니다.")
        return []
    # 4. AI_HUB_500_DIR에서 해당 이미지 찾기
    if not os.path.exists(AI_HUB_500_DIR):
        OpLog(f"AI Hub 디렉토리가 존재하지 않습니다: {AI_HUB_500_DIR}")
        return []
    # AI Hub 이미지 수집 (파일명을 키로)
    aihub_images = {}  # {filename: [full_path, ...]}
    OpLog("AI Hub 이미지 수집 중...")
    for root, dirs, files in os.walk(AI_HUB_500_DIR):
        for img_file in files:
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                full_path = os.path.join(root, img_file)
                if img_file not in aihub_images:
                    aihub_images[img_file] = []
                aihub_images[img_file].append(full_path)
    OpLog(f"AI Hub 이미지: {len(aihub_images)}개 파일명")
    # 5. 디렉토리 생성
    makedirs(MISSING_ONLY_ANNOTAIIONS_IMG)
    makedirs(MISSING_ONLY_ANNOTATIONS_ANNO)
    # 6. 각 고유 missing 이미지 처리
    copied_count = 0
    not_found_count = 0
    copied_json_count = 0
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    makedirs(log_dir)
    add_json_csv_path = os.path.join(log_dir, 'noImage_okAnno_addJson.csv')
    add_json_csv_rows = []
    with open(add_json_result_path, 'w', encoding='utf-8') as add_json_f:
        for img_filename, (json_path, dl_name) in unique_missing.items():
            base_name = os.path.splitext(img_filename)[0]
            ext = os.path.splitext(img_filename)[1]
            json_filename = f"{base_name}.json"
            # AI Hub에서 이미지 찾기 (처음 발견되는 것만 복사)
            if img_filename in aihub_images and len(aihub_images[img_filename]) > 0:
                aihub_img_path = aihub_images[img_filename][0]
                dest_img_path = os.path.join(MISSING_ONLY_ANNOTAIIONS_IMG, img_filename)
                try:
                    shutil.copy2(aihub_img_path, dest_img_path)
                    copied_count += 1
                    OpLog(f"이미지 복사: {img_filename}")
                except Exception as e:
                    OpLog(f"이미지 복사 오류 {img_filename}: {e}")
            else:
                not_found_count += 1
                OpLog(f"AI Hub에서 찾을 수 없음: {img_filename} (dl_name: {dl_name})")
            # === 어노테이션도 구조 유지하여 복사 ===
            rel_json_path = os.path.relpath(json_path, train_annotation_dir)
            dest_json_path = os.path.join(MISSING_ONLY_ANNOTATIONS_ANNO, rel_json_path)
            makedirs(os.path.dirname(dest_json_path))
            try:
                shutil.copy2(json_path, dest_json_path)
                OpLog(f"어노테이션 복사: {rel_json_path}")
            except Exception as e:
                OpLog(f"어노테이션 복사 오류 {rel_json_path}: {e}")
            # === AI_HUB_LABELING_DIR에서 추가 JSON 복사 (01/02/03 등 디렉토리) ===
            if os.path.exists(AI_HUB_LABELING_DIR):
                found_aihub_jsons = []
                for root, dirs, files in os.walk(AI_HUB_LABELING_DIR):
                    if json_filename in files:
                        aihub_json_path = os.path.join(root, json_filename)
                        try:
                            with open(aihub_json_path, 'r', encoding='utf-8') as f:
                                aihub_data = json.load(f)
                            aihub_dl_name = 'Unknown'
                            if 'images' in aihub_data and len(aihub_data['images']) > 0:
                                aihub_dl_name = aihub_data['images'][0].get('dl_name', 'Unknown')
                            found_aihub_jsons.append((aihub_json_path, aihub_dl_name))
                        except Exception as e:
                            OpLog(f"AI Hub JSON 읽기 오류 {aihub_json_path}: {e}")
                additional_jsons = []
                for aihub_json_path, aihub_dl_name in found_aihub_jsons:
                    if aihub_dl_name != dl_name:
                        additional_jsons.append(aihub_json_path)
                if additional_jsons:
                    anno_add_dir = os.path.join(MISSING_ONLY_ANNOTATIONS_ANNO, base_name)
                    makedirs(anno_add_dir)
                    dir_01 = os.path.join(anno_add_dir, "01")
                    makedirs(dir_01)
                    dest_json_01 = os.path.join(dir_01, json_filename)
                    try:
                        shutil.copy2(json_path, dest_json_01)
                        copied_json_count += 1
                        OpLog(f"  원본 JSON 복사: {base_name}/01/{json_filename}")
                        add_json_f.write(dest_json_01 + "\n")
                    except Exception as e:
                        OpLog(f"  원본 JSON 복사 오류 {json_path}: {e}")
                    for idx, aihub_json_path in enumerate(additional_jsons, 2):
                        dir_num = os.path.join(anno_add_dir, f"{idx:02d}")
                        makedirs(dir_num)
                        dest_json = os.path.join(dir_num, json_filename)
                        try:
                            shutil.copy2(aihub_json_path, dest_json)
                            copied_json_count += 1
                            OpLog(f"  추가 JSON 복사: {base_name}/{idx:02d}/{json_filename}")
                            add_json_f.write(dest_json + "\n")
                            add_json_csv_rows.append([img_filename, base_name, json_filename, aihub_json_path, dest_json])
                        except Exception as e:
                            OpLog(f"  추가 JSON 복사 오류 {aihub_json_path}: {e}")
                    OpLog(f"JSON 복사 완료: {base_name} (원본 1개 + 추가 {len(additional_jsons)}개)")
    # 추가 JSON 복사 경로 csv 저장
    if add_json_csv_rows:
        import csv
        with open(add_json_csv_path, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['img_filename', 'base_name', 'json_filename', 'src_json_path', 'dest_json_path'])
            writer.writerows(add_json_csv_rows)
    OpLog(f"추가 JSON 복사 경로 저장: {add_json_csv_path}")
    OpLog(f"처리 완료:")
    OpLog(f"  - 이미지 복사: {copied_count}개")
    OpLog(f"  - AI Hub에서 찾을 수 없음: {not_found_count}개")
    OpLog(f"  - JSON 복사: {copied_json_count}개")
    OpLog(f"  - 저장 위치 (이미지): {MISSING_ONLY_ANNOTAIIONS_IMG}")
    OpLog(f"  - 저장 위치 (추가 JSON): {MISSING_ONLY_ANNOTATIONS_ANNO}")
    return list(unique_missing.items())
def calculate_image_checksum(image_path):
    """
    이미지 파일의 체크섬(MD5 해시) 계산
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        str: MD5 해시 값
    """
    md5_hash = hashlib.md5()
    try:
        with open(image_path, 'rb') as f:
            # 파일을 청크 단위로 읽어서 해시 계산 (메모리 효율적)
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except Exception as e:
        OpLog(f"체크섬 계산 오류 {image_path}: {e}")
        return None


def compare_images_pixel(img1_path, img2_path):
    """
    두 이미지를 픽셀 단위로 비교
    
    Args:
        img1_path: 첫 번째 이미지 경로
        img2_path: 두 번째 이미지 경로
        Returns: bool: 이미지가 동일하면 True, 아니면 False """
    try:
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        # 크기가 다르면 다른 이미지
        if img1.size != img2.size:
            return False
        
        # numpy 배열로 변환하여 비교
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        
        # 배열이 완전히 동일한지 확인
        return np.array_equal(arr1, arr2)
    except Exception as e:
        OpLog(f"이미지 비교 오류 {img1_path} vs {img2_path}: {e}")
        return False
    



def copy_missing_data(duplicates,dataset_info):
    """
    중복 이미지와 관련 annotation을 복사
    
    Args:
        duplicates: find_duplicate_images()의 반환값 [(train_img, aihub_img, method), ...]
    """
    import shutil
    
    OpLog(f"누락 데이터 복사 시작: {len(duplicates)}개")
    
    train_imge_dir, train_annotation_dir = dataset_info.get_org_data()
    AI_HUB_500_DIR, AI_HUB_LABELING_DIR = dataset_info.GetAiHubDir()    
    MISSING_TRAIN_IMAGES, MISSING_ANNOTATIONS = dataset_info.get_okImage_noAnno()

    # 디렉토리 생성
    makedirs(MISSING_TRAIN_IMAGES)
    makedirs(MISSING_ANNOTATIONS)
    
    copied_images = 0
    copied_jsons = 0
    
    for train_img_name, aihub_img_rel_path, match_method in duplicates:
        # 1. 이미지 복사
        # TRAIN_IMG_DIR의 원본 이미지
        train_img_path = os.path.join(train_imge_dir, train_img_name)
        
        # AI_HUB의 이미지 (상대 경로를 전체 경로로 변환)
        aihub_img_path = os.path.join(AI_HUB_500_DIR, aihub_img_rel_path)
        
        # 복사 대상 확인
        if os.path.exists(train_img_path):
            # TRAIN_IMG_DIR의 이미지를 그대로 이름으로 복사
            dest_train = os.path.join(MISSING_TRAIN_IMAGES, train_img_name)
            try:
                shutil.copy2(train_img_path, dest_train)
                OpLog(f"이미지 복사: {train_img_name}")
                copied_images += 1
            except Exception as e:
                OpLog(f"이미지 복사 오류 {train_img_name}: {e}")
        
        
        # if os.path.exists(aihub_img_path):
        #     # AI_HUB의 이미지를 .AI_HUB 접미사로 복사
        #     base_name = os.path.splitext(train_img_name)[0]
        #     ext = os.path.splitext(train_img_name)[1]
        #     aihub_dest_name = f"{base_name}.AI_HUB{ext}"
        #     dest_aihub = os.path.join(MISSING_TRAIN_IMAGES, aihub_dest_name)
        #     try:
        #         shutil.copy2(aihub_img_path, dest_aihub)
        #         OpLog(f"AI Hub 이미지 복사: {aihub_dest_name}")
        #         copied_images += 1
        #     except Exception as e:
        #         OpLog(f"AI Hub 이미지 복사 오류 {aihub_img_rel_path}: {e}")
        
        # 2. JSON annotation 찾기 및 복사
        # A.png -> A.json을 AI_HUB_LABELING_DIR에서 찾기
        base_name = os.path.splitext(train_img_name)[0]
        json_name = f"{base_name}.json"
        
        # MISSING_ANNOTATIONS 밑에 이미지 이름의 디렉토리 생성
        img_anno_dir = os.path.join(MISSING_ANNOTATIONS, base_name)
        makedirs(img_anno_dir)
        
        # AI_HUB_LABELING_DIR에서 재귀적으로 JSON 파일 찾기
        found_jsons = []
        if os.path.exists(AI_HUB_LABELING_DIR):
            for root, dirs, files in os.walk(AI_HUB_LABELING_DIR):
                if json_name in files:
                    json_path = os.path.join(root, json_name)
                    found_jsons.append(json_path)
        
        # 찾은 JSON 파일들을 01, 02, 03... 디렉토리에 복사
        if found_jsons:
            OpLog(f"{train_img_name}에 대한 JSON {len(found_jsons)}개 발견")
            for idx, json_path in enumerate(found_jsons, 1):
                # 01, 02, 03 형식의 디렉토리 생성
                json_subdir = os.path.join(img_anno_dir, f"{idx:02d}")
                makedirs(json_subdir)
                
                # JSON 파일 복사
                dest_json = os.path.join(json_subdir, json_name)
                try:
                    shutil.copy2(json_path, dest_json)
                    OpLog(f"  JSON 복사: {json_name} -> {base_name}/{idx:02d}/")
                    copied_jsons += 1
                except Exception as e:
                    OpLog(f"  JSON 복사 오류 {json_path}: {e}")
        else:
            OpLog(f"{train_img_name}에 대한 JSON을 찾을 수 없음")
    
    OpLog(f"복사 완료: 이미지 {copied_images}개, JSON {copied_jsons}개")
    OpLog(f"이미지 저장 위치: {MISSING_TRAIN_IMAGES}")
    OpLog(f"Annotation 저장 위치: {MISSING_ANNOTATIONS}")
    
    return copied_images, copied_jsons


def CopyAnnotionWithOnlyImage(dataset_info):


    duplicates = find_duplicate_images(dataset_info, method='checksum')
    copy_missing_data(duplicates,dataset_info)

def chai(existdir,notfounddir,svecsvfile):
#    dir1 = r"D:\01.project\EntryPrj\data\drug\drug_no_image_ok_Anno\train_images"
#    dir2 = r"D:\01.project\EntryPrj\data\eda\noImage_okAnno\train_images"
    # dir1 에는 있고 dir2 에는 없는 이미지 찾기
    images_dir1 = set(os.listdir(existdir))
    images_dir2 = set(os.listdir(notfounddir))
    missing_images = images_dir1 - images_dir2  # dir1에만 있는 이미지
    print(f"{images_dir1}에만 있는 이미지: {len(missing_images)}개")
    # csv 파일로 저장   
    with open(svecsvfile, 'w', encoding='utf-8') as f:
        f.write("missing_images\n")
        for img_name in sorted(missing_images):
            f.write(f"{img_name}\n")



# ════════════════════════════════════════
# ▣ 04. 실행
# ════════════════════════════════════════

if __name__ == "__main__":
    # 이미지 및 어노테이션 디렉토리 경로 설정
    base_dir = r"D:\01.project\EntryPrj\data\eda"
    dataset_info = DatasetInfo(base_dir)

    Save_Submission_box()

    #img_to_json,json_to_img = MakeStatistic(dataset_info) # 통계 시각화 실행
    #CopyOkImageOkJson(dataset_info=dataset_info,img_to_json_mapping=img_to_json)
    CopyImageWithOnlyAnno(dataset_info=dataset_info)
    CopyAnnotionWithOnlyImage(dataset_info=dataset_info)

