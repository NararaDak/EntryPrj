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

# ════════════════════════════════════════
# ▣ 01. 디렉토리 및 유틸 함수 설정 
# ════════════════════════════════════════
BASE_DIR = r"D:\01.project\EntryPrj\data"
LOG_FILE = os.path.join(BASE_DIR, "operation.log")
train_annotation_dir = os.path.join(BASE_DIR, "oraldrug", "train_annotations")
train_imge_dir = os.path.join(BASE_DIR, "oraldrug", "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "oraldrug", "test_images")
MODEL_FILES = os.path.join(BASE_DIR, "oraldrug", "models")
IMG_TO_JSON = os.path.join(BASE_DIR, "oraldrug", "img_to_json.csv")
JSON_TO_IMG = os.path.join(BASE_DIR, "oraldrug", "json_to_img.csv")
DUPLICATE_RESULT = os.path.join(BASE_DIR, "oraldrug", "duplicate_images.csv")
AI_HUB_500_DIR = r"D:\01.project\EntryPrj\data\AI_HUB\DATA_01\1.Training\원천데이터\경구약제조합 5000종"
AI_HUB_LABELING_DIR = r"D:\01.project\EntryPrj\data\AI_HUB\DATA_01\1.Training\라벨링데이터\경구약제조합 5000종"
MISSING_ANNOTATIONS = r"D:\01.project\EntryPrj\data\oraldrug\missingJson\annotations"
MISSING_TRAIN_IMAGES = r"D:\01.project\EntryPrj\data\oraldrug\missingJson\train_image"
AI_HUB_LABELING_DIR = r"D:\01.project\EntryPrj\data\AI_HUB\DATA_01\1.Training\라벨링데이터\경구약제조합 5000종"
MISSING_ONLY_ANNOTAIIONS_IMG = r"D:\01.project\EntryPrj\data\oraldrug\missingJson\only_annoataions_img"
MISSING_ONLY_ANNOTATIONS_ADD = r"D:\01.project\EntryPrj\data\oraldrug\missingJson\only_annoataions_add"
##------------------------

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
            with open(log_filename, 'a', encoding='utf-8') as f:
                f.write(log_content)
    except Exception as e:
        print(f"Log write error: {e}")

# ════════════════════════════════════════
# ▣ 02. 이미지-JSON 매핑 분석 함수
# ════════════════════════════════════════

def analyze_image_json_mapping(train_img_dir,train_annotation_dir):
    """
    이미지와 JSON 파일 간의 매핑 관계를 분석하여 CSV 파일로 저장
    
    생성 파일:
        - IMG_TO_JSON: 이미지 -> JSON 매핑 (이미지명, JSON경로)
        - JSON_TO_IMG: JSON -> 이미지 매핑 (JSON경로, 이미지명)
    
    Returns:
        tuple: (img_to_json_mapping, json_to_img_mapping)
    """
    OpLog("이미지-JSON 매핑 분석 시작", bLines=True)
    
    # 1. train_imge_dir 밑의 모든 이미지 파일 수집
    image_files = {}  # {filename: full_path}
    if os.path.exists(train_img_dir):
        for img_file in os.listdir(train_img_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_files[img_file] = os.path.join(train_img_dir, img_file)
    
    OpLog(f"이미지 파일 수: {len(image_files)}", bLines=True)
    
    # 2. train_annotation_dir 밑의 모든 JSON 파일 수집
    json_files = []  # [(json_path, images_in_json), ...]
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
                                # JSON 파일에서 이미지 정보 추출
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
                                    OpLog(f"JSON 파일 읽기 오류 {json_path}: {e}", bLines=True)
    
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
    
    # 5. IMG_TO_JSON CSV 파일 저장
    makedirs(os.path.dirname(IMG_TO_JSON))
    with open(IMG_TO_JSON, 'w', encoding='utf-8') as f:
        f.write("Image,JSON\n")
        for img_name, json_path in img_to_json_mapping:
            f.write(f"{img_name},{json_path}\n")
    
    OpLog(f"IMG_TO_JSON 저장 완료: {IMG_TO_JSON} ({len(img_to_json_mapping)}개 매핑)", bLines=True)
    
    # 6. JSON_TO_IMG CSV 파일 저장
    with open(JSON_TO_IMG, 'w', encoding='utf-8') as f:
        f.write("JSON,Image\n")
        for json_path, img_name in json_to_img_mapping:
            f.write(f"{json_path},{img_name}\n")
    
    OpLog(f"JSON_TO_IMG 저장 완료: {JSON_TO_IMG} ({len(json_to_img_mapping)}개 매핑)", bLines=True)
    
    # 7. 통계 정보 출력
    img_without_json = sum(1 for _, json_path in img_to_json_mapping if json_path == "NONE")
    json_without_img = sum(1 for _, img_name in json_to_img_mapping if img_name.startswith("NONE"))
    
    OpLog(f"매핑 분석 완료:", bLines=True)
    OpLog(f"전체 이미지: {len(image_files)}개", bLines=True)
    OpLog(f"전체 JSON: {len(json_files)}개", bLines=True)
    OpLog(f"JSON 없는 이미지: {img_without_json}개", bLines=True)
    OpLog(f"이미지 없는 JSON: {json_without_img}개", bLines=True)
    return img_to_json_mapping, json_to_img_mapping

def visualize_mapping_statistics(img_to_json_mapping, json_to_img_mapping):
    """
    이미지-JSON 매핑 통계를 시각화
    
    Args:
        img_to_json_mapping: analyze_image_json_mapping()의 첫 번째 반환값
        json_to_img_mapping: analyze_image_json_mapping()의 두 번째 반환값
    """
    OpLog("매핑 통계 시각화 시작", bLines=True)
    
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
    summary_text = f"""전체 통계 요약
    
총 이미지: {total_images:,}개
JSON 있는 이미지: {images_with_json:,}개
JSON 없는 이미지: {images_with_no_json:,}개

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
    
    ax4.set_xlabel('이미지당 JSON 개수', fontsize=10)
    ax4.set_ylabel('이미지 수', fontsize=10)
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
    
    ax5.set_xlabel('JSON당 이미지 개수', fontsize=10)
    ax5.set_ylabel('JSON 수', fontsize=10)
    ax5.set_title('▶ JSON당 이미지 개수 분포', fontsize=12, fontweight='bold', pad=10)
    ax5.grid(axis='y', alpha=0.3)
    
    # 2-6. 이미지당 JSON 개수 상세 통계
    ax6 = plt.subplot(3, 3, 6)
    img_json_counts = list(img_json_count.values())
    stats_text = f"""이미지당 JSON 개수 통계

최소: {min(img_json_counts)}개
최대: {max(img_json_counts)}개
평균: {sum(img_json_counts)/len(img_json_counts):.2f}개
중앙값: {sorted(img_json_counts)[len(img_json_counts)//2]}개

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
    stats_text = f"""JSON당 이미지 개수 통계

최소: {min(json_img_counts)}개
최대: {max(json_img_counts)}개
평균: {sum(json_img_counts)/len(json_img_counts):.2f}개
중앙값: {sorted(json_img_counts)[len(json_img_counts)//2]}개

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
    save_path = os.path.join(BASE_DIR, "oraldrug", "mapping_statistics.png")
    makedirs(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    OpLog(f"통계 그래프 저장: {save_path}", bLines=True)
    
    plt.show()
    plt.close()

# ════════════════════════════════════════
# ▣ 03. 이미지 중복 검사 함수
# ════════════════════════════════════════

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
        OpLog(f"체크섬 계산 오류 {image_path}: {e}", bLines=True)
        return None

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
        OpLog(f"이미지 비교 오류 {img1_path} vs {img2_path}: {e}", bLines=True)
        return False

def find_duplicate_images(train_imge_dir,train_annotation_dir, method='checksum'):
    """
    JSON 파일이 없는 이미지를 찾아서 AI_HUB_500_DIR에 중복이 있는지 검사
    
    Args:
        method: 비교 방법 ('checksum' 또는 'pixel')
            - 'checksum': MD5 해시 기반 빠른 비교
            - 'pixel': 픽셀 단위 정확한 비교 (느림)
    
    Returns:
        list: [(train_img, aihub_img, match_method), ...] 중복 이미지 리스트
    """
    OpLog(f"중복 이미지 검사 시작 (방법: {method})", bLines=True)
    
    # 1. JSON 파일이 없는 이미지 찾기
    OpLog("JSON 없는 이미지 검색 중...", bLines=True)
    
    # TRAIN_IMG_DIR의 모든 이미지 수집
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
    
    OpLog(f"JSON 없는 이미지: {len(images_without_json)}개", bLines=True)
    
    if len(images_without_json) == 0:
        OpLog("JSON 없는 이미지가 없습니다.", bLines=True)
        return []
    
    # 2. AI_HUB_500_DIR의 이미지 수집 (서브 디렉토리 포함)
    if not os.path.exists(AI_HUB_500_DIR):
        OpLog(f"AI Hub 디렉토리가 존재하지 않습니다: {AI_HUB_500_DIR}", bLines=True)
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
    
    OpLog(f"AI Hub 이미지: {len(aihub_images)}개 (서브 디렉토리 포함)", bLines=True)
    
    # 3. 중복 검사
    duplicates = []
    
    if method == 'checksum':
        OpLog("체크섬 기반 중복 검사 중...", bLines=True)
        
        # AI Hub 이미지의 체크섬 미리 계산
        aihub_checksums = {}
        for idx, (aihub_name, aihub_path) in enumerate(aihub_images.items(), 1):
            if idx % 100 == 0:
                OpLog(f"AI Hub 체크섬 계산 중: {idx}/{len(aihub_images)}", bLines=True)
            checksum = calculate_image_checksum(aihub_path)
            if checksum:
                if checksum not in aihub_checksums:
                    aihub_checksums[checksum] = []
                aihub_checksums[checksum].append((aihub_name, aihub_path))
        
        # Train 이미지와 비교
        for idx, (train_name, train_path) in enumerate(images_without_json.items(), 1):
            if idx % 10 == 0:
                OpLog(f"중복 검사 중: {idx}/{len(images_without_json)}", bLines=True)
            
            train_checksum = calculate_image_checksum(train_path)
            if train_checksum and train_checksum in aihub_checksums:
                for aihub_name, aihub_path in aihub_checksums[train_checksum]:
                    duplicates.append((train_name, aihub_name, 'checksum'))
                    OpLog(f"중복 발견 (체크섬): {train_name} == {aihub_name}", bLines=True)
    
    elif method == 'pixel':
        OpLog("픽셀 기반 중복 검사 중...", bLines=True)
        
        total_comparisons = len(images_without_json) * len(aihub_images)
        current = 0
        
        for train_name, train_path in images_without_json.items():
            for aihub_name, aihub_path in aihub_images.items():
                current += 1
                if current % 1000 == 0:
                    OpLog(f"픽셀 비교 중: {current}/{total_comparisons} ({current*100/total_comparisons:.1f}%)", bLines=True)
                
                if compare_images_pixel(train_path, aihub_path):
                    duplicates.append((train_name, aihub_name, 'pixel'))
                    OpLog(f"중복 발견 (픽셀): {train_name} == {aihub_name}", bLines=True)
    
    else:
        OpLog(f"알 수 없는 비교 방법: {method}", bLines=True)
        return []
    
    # 4. 결과 저장
    makedirs(os.path.dirname(DUPLICATE_RESULT))
    try:
        with open(DUPLICATE_RESULT, 'w', encoding='utf-8') as f:
            f.write("Train_Image,AIHub_Image,Match_Method\n")
            for train_img, aihub_img, match_method in duplicates:
                f.write(f"{train_img},{aihub_img},{match_method}\n")
        OpLog(f"결과 저장: {DUPLICATE_RESULT}", bLines=True)
    except PermissionError:
        OpLog(f"CSV 파일 쓰기 권한 오류: {DUPLICATE_RESULT}", bLines=True)
        OpLog("파일이 다른 프로그램에서 열려있는지 확인하세요.", bLines=True)
    except Exception as e:
        OpLog(f"결과 저장 오류: {e}", bLines=True)
    
    OpLog(f"중복 이미지 검사 완료: {len(duplicates)}개 발견", bLines=True)
    
    return duplicates

def copy_missing_data(duplicates,train_imge_dir, train_annotation_dir):
    """
    중복 이미지와 관련 annotation을 복사
    
    Args:
        duplicates: find_duplicate_images()의 반환값 [(train_img, aihub_img, method), ...]
    """
    import shutil
    
    OpLog(f"누락 데이터 복사 시작: {len(duplicates)}개", bLines=True)
    
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
                OpLog(f"이미지 복사: {train_img_name}", bLines=True)
                copied_images += 1
            except Exception as e:
                OpLog(f"이미지 복사 오류 {train_img_name}: {e}", bLines=True)
        
        if os.path.exists(aihub_img_path):
            # AI_HUB의 이미지를 .AI_HUB 접미사로 복사
            base_name = os.path.splitext(train_img_name)[0]
            ext = os.path.splitext(train_img_name)[1]
            aihub_dest_name = f"{base_name}.AI_HUB{ext}"
            dest_aihub = os.path.join(MISSING_TRAIN_IMAGES, aihub_dest_name)
            try:
                shutil.copy2(aihub_img_path, dest_aihub)
                OpLog(f"AI Hub 이미지 복사: {aihub_dest_name}", bLines=True)
                copied_images += 1
            except Exception as e:
                OpLog(f"AI Hub 이미지 복사 오류 {aihub_img_rel_path}: {e}", bLines=True)
        
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
            OpLog(f"{train_img_name}에 대한 JSON {len(found_jsons)}개 발견", bLines=True)
            for idx, json_path in enumerate(found_jsons, 1):
                # 01, 02, 03 형식의 디렉토리 생성
                json_subdir = os.path.join(img_anno_dir, f"{idx:02d}")
                makedirs(json_subdir)
                
                # JSON 파일 복사
                dest_json = os.path.join(json_subdir, json_name)
                try:
                    shutil.copy2(json_path, dest_json)
                    OpLog(f"  JSON 복사: {json_name} -> {base_name}/{idx:02d}/", bLines=True)
                    copied_jsons += 1
                except Exception as e:
                    OpLog(f"  JSON 복사 오류 {json_path}: {e}", bLines=True)
        else:
            OpLog(f"{train_img_name}에 대한 JSON을 찾을 수 없음", bLines=True)
    
    OpLog(f"복사 완료: 이미지 {copied_images}개, JSON {copied_jsons}개", bLines=True)
    OpLog(f"이미지 저장 위치: {MISSING_TRAIN_IMAGES}", bLines=True)
    OpLog(f"Annotation 저장 위치: {MISSING_ANNOTATIONS}", bLines=True)
    
    return copied_images, copied_jsons

def process_duplicate_result(train_imge_dir):
    """
    DUPLICATE_RESULT 파일을 읽어서:
    1. 이미지 복사 (Train 이미지 + AI Hub 이미지)
    2. JSON annotation 복사
    3. 바운딩 박스 시각화 (A.box.png)
    """
    OpLog("DUPLICATE_RESULT 처리 시작", bLines=True)
    
    # 1. DUPLICATE_RESULT 파일 읽기
    if not os.path.exists(DUPLICATE_RESULT):
        OpLog(f"DUPLICATE_RESULT 파일을 찾을 수 없습니다: {DUPLICATE_RESULT}", bLines=True)
        return
    
    duplicates = []
    try:
        with open(DUPLICATE_RESULT, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 헤더 제외
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    train_img, aihub_img, method = parts[0], parts[1], parts[2]
                    duplicates.append((train_img, aihub_img, method))
        
        OpLog(f"DUPLICATE_RESULT에서 {len(duplicates)}개 항목 읽음", bLines=True)
    except Exception as e:
        OpLog(f"DUPLICATE_RESULT 읽기 오류: {e}", bLines=True)
        return
    
    # 2. 디렉토리 생성
    makedirs(MISSING_TRAIN_IMAGES)
    makedirs(MISSING_ANNOTATIONS)
    
    copied_images = 0
    copied_jsons = 0
    visualized = 0
    
    # 3. 각 중복 항목 처리
    for train_img_name, aihub_img_rel_path, match_method in duplicates:
        base_name = os.path.splitext(train_img_name)[0]
        ext = os.path.splitext(train_img_name)[1]
        
        # 3-1. 이미지 복사
        train_img_path = os.path.join(train_imge_dir, train_img_name)
        aihub_img_path = os.path.join(AI_HUB_500_DIR, aihub_img_rel_path)
        
        # Train 이미지 복사 (원본 이름)
        if os.path.exists(train_img_path):
            dest_train = os.path.join(MISSING_TRAIN_IMAGES, train_img_name)
            try:
                shutil.copy2(train_img_path, dest_train)
                copied_images += 1
            except Exception as e:
                OpLog(f"이미지 복사 오류 {train_img_name}: {e}", bLines=True)
        
        # AI Hub 이미지 복사 (.AI_HUB 접미사)
        if os.path.exists(aihub_img_path):
            aihub_dest_name = f"{base_name}.AI_HUB{ext}"
            dest_aihub = os.path.join(MISSING_TRAIN_IMAGES, aihub_dest_name)
            try:
                shutil.copy2(aihub_img_path, dest_aihub)
                copied_images += 1
            except Exception as e:
                OpLog(f"AI Hub 이미지 복사 오류 {aihub_img_rel_path}: {e}", bLines=True)
        
        # 3-2. JSON annotation 찾기 및 복사
        json_name = f"{base_name}.json"
        img_anno_dir = os.path.join(MISSING_ANNOTATIONS, base_name)
        makedirs(img_anno_dir)
        
        # AI_HUB_LABELING_DIR에서 재귀적으로 JSON 찾기
        found_jsons = []
        if os.path.exists(AI_HUB_LABELING_DIR):
            for root, dirs, files in os.walk(AI_HUB_LABELING_DIR):
                if json_name in files:
                    json_path = os.path.join(root, json_name)
                    found_jsons.append(json_path)
        
        # JSON 복사 (01, 02, 03... 디렉토리에)
        json_files_for_viz = []
        if found_jsons:
            for idx, json_path in enumerate(found_jsons, 1):
                json_subdir = os.path.join(img_anno_dir, f"{idx:02d}")
                makedirs(json_subdir)
                
                dest_json = os.path.join(json_subdir, json_name)
                try:
                    shutil.copy2(json_path, dest_json)
                    json_files_for_viz.append(json_path)
                    copied_jsons += 1
                except Exception as e:
                    OpLog(f"JSON 복사 오류 {json_path}: {e}", bLines=True)
        
        # 3-3. 바운딩 박스 시각화 (A.box.png)
        if os.path.exists(train_img_path) and json_files_for_viz:
            try:
                visualize_bounding_boxes(train_img_path, json_files_for_viz, base_name)
                visualized += 1
            except Exception as e:
                OpLog(f"바운딩 박스 시각화 오류 {train_img_name}: {e}", bLines=True)
    
    OpLog(f"처리 완료:", bLines=True)
    OpLog(f"  - 이미지 복사: {copied_images}개", bLines=True)
    OpLog(f"  - JSON 복사: {copied_jsons}개", bLines=True)
    OpLog(f"  - 바운딩 박스 시각화: {visualized}개", bLines=True)
    OpLog(f"  - 저장 위치: {MISSING_TRAIN_IMAGES}", bLines=True)

def visualize_bounding_boxes(image_path, json_paths, base_name):
    """
    이미지에 모든 JSON 파일의 바운딩 박스를 그리고 dl_name 표시
    
    Args:
        image_path: 원본 이미지 경로 (A.png)
        json_paths: JSON 파일 경로 리스트 [path1, path2, ...]
        base_name: 파일 기본 이름 (A)
    """
    # 이미지 로드
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        OpLog(f"이미지 로드 실패 {image_path}: {e}", bLines=True)
        return
    
    draw = ImageDraw.Draw(img)
    
    # 폰트 설정 (시스템 기본 폰트 사용)
    try:
        # Windows 한글 폰트
        font = ImageFont.truetype("malgun.ttf", 20)
        small_font = ImageFont.truetype("malgun.ttf", 14)
    except:
        # 폰트 로드 실패 시 기본 폰트
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # 색상 팔레트 (여러 JSON 구분용)
    colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
    
    # 모든 JSON 파일 처리
    all_dl_names = []
    for json_idx, json_path in enumerate(json_paths):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # dl_name 추출
            dl_name = "Unknown"
            if 'images' in data and len(data['images']) > 0:
                dl_name = data['images'][0].get('dl_name', 'Unknown')
            all_dl_names.append(dl_name)
            
            # annotations 처리
            if 'annotations' in data:
                color = colors[json_idx % len(colors)]
                
                for anno in data['annotations']:
                    if 'bbox' in anno:
                        bbox = anno['bbox']
                        # bbox 형식: [x, y, width, height]
                        x, y, w, h = bbox
                        
                        # 바운딩 박스 그리기
                        draw.rectangle(
                            [(x, y), (x + w, y + h)],
                            outline=color,
                            width=3
                        )
                        
                        # 카테고리 ID 표시 (있으면)
                        if 'category_id' in anno:
                            cat_id = anno['category_id']
                            draw.text(
                                (x, y - 20),
                                f"Cat:{cat_id}",
                                fill=color,
                                font=small_font
                            )
        
        except Exception as e:
            OpLog(f"JSON 파싱 오류 {json_path}: {e}", bLines=True)
            continue
    
    # dl_name 텍스트를 이미지 상단에 표시
    if all_dl_names:
        y_pos = 10
        for idx, dl_name in enumerate(all_dl_names):
            color = colors[idx % len(colors)]
            text = f"[{idx+1}] {dl_name}"
            draw.text((10, y_pos), text, fill=color, font=font)
            y_pos += 30
    
    # 저장
    output_path = os.path.join(MISSING_TRAIN_IMAGES, f"{base_name}.box.png")
    try:
        img.save(output_path)
        OpLog(f"바운딩 박스 이미지 저장: {base_name}.box.png ({len(json_paths)}개 JSON)", bLines=True)
    except Exception as e:
        OpLog(f"이미지 저장 실패 {output_path}: {e}", bLines=True)



def find_missing_images_with_json(train_imge_dir, train_annotation_dir):
    """
    ANNOTATION_DIR에는 JSON 파일이 있지만 TRAIN_IMG_DIR에는 이미지가 없는 경우를 찾아서
    AI_HUB_500_DIR에서 이미지를 찾아 MISSING_ONLY_ANNOTAIIONS_IMG 복사하고 바운딩 박스 시각화
    AI_HUB_LABELING_DIR에서 추가 JSON 파일을 찾아 MISSING_ONLY_ANNOTATIONS_ADD에 복사
    """
    OpLog("JSON은 있지만 이미지가 없는 경우 검색 시작", bLines=True)
    
    # 1. TRAIN_IMG_DIR의 모든 이미지 파일 수집
    train_images = set()
    if os.path.exists(train_imge_dir):
        for img_file in os.listdir(train_imge_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                train_images.add(img_file)
    
    OpLog(f"Train 이미지: {len(train_images)}개", bLines=True)
    
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
                        OpLog(f"JSON 파일 읽기 오류 {json_path}: {e}", bLines=True)
    
    OpLog(f"JSON 파일: {len(json_image_mapping)}개", bLines=True)
    
    # 3. Train에 없는 이미지 찾기
    missing_images = []  # [(json_path, img_filename, dl_name), ...]
    for json_path, img_filename, dl_name in json_image_mapping:
        if img_filename not in train_images:
            missing_images.append((json_path, img_filename, dl_name))
    
    OpLog(f"Train에 없는 이미지: {len(missing_images)}개", bLines=True)
    
    if len(missing_images) == 0:
        OpLog("모든 JSON에 대응하는 이미지가 Train 디렉토리에 있습니다.", bLines=True)
        return []
    
    # 4. AI_HUB_500_DIR에서 해당 이미지 찾기
    if not os.path.exists(AI_HUB_500_DIR):
        OpLog(f"AI Hub 디렉토리가 존재하지 않습니다: {AI_HUB_500_DIR}", bLines=True)
        return []
    
    # AI Hub 이미지 수집 (파일명을 키로)
    aihub_images = {}  # {filename: full_path}
    OpLog("AI Hub 이미지 수집 중...", bLines=True)
    for root, dirs, files in os.walk(AI_HUB_500_DIR):
        for img_file in files:
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                full_path = os.path.join(root, img_file)
                # 파일명이 중복될 수 있으므로 리스트로 저장
                if img_file not in aihub_images:
                    aihub_images[img_file] = []
                aihub_images[img_file].append(full_path)
    
    OpLog(f"AI Hub 이미지: {len(aihub_images)}개 파일명", bLines=True)
    
    # 5. 디렉토리 생성
    makedirs(MISSING_ONLY_ANNOTAIIONS_IMG)
    makedirs(MISSING_ONLY_ANNOTATIONS_ADD)
    
    # 6. 각 missing 이미지 처리
    copied_count = 0
    visualized_count = 0
    not_found_count = 0
    copied_json_count = 0
    
    for json_path, img_filename, dl_name in missing_images:
        base_name = os.path.splitext(img_filename)[0]
        ext = os.path.splitext(img_filename)[1]
        json_filename = f"{base_name}.json"
        
        # AI Hub에서 이미지 찾기
        if img_filename in aihub_images:
            # 첫 번째 매칭되는 이미지 사용
            aihub_img_path = aihub_images[img_filename][0]
            
            # 이미지 복사
            dest_img_path = os.path.join(MISSING_ONLY_ANNOTAIIONS_IMG, img_filename)
            try:
                shutil.copy2(aihub_img_path, dest_img_path)
                copied_count += 1
                OpLog(f"이미지 복사: {img_filename}", bLines=True)
                    
            except Exception as e:
                OpLog(f"이미지 복사 오류 {img_filename}: {e}", bLines=True)
        else:
            not_found_count += 1
            OpLog(f"AI Hub에서 찾을 수 없음: {img_filename} (dl_name: {dl_name})", bLines=True)
        
        # 7. AI_HUB_LABELING_DIR에서 추가 JSON 파일 찾기
        all_jsons_for_viz = [json_path]  # 원본 JSON부터 시작
        if os.path.exists(AI_HUB_LABELING_DIR):
            # AI_HUB_LABELING_DIR에서 모든 a.json 찾기
            found_aihub_jsons = []
            for root, dirs, files in os.walk(AI_HUB_LABELING_DIR):
                if json_filename in files:
                    aihub_json_path = os.path.join(root, json_filename)
                    # dl_name 추출
                    try:
                        with open(aihub_json_path, 'r', encoding='utf-8') as f:
                            aihub_data = json.load(f)
                        aihub_dl_name = 'Unknown'
                        if 'images' in aihub_data and len(aihub_data['images']) > 0:
                            aihub_dl_name = aihub_data['images'][0].get('dl_name', 'Unknown')
                        
                        found_aihub_jsons.append((aihub_json_path, aihub_dl_name))
                    except Exception as e:
                        OpLog(f"AI Hub JSON 읽기 오류 {aihub_json_path}: {e}", bLines=True)
            
            # ANNOTATION_DIR의 dl_name과 다른 JSON만 필터링
            additional_jsons = []
            for aihub_json_path, aihub_dl_name in found_aihub_jsons:
                if aihub_dl_name != dl_name:
                    additional_jsons.append(aihub_json_path)
            
            # 바운딩 박스 시각화를 위해 추가 JSON들도 리스트에 추가
            all_jsons_for_viz.extend(additional_jsons)
            
            if additional_jsons:
                # MISSING_ONLY_ANNOTATIONS_ADD/a/ 디렉토리 생성
                anno_add_dir = os.path.join(MISSING_ONLY_ANNOTATIONS_ADD, base_name)
                makedirs(anno_add_dir)
                
                # 01 디렉토리에 ANNOTATION_DIR의 원본 JSON 복사
                dir_01 = os.path.join(anno_add_dir, "01")
                makedirs(dir_01)
                dest_json_01 = os.path.join(dir_01, json_filename)
                try:
                    shutil.copy2(json_path, dest_json_01)
                    copied_json_count += 1
                    OpLog(f"  원본 JSON 복사: {base_name}/01/{json_filename}", bLines=True)
                except Exception as e:
                    OpLog(f"  원본 JSON 복사 오류 {json_path}: {e}", bLines=True)
                
                # 02, 03, ... 디렉토리에 AI Hub의 추가 JSON 복사
                for idx, aihub_json_path in enumerate(additional_jsons, 2):
                    dir_num = os.path.join(anno_add_dir, f"{idx:02d}")
                    makedirs(dir_num)
                    dest_json = os.path.join(dir_num, json_filename)
                    try:
                        shutil.copy2(aihub_json_path, dest_json)
                        copied_json_count += 1
                        OpLog(f"  추가 JSON 복사: {base_name}/{idx:02d}/{json_filename}", bLines=True)
                    except Exception as e:
                        OpLog(f"  추가 JSON 복사 오류 {aihub_json_path}: {e}", bLines=True)
                
                OpLog(f"JSON 복사 완료: {base_name} (원본 1개 + 추가 {len(additional_jsons)}개)", bLines=True)
    
    OpLog(f"처리 완료:", bLines=True)
    OpLog(f"  - 이미지 복사: {copied_count}개", bLines=True)
    OpLog(f"  - AI Hub에서 찾을 수 없음: {not_found_count}개", bLines=True)
    OpLog(f"  - JSON 복사: {copied_json_count}개", bLines=True)
    OpLog(f"  - 저장 위치 (이미지): {MISSING_ONLY_ANNOTAIIONS_IMG}", bLines=True)
    OpLog(f"  - 저장 위치 (추가 JSON): {MISSING_ONLY_ANNOTATIONS_ADD}", bLines=True)
    
    # 9. 바운딩 박스 시각화 일괄 처리
    OpLog("바운딩 박스 시각화 시작...", bLines=True)
   # process_missing_only_visualization()
    
    return missing_images

def visualize_single_image_boxes(image_path, json_path, base_name, dl_name):
    """
    단일 이미지에 JSON 파일의 바운딩 박스를 그리고 dl_name 표시
    
    Args:
        image_path: 원본 이미지 경로
        json_path: JSON 파일 경로
        base_name: 파일 기본 이름
        dl_name: 클래스 이름
    """
    # 이미지 로드
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        OpLog(f"이미지 로드 실패 {image_path}: {e}", bLines=True)
        return
    
    draw = ImageDraw.Draw(img)
    
    # 폰트 설정
    try:
        font = ImageFont.truetype("malgun.ttf", 24)
        small_font = ImageFont.truetype("malgun.ttf", 16)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # JSON 파일 읽기
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # dl_name을 이미지 상단에 표시
        draw.text((10, 10), f"Class: {dl_name}", fill='red', font=font)
        
        # annotations 처리
        if 'annotations' in data:
            for anno in data['annotations']:
                if 'bbox' in anno:
                    bbox = anno['bbox']
                    # bbox 형식: [x, y, width, height]
                    x, y, w, h = bbox
                    
                    # 바운딩 박스 그리기
                    draw.rectangle(
                        [(x, y), (x + w, y + h)],
                        outline='red',
                        width=3
                    )
                    
                    # 카테고리 ID 표시
                    if 'category_id' in anno:
                        cat_id = anno['category_id']
                        draw.text(
                            (x, y - 25),
                            f"Cat:{cat_id}",
                        fill='red',
                        font=small_font
                    )
    
    except Exception as e:
        OpLog(f"JSON 파싱 오류 {json_path}: {e}", bLines=True)
        # 파싱 오류를 별도 로그 파일에 기록
        parse_err_log = os.path.join(os.path.dirname(LOG_FILE), "parseerr.log")
        with open(parse_err_log, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.datetime.now()}] JSON 파싱 오류: {json_path}\n")
            f.write(f"  오류 내용: {e}\n\n")
        return
    
    # 저장
    output_path = os.path.join(MISSING_ONLY_ANNOTAIIONS_IMG, f"{base_name}.box.png")
    try:
        img.save(output_path)
        OpLog(f"바운딩 박스 이미지 저장: {base_name}.box.png", bLines=True)
    except Exception as e:
        OpLog(f"이미지 저장 실패 {output_path}: {e}", bLines=True)

def visualize_missing_only_boxes(image_path, json_paths, base_name):
    """
    이미지에 모든 JSON 파일의 바운딩 박스를 그리고 dl_name 표시
    (MISSING_ONLY_ANNOTAIIONS_IMG에 저장)
    
    Args:
        image_path: 원본 이미지 경로
        json_paths: JSON 파일 경로 리스트 [path1, path2, ...]
        base_name: 파일 기본 이름
    """
    # 이미지 로드
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        OpLog(f"이미지 로드 실패 {image_path}: {e}", bLines=True)
        return
    
    draw = ImageDraw.Draw(img)
    
    # 폰트 설정
    try:
        font = ImageFont.truetype("malgun.ttf", 20)
        small_font = ImageFont.truetype("malgun.ttf", 14)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # 색상 팔레트 (여러 JSON 구분용)
    colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
    
    # 모든 JSON 파일 처리
    all_dl_names = []
    for json_idx, json_path in enumerate(json_paths):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # dl_name 추출
            dl_name = "Unknown"
            if 'images' in data and len(data['images']) > 0:
                dl_name = data['images'][0].get('dl_name', 'Unknown')
            all_dl_names.append(dl_name)
            
            # annotations 처리
            if 'annotations' in data:
                color = colors[json_idx % len(colors)]
                
                for anno in data['annotations']:
                    if 'bbox' in anno:
                        bbox = anno['bbox']
                        # bbox 형식: [x, y, width, height]
                        if len(bbox) != 4:
                            OpLog(f"잘못된 bbox 형식 (4개 값 필요, {len(bbox)}개 발견): {bbox} in {json_path}", bLines=True)
                            # bbox 형식 오류를 parseerr.log에 기록
                            parse_err_log = os.path.join(os.path.dirname(LOG_FILE), "parseerr.log")
                            with open(parse_err_log, 'a', encoding='utf-8') as f:
                                f.write(f"[{datetime.datetime.now()}] 잘못된 bbox 형식: {json_path}\n")
                                f.write(f"  bbox 값: {bbox} (4개 값 필요, {len(bbox)}개 발견)\n\n")
                            continue
                        
                        x, y, w, h = bbox
                        
                        # 바운딩 박스 그리기
                        draw.rectangle(
                            [(x, y), (x + w, y + h)],
                            outline=color,
                            width=3
                        )
                        
                        # 카테고리 ID 표시
                        if 'category_id' in anno:
                            cat_id = anno['category_id']
                            draw.text(
                                (x, y - 20),
                                f"Cat:{cat_id}",
                                fill=color,
                                font=small_font
                            )
        
        except Exception as e:
            OpLog(f"JSON 파싱 오류 {json_path}: {e}", bLines=True)
            # 파싱 오류를 별도 로그 파일에 기록
            parse_err_log = os.path.join(os.path.dirname(LOG_FILE), "parseerr.log")
            with open(parse_err_log, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.datetime.now()}] JSON 파싱 오류: {json_path}\n")
                f.write(f"  오류 내용: {e}\n\n")
            continue
    
    # dl_name 텍스트를 이미지 상단에 표시
    if all_dl_names:
        y_pos = 10
        for idx, dl_name in enumerate(all_dl_names):
            color = colors[idx % len(colors)]
            text = f"[{idx+1}] {dl_name}"
            draw.text((10, y_pos), text, fill=color, font=font)
            y_pos += 30
    
    # MISSING_ONLY_ANNOTAIIONS_IMG에 저장
    output_path = os.path.join(MISSING_ONLY_ANNOTAIIONS_IMG, f"{base_name}.box.png")
    try:
        img.save(output_path)
        OpLog(f"바운딩 박스 이미지 저장: {base_name}.box.png ({len(json_paths)}개 JSON)", bLines=True)
    except Exception as e:
        OpLog(f"이미지 저장 실패 {output_path}: {e}", bLines=True)

def change_category_id(json_path):
    pass


def visualize_single_file_boxes(image_filename):
    """
    단일 이미지 파일에 대한 바운딩 박스 시각화
    MISSING_ONLY_ANNOTAIIONS_IMG에서 이미지를 찾고
    MISSING_ONLY_ANNOTATIONS_ADD에서 JSON들을 찾아서 시각화
    
    Args:
        image_filename: 이미지 파일명 (예: "A.png")
    """
    OpLog(f"단일 파일 바운딩 박스 시각화: {image_filename}", bLines=True)
    
    # 1. 이미지 파일 경로 확인
    img_path = os.path.join(MISSING_ONLY_ANNOTAIIONS_IMG, image_filename)
    if not os.path.exists(img_path):
        OpLog(f"이미지 파일을 찾을 수 없습니다: {img_path}", bLines=True)
        return False
    
    # 2. base_name 추출
    base_name = os.path.splitext(image_filename)[0]
    
    # 3. JSON 디렉토리 확인
    anno_dir = os.path.join(MISSING_ONLY_ANNOTATIONS_ADD, base_name)
    if not os.path.exists(anno_dir):
        OpLog(f"JSON 디렉토리를 찾을 수 없습니다: {anno_dir}", bLines=True)
        return False
    
    # 4. 모든 서브 디렉토리(01, 02, 03...)에서 JSON 파일 수집
    json_files = []
    for subdir in sorted(os.listdir(anno_dir)):
        subdir_path = os.path.join(anno_dir, subdir)
        if os.path.isdir(subdir_path):
            json_filename = f"{base_name}.json"
            json_path = os.path.join(subdir_path, json_filename)
            if os.path.exists(json_path):
                json_files.append(json_path)
                OpLog(f"  JSON 발견: {subdir}/{json_filename}", bLines=True)
    
    if not json_files:
        OpLog(f"JSON 파일을 찾을 수 없습니다: {anno_dir}", bLines=True)
        return False
    
    # 5. 바운딩 박스 시각화
    try:
        visualize_missing_only_boxes(img_path, json_files, base_name)
        OpLog(f"시각화 완료: {base_name}.box.png ({len(json_files)}개 JSON)", bLines=True)
        return True
    except Exception as e:
        OpLog(f"바운딩 박스 시각화 오류: {e}", bLines=True)
        return False

def process_missing_only_visualization():
    """
    MISSING_ONLY_ANNOTAIIONS_IMG의 이미지와 MISSING_ONLY_ANNOTATIONS_ADD의 JSON을 읽어서
    바운딩 박스 시각화를 수행
    """
    OpLog("MISSING_ONLY 바운딩 박스 시각화 시작", bLines=True)
    
    # 1. MISSING_ONLY_ANNOTAIIONS_IMG의 이미지 파일 수집
    if not os.path.exists(MISSING_ONLY_ANNOTAIIONS_IMG):
        OpLog(f"이미지 디렉토리가 존재하지 않습니다: {MISSING_ONLY_ANNOTAIIONS_IMG}", bLines=True)
        return
    
    image_files = {}
    for img_file in os.listdir(MISSING_ONLY_ANNOTAIIONS_IMG):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) and not img_file.endswith('.box.png'):
            base_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(MISSING_ONLY_ANNOTAIIONS_IMG, img_file)
            image_files[base_name] = img_path
    
    OpLog(f"이미지 파일: {len(image_files)}개", bLines=True)
    
    # 2. MISSING_ONLY_ANNOTATIONS_ADD의 JSON 파일 수집
    if not os.path.exists(MISSING_ONLY_ANNOTATIONS_ADD):
        OpLog(f"JSON 디렉토리가 존재하지 않습니다: {MISSING_ONLY_ANNOTATIONS_ADD}", bLines=True)
        return
    
    visualized_count = 0
    skipped_count = 0
    
    # 3. 각 이미지에 대해 처리
    for base_name, img_path in image_files.items():
        # MISSING_ONLY_ANNOTATIONS_ADD/{base_name}/ 디렉토리 확인
        anno_dir = os.path.join(MISSING_ONLY_ANNOTATIONS_ADD, base_name)
        
        if not os.path.exists(anno_dir):
            OpLog(f"JSON 디렉토리 없음: {base_name}", bLines=True)
            skipped_count += 1
            continue
        
        # 모든 서브 디렉토리(01, 02, 03...)에서 JSON 파일 수집
        json_files = []
        for subdir in sorted(os.listdir(anno_dir)):
            subdir_path = os.path.join(anno_dir, subdir)
            if os.path.isdir(subdir_path):
                # 해당 디렉토리에서 {base_name}.json 찾기
                json_filename = f"{base_name}.json"
                json_path = os.path.join(subdir_path, json_filename)
                if os.path.exists(json_path):
                    json_files.append(json_path)
        
        if json_files:
            try:
                visualize_missing_only_boxes(img_path, json_files, base_name)
                visualized_count += 1
            except Exception as e:
                OpLog(f"바운딩 박스 시각화 오류 {base_name}: {e}", bLines=True)
        else:
            OpLog(f"JSON 파일 없음: {base_name}", bLines=True)
            skipped_count += 1
    
    OpLog(f"시각화 완료:", bLines=True)
    OpLog(f"  - 시각화 성공: {visualized_count}개", bLines=True)
    OpLog(f"  - 건너뜀: {skipped_count}개", bLines=True)
    OpLog(f"  - 저장 위치: {MISSING_ONLY_ANNOTAIIONS_IMG}", bLines=True)

def interactive_bbox_editor(image_path, json_path, bbox_error_info, dl_name="Unknown"):
    """
    잘못된 bbox를 수정할 수 있는 대화형 에디터 (GUI 입력창 사용)
    
    Args:
        image_path: 이미지 경로
        json_path: JSON 파일 경로
        bbox_error_info: 오류 정보 dict {'bbox': [], 'anno_index': 0}
        dl_name: 약물 이름
    
    Returns:
        수정된 bbox [x, y, width, height] 또는 None (건너뛰기)
    """
    print("\n" + "="*80)
    print(f"📦 Bounding Box 수정 도구")
    print(f"이미지: {os.path.basename(image_path)}")
    print(f"JSON: {os.path.basename(json_path)}")
    print(f"dl_name: {dl_name}")
    print(f"현재 bbox: {bbox_error_info['bbox']}")
    print("="*80)
    
    # 이미지 표시
    img = Image.open(image_path).convert('RGB')
    print(f"이미지 크기: {img.size[0]} x {img.size[1]} (width x height)")
    
    # matplotlib으로 이미지 표시
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(f"Image: {os.path.basename(image_path)}\nJSON: {os.path.basename(json_path)}\ndl_name: {dl_name}")
    plt.axis('on')
    
    # 현재 bbox가 있으면 표시 (부분적으로라도)
    current_bbox = bbox_error_info['bbox']
    if len(current_bbox) >= 2:
        x = current_bbox[0] if len(current_bbox) > 0 else 0
        y = current_bbox[1] if len(current_bbox) > 1 else 0
        w = current_bbox[2] if len(current_bbox) > 2 else 100
        h = current_bbox[3] if len(current_bbox) > 3 else 100
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2, linestyle='--')
        plt.gca().add_patch(rect)
        plt.text(x, y-10, f"Current (incomplete): {current_bbox}", color='red', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    
    # tkinter 루트 윈도우 생성 (숨김)
    root = tk.Tk()
    root.withdraw()
    
    # 커스텀 입력 대화상자 생성
    class BBoxInputDialog:
        def __init__(self, parent, image_path, dl_name, current_bbox, img_size):
            self.result = None
            self.preview_mode = False
            
            self.dialog = tk.Toplevel(parent)
            self.dialog.title("Bounding Box 수정")
            self.dialog.geometry("450x350")
            self.dialog.resizable(False, False)
            
            # 정보 표시
            info_frame = tk.Frame(self.dialog, padx=10, pady=10)
            info_frame.pack(fill=tk.BOTH, expand=True)
            
            tk.Label(info_frame, text=f"이미지: {os.path.basename(image_path)}", 
                    font=("맑은 고딕", 10, "bold")).pack(anchor=tk.W)
            tk.Label(info_frame, text=f"dl_name: {dl_name}", 
                    font=("맑은 고딕", 10)).pack(anchor=tk.W, pady=(5,0))
            tk.Label(info_frame, text=f"현재 bbox: {current_bbox}", 
                    font=("맑은 고딕", 9), fg="red").pack(anchor=tk.W, pady=(5,0))
            tk.Label(info_frame, text=f"이미지 크기: {img_size[0]} x {img_size[1]}", 
                    font=("맑은 고딕", 9)).pack(anchor=tk.W, pady=(5,0))
            
            # 구분선
            tk.Frame(info_frame, height=2, bg="gray").pack(fill=tk.X, pady=10)
            
            # 입력 필드
            tk.Label(info_frame, text="새로운 bbox 값을 입력하세요:", 
                    font=("맑은 고딕", 10, "bold")).pack(anchor=tk.W, pady=(5,5))
            
            input_frame = tk.Frame(info_frame)
            input_frame.pack(fill=tk.X, pady=5)
            
            tk.Label(input_frame, text="Left:", width=6, anchor=tk.W).grid(row=0, column=0, padx=2)
            self.left_entry = tk.Entry(input_frame, width=10)
            self.left_entry.grid(row=0, column=1, padx=2)
            
            tk.Label(input_frame, text="Top:", width=6, anchor=tk.W).grid(row=0, column=2, padx=2)
            self.top_entry = tk.Entry(input_frame, width=10)
            self.top_entry.grid(row=0, column=3, padx=2)
            
            tk.Label(input_frame, text="Width:", width=6, anchor=tk.W).grid(row=1, column=0, padx=2, pady=5)
            self.width_entry = tk.Entry(input_frame, width=10)
            self.width_entry.grid(row=1, column=1, padx=2, pady=5)
            
            tk.Label(input_frame, text="Height:", width=6, anchor=tk.W).grid(row=1, column=2, padx=2, pady=5)
            self.height_entry = tk.Entry(input_frame, width=10)
            self.height_entry.grid(row=1, column=3, padx=2, pady=5)
            
            # 기본값 설정 (현재 bbox가 있으면)
            if len(current_bbox) >= 1:
                self.left_entry.insert(0, str(current_bbox[0]))
            if len(current_bbox) >= 2:
                self.top_entry.insert(0, str(current_bbox[1]))
            if len(current_bbox) >= 3:
                self.width_entry.insert(0, str(current_bbox[2]))
            
            # 버튼 프레임
            button_frame = tk.Frame(info_frame)
            button_frame.pack(fill=tk.X, pady=10)
            
            tk.Button(button_frame, text="그리기", width=10, command=self.preview, 
                     bg="#4CAF50", fg="white", font=("맑은 고딕", 9, "bold")).pack(side=tk.LEFT, padx=5)
            tk.Button(button_frame, text="저장", width=10, command=self.save, 
                     bg="#2196F3", fg="white", font=("맑은 고딕", 9, "bold")).pack(side=tk.LEFT, padx=5)
            tk.Button(button_frame, text="건너뛰기", width=10, command=self.skip, 
                     bg="#FF9800", fg="white", font=("맑은 고딕", 9, "bold")).pack(side=tk.LEFT, padx=5)
            tk.Button(button_frame, text="취소", width=10, command=self.cancel, 
                     bg="#f44336", fg="white", font=("맑은 고딕", 9, "bold")).pack(side=tk.LEFT, padx=5)
            
            self.img = img
            self.img_size = img_size
            
        def get_bbox_values(self):
            try:
                left = int(self.left_entry.get().strip())
                top = int(self.top_entry.get().strip())
                width = int(self.width_entry.get().strip())
                height = int(self.height_entry.get().strip())
                return [left, top, width, height]
            except ValueError:
                return None
        
        def preview(self):
            """그리기 버튼 클릭"""
            bbox = self.get_bbox_values()
            if bbox is None:
                messagebox.showerror("입력 오류", "모든 필드에 숫자를 입력하세요.", parent=self.dialog)
                return
            
            x, y, w, h = bbox
            
            # 유효성 검사
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                messagebox.showerror("유효성 오류", 
                                    f"유효하지 않은 값입니다.\nx={x}, y={y}, w={w}, h={h}", 
                                    parent=self.dialog)
                return
            
            if x + w > self.img_size[0] or y + h > self.img_size[1]:
                result = messagebox.askyesno("경고",
                    f"bbox가 이미지 범위를 벗어납니다.\n"
                    f"이미지: {self.img_size[0]}x{self.img_size[1]}\n"
                    f"bbox: x={x}, y={y}, w={w}, h={h}\n\n"
                    "계속하시겠습니까?",
                    parent=self.dialog)
                if not result:
                    return
            
            # bbox 미리보기
            plt.clf()
            plt.imshow(self.img)
            plt.title(f"미리보기: bbox = {bbox}")
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='green', linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(x, y-10, f"Preview: [{x}, {y}, {w}, {h}]", color='green', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.axis('on')
            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)
            self.preview_mode = True
        
        def save(self):
            """저장 버튼 클릭"""
            bbox = self.get_bbox_values()
            if bbox is None:
                messagebox.showerror("입력 오류", "모든 필드에 숫자를 입력하세요.", parent=self.dialog)
                return
            
            self.result = bbox
            self.dialog.destroy()
        
        def skip(self):
            """건너뛰기 버튼 클릭"""
            self.result = 'skip'
            self.dialog.destroy()
         
        def cancel(self):
            """취소 버튼 클릭"""
            self.result = None
            self.dialog.destroy()
    
    # 대화상자 표시
    dialog = BBoxInputDialog(root, image_path, dl_name, current_bbox, img.size)
    root.wait_window(dialog.dialog)
    
    # 결과 처리
    if dialog.result == 'skip':
        plt.close(fig)
        root.destroy()
        return None
    elif dialog.result is None:
        plt.close(fig)
        root.destroy()
        return None
    else:
        plt.close(fig)
        root.destroy()
        return dialog.result

def visualize_missing_only_boxes_with_edit(image_path, json_paths, base_name):
    """
    이미지에 모든 JSON 파일의 바운딩 박스를 그리고, 오류 발생시 대화형 수정 도구 실행
    
    Args:
        image_path: 원본 이미지 경로
        json_paths: JSON 파일 경로 리스트
        base_name: 파일 기본 이름
    
    Returns:
        수정된 JSON 파일 개수
    """
    # 이미지 로드
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        OpLog(f"이미지 로드 실패 {image_path}: {e}", bLines=True)
        return 0
    
    draw = ImageDraw.Draw(img)
    
    # 폰트 설정
    try:
        font = ImageFont.truetype("malgun.ttf", 20)
        small_font = ImageFont.truetype("malgun.ttf", 14)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # 색상 팔레트
    colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
    
    all_dl_names = []
    modified_count = 0
    
    # 오류가 있는 JSON들을 나중에 처리하기 위한 리스트
    error_jsons = []
    
    # 1차: 성공한 JSON만 먼저 처리 (오류 없는 것들)
    for json_idx, json_path in enumerate(json_paths):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # dl_name 추출
            dl_name = "Unknown"
            if 'images' in data and len(data['images']) > 0:
                dl_name = data['images'][0].get('dl_name', 'Unknown')
            all_dl_names.append(dl_name)
            
            # annotations 처리
            has_error = False
            if 'annotations' in data:
                color = colors[json_idx % len(colors)]
                
                for anno_idx, anno in enumerate(data['annotations']):
                    if 'bbox' in anno:
                        bbox = anno['bbox']
                        
                        # bbox 형식 검사
                        if len(bbox) != 4:
                            # 오류가 있는 경우 나중에 처리
                            has_error = True
                            error_jsons.append({
                                'json_idx': json_idx,
                                'json_path': json_path,
                                'data': data,
                                'color': color
                            })
                            OpLog(f"bbox 오류 발견 (나중에 수정): {json_path}", bLines=True)
                            break  # 이 JSON은 나중에 처리
                        
                        x, y, w, h = bbox
                        
                        # 바운딩 박스 그리기
                        draw.rectangle(
                            [(x, y), (x + w, y + h)],
                            outline=color,
                            width=3
                        )
                        
                        # 카테고리 ID 표시
                        if 'category_id' in anno:
                            cat_id = anno['category_id']
                            draw.text(
                                (x, y - 20),
                                f"Cat:{cat_id}",
                                fill=color,
                                font=small_font
                            )
                
                # 오류가 없었던 경우만 로그
                if not has_error:
                    OpLog(f"bbox 정상: {os.path.basename(json_path)}", bLines=True)
        
        except Exception as e:
            OpLog(f"JSON 파싱 오류 {json_path}: {e}", bLines=True)
            parse_err_log = os.path.join(os.path.dirname(LOG_FILE), "parseerr.log")
            with open(parse_err_log, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.datetime.now()}] JSON 파싱 오류: {json_path}\n")
                f.write(f"  오류 내용: {e}\n\n")
            continue
    
    # 2차: 오류가 있었던 JSON들을 대화형으로 수정
    if error_jsons:
        OpLog(f"\n📝 bbox 오류가 있는 JSON {len(error_jsons)}개를 수정합니다...", bLines=True)
        
        for error_info in error_jsons:
            json_idx = error_info['json_idx']
            json_path = error_info['json_path']
            data = error_info['data']
            color = error_info['color']
            
            # dl_name 추출
            dl_name = "Unknown"
            if 'images' in data and len(data['images']) > 0:
                dl_name = data['images'][0].get('dl_name', 'Unknown')
            
            json_modified = False
            
            if 'annotations' in data:
                for anno_idx, anno in enumerate(data['annotations']):
                    if 'bbox' in anno:
                        bbox = anno['bbox']
                        
                        if len(bbox) != 4:
                            OpLog(f"잘못된 bbox 형식 수정 시작 [dl_name: {dl_name}]: {bbox} in {json_path}", bLines=True)
                            
                            # 대화형 수정 도구 실행
                            new_bbox = interactive_bbox_editor(
                                image_path, 
                                json_path, 
                                {'bbox': bbox, 'anno_index': anno_idx},
                                dl_name
                            )
                            
                            if new_bbox:
                                # JSON 파일 수정
                                data['annotations'][anno_idx]['bbox'] = new_bbox
                                json_modified = True
                                OpLog(f"bbox 수정됨 [dl_name: {dl_name}]: {bbox} → {new_bbox}", bLines=True)
                                bbox = new_bbox
                                
                                # 수정된 bbox를 이미지에 그리기
                                x, y, w, h = bbox
                                draw.rectangle(
                                    [(x, y), (x + w, y + h)],
                                    outline=color,
                                    width=3
                                )
                                
                                if 'category_id' in anno:
                                    cat_id = anno['category_id']
                                    draw.text(
                                        (x, y - 20),
                                        f"Cat:{cat_id}",
                                        fill=color,
                                        font=small_font
                                    )
                            else:
                                OpLog(f"bbox 수정 건너뜀: {json_path}", bLines=True)
            
            # JSON 파일이 수정되었으면 저장
            if json_modified:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                OpLog(f"JSON 파일 저장됨: {json_path}", bLines=True)
                modified_count += 1
    
    # dl_name 텍스트를 이미지 상단에 표시
    if all_dl_names:
        y_pos = 10
        for idx, dl_name in enumerate(all_dl_names):
            color = colors[idx % len(colors)]
            text = f"[{idx+1}] {dl_name}"
            draw.text((10, y_pos), text, fill=color, font=font)
            y_pos += 30
    
    # 이미지 저장
    output_path = os.path.join(MISSING_ONLY_ANNOTAIIONS_IMG, f"{base_name}.box.png")
    try:
        img.save(output_path)
        OpLog(f"바운딩 박스 이미지 저장: {base_name}.box.png (JSON {len(json_paths)}개, 수정 {modified_count}개)", bLines=True)
    except Exception as e:
        OpLog(f"이미지 저장 실패 {output_path}: {e}", bLines=True)
    
    return modified_count

def process_missing_only_visualization_new():
    """
    MISSING_ONLY_ANNOTAIIONS_IMG의 이미지와 MISSING_ONLY_ANNOTATIONS_ADD의 JSON을 읽어서
    바운딩 박스 시각화를 수행하고, 오류 발생시 대화형 수정 도구 실행
    """
    OpLog("MISSING_ONLY 바운딩 박스 시각화 시작 (대화형 수정 모드)", bLines=True)
    
    # 1. MISSING_ONLY_ANNOTAIIONS_IMG의 이미지 파일 수집
    if not os.path.exists(MISSING_ONLY_ANNOTAIIONS_IMG):
        OpLog(f"이미지 디렉토리가 존재하지 않습니다: {MISSING_ONLY_ANNOTAIIONS_IMG}", bLines=True)
        return
    
    image_files = {}
    for img_file in os.listdir(MISSING_ONLY_ANNOTAIIONS_IMG):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) and not img_file.endswith('.box.png'):
            base_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(MISSING_ONLY_ANNOTAIIONS_IMG, img_file)
            image_files[base_name] = img_path
    
    OpLog(f"이미지 파일: {len(image_files)}개", bLines=True)
    
    # 2. MISSING_ONLY_ANNOTATIONS_ADD의 JSON 파일 수집
    if not os.path.exists(MISSING_ONLY_ANNOTATIONS_ADD):
        OpLog(f"JSON 디렉토리가 존재하지 않습니다: {MISSING_ONLY_ANNOTATIONS_ADD}", bLines=True)
        return
    
    visualized_count = 0
    skipped_count = 0
    total_modified = 0
    
    # 3. 각 이미지에 대해 처리
    for base_name, img_path in image_files.items():
        # MISSING_ONLY_ANNOTATIONS_ADD/{base_name}/ 디렉토리 확인
        anno_dir = os.path.join(MISSING_ONLY_ANNOTATIONS_ADD, base_name)
        
        if not os.path.exists(anno_dir):
            OpLog(f"JSON 디렉토리 없음: {base_name}", bLines=True)
            skipped_count += 1
            continue
        
        # 모든 서브 디렉토리(01, 02, 03...)에서 JSON 파일 수집
        json_files = []
        for subdir in sorted(os.listdir(anno_dir)):
            subdir_path = os.path.join(anno_dir, subdir)
            if os.path.isdir(subdir_path):
                # 해당 디렉토리에서 {base_name}.json 찾기
                json_filename = f"{base_name}.json"
                json_path = os.path.join(subdir_path, json_filename)
                if os.path.exists(json_path):
                    json_files.append(json_path)
        
        if json_files:
            try:
                modified = visualize_missing_only_boxes_with_edit(img_path, json_files, base_name)
                visualized_count += 1
                total_modified += modified
            except Exception as e:
                OpLog(f"바운딩 박스 시각화 오류 {base_name}: {e}", bLines=True)
        else:
            OpLog(f"JSON 파일 없음: {base_name}", bLines=True)
            skipped_count += 1
    
    OpLog(f"시각화 완료:", bLines=True)
    OpLog(f"  - 시각화 성공: {visualized_count}개", bLines=True)
    OpLog(f"  - JSON 수정: {total_modified}개", bLines=True)
    OpLog(f"  - 건너뜀: {skipped_count}개", bLines=True)
    OpLog(f"  - 저장 위치: {MISSING_ONLY_ANNOTAIIONS_IMG}", bLines=True)
def process_All_edit(image_dir, anno_dir):
    """
    image_dir의 모든 이미지 파일을 순회하며 process_missing_only_visualization_Ex를 호출합니다.
    
    Args:
        image_dir: 이미지가 있는 디렉토리 경로
        anno_dir: annotation이 있는 디렉토리 경로
    """
    OpLog(f"process_All_edit 시작: image_dir={image_dir}, anno_dir={anno_dir}", bLines=True)
    
    if not os.path.exists(image_dir):
        OpLog(f"process_All_edit: 이미지 디렉토리 없음 -> {image_dir}", bLines=True)
        return
    
    # image_dir의 모든 이미지 파일 수집
    image_files = []
    for img_file in os.listdir(image_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) and not img_file.endswith('.box.png'):
            img_path = os.path.join(image_dir, img_file)
            image_files.append(img_path)
    
    OpLog(f"process_All_edit: 처리할 이미지 {len(image_files)}개 발견", bLines=True)
    
    # 각 이미지에 대해 process_missing_only_visualization_Ex 호출
    for idx, img_path in enumerate(image_files, 1):
        OpLog(f"process_All_edit: [{idx}/{len(image_files)}] {os.path.basename(img_path)} 처리 중...", bLines=True)
        try:
            process_missing_only_visualization_Ex(img_path, anno_dir)
        except Exception as e:
            OpLog(f"process_All_edit: 처리 실패 -> {os.path.basename(img_path)}: {e}", bLines=True)
    
    OpLog(f"process_All_edit 완료: {len(image_files)}개 이미지 처리됨", bLines=True)


def process_missing_only_visualization_Ex(filepath,anno_dir):
    """
    주어진 이미지(filepath)에 대해 박스를 그리고, 화면에 표시한 뒤
    콤보박스로 어떤 서브디렉토리(01/02/03...)의 JSON을 선택할지 확인하게 합니다.
    선택된 JSON의 `dl_name`과 bbox를 수정할 수 있으며, 추가 버튼으로 새로운
    순번 디렉토리를 생성하고 최소한의 json( dl_name + bbox )을 저장할 수 있습니다.

    Args:
        filepath: 이미지 파일명 또는 절대경로 (예: "A.png" 또는 full path)
    """
    img_path = filepath

    if not os.path.exists(img_path):
        OpLog(f"이미지 파일을 찾을 수 없습니다: {img_path}", bLines=True)
        return False

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    if not os.path.exists(anno_dir):
        OpLog(f"JSON 디렉토리가 존재하지 않아 생성합니다: {anno_dir}", bLines=True)
        os.makedirs(anno_dir, exist_ok=True)

    # anno_dir를 재귀적으로 검사하여 base_name.json 파일 수집
    json_entries = []  # list of tuples (subdir, json_path, dl_name)
    json_name = f"{base_name}.json"
    
    for root, dirs, files in os.walk(anno_dir):
        if json_name in files:
            jp = os.path.join(root, json_name)
            # anno_dir 기준 상대경로를 subdir로 사용
            rel_path = os.path.relpath(root, anno_dir)
            try:
                with open(jp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                dl = 'Unknown'
                if 'images' in data and len(data['images'])>0:
                    dl = data['images'][0].get('dl_name', 'Unknown')
                json_entries.append((rel_path, jp, dl))
            except Exception as e:
                json_entries.append((rel_path, jp, 'PARSE_ERR'))
    
    # 준비된 콤보 목록 (표시용에 dl_name 포함)
    combo_items = [f"{sd} - {dl}" for sd, jp, dl in json_entries]

    # 이미지를 표시할 간단한 Tkinter UI
    root = tk.Tk()
    root.title(f"BBox Editor: {base_name}")
    root.geometry('900x700')

    # 좌: 이미지, 우: 컨트롤
    left = tk.Frame(root); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    right = tk.Frame(root, width=340); right.pack(side=tk.RIGHT, fill=tk.Y)

    # 이미지 표시 (PIL->ImageTk)
    img = Image.open(img_path).convert('RGB')
    disp = img.copy()
    # Pillow 10+ removed Image.ANTIALIAS; use Resampling enum when available
    if hasattr(Image, 'Resampling'):
        resample_filter = Image.Resampling.LANCZOS
    else:
        # older versions expose LANCZOS at module level
        resample_filter = getattr(Image, 'LANCZOS', None)
    if resample_filter is not None:
        disp.thumbnail((800,800), resample_filter)
    else:
        # fallback: call thumbnail without explicit resample
        disp.thumbnail((800,800))
    imgtk = ImageTk.PhotoImage(disp)
    img_label = tk.Label(left, image=imgtk)
    img_label.image = imgtk
    img_label.pack(fill=tk.BOTH, expand=True)

    # 콤보박스와 dl_name, bbox 필드
    from tkinter import ttk
    ttk.Label(right, text='JSON 선택(디렉토리 - dl_name):').pack(anchor=tk.W, padx=8, pady=(8,0))
    combo = ttk.Combobox(right, values=['(없음)'] + combo_items, state='readonly')
    combo.current(0)
    combo.pack(fill=tk.X, padx=8, pady=4)

    tk.Label(right, text='dl_name:').pack(anchor=tk.W, padx=8)
    dl_entry = tk.Entry(right); dl_entry.pack(fill=tk.X, padx=8, pady=2)

    tk.Label(right, text='bbox (left, top, width, height):').pack(anchor=tk.W, padx=8)
    bbox_frame = tk.Frame(right); bbox_frame.pack(fill=tk.X, padx=8)
    left_e = tk.Entry(bbox_frame, width=6); left_e.grid(row=0,column=0,padx=2)
    top_e = tk.Entry(bbox_frame, width=6); top_e.grid(row=0,column=1,padx=2)
    w_e = tk.Entry(bbox_frame, width=6); w_e.grid(row=0,column=2,padx=2)
    h_e = tk.Entry(bbox_frame, width=6); h_e.grid(row=0,column=3,padx=2)

    status = tk.Label(right, text='')
    status.pack(anchor=tk.W, padx=8, pady=(6,0))

    # 선택 변경시 로드
    def on_select(ev=None):
        sel = combo.get()
        if sel == '(없음)':
            dl_entry.delete(0, tk.END); left_e.delete(0, tk.END); top_e.delete(0, tk.END); w_e.delete(0, tk.END); h_e.delete(0, tk.END)
            status.config(text='선택 없음')
            return
        # sel is like '01 - dl'
        sd = sel.split(' - ')[0]
        jp = os.path.join(anno_dir, sd, f"{base_name}.json")
        if not os.path.exists(jp):
            status.config(text='JSON 없음')
            return
        try:
            with open(jp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            dl = 'Unknown'
            if 'images' in data and len(data['images'])>0:
                dl = data['images'][0].get('dl_name','Unknown')
            dl_entry.delete(0, tk.END); dl_entry.insert(0, dl)
            if 'annotations' in data and len(data['annotations'])>0:
                bbox = data['annotations'][0].get('bbox', [])
                if len(bbox)==4:
                    left_e.delete(0, tk.END); left_e.insert(0,str(bbox[0]))
                    top_e.delete(0, tk.END); top_e.insert(0,str(bbox[1]))
                    w_e.delete(0, tk.END); w_e.insert(0,str(bbox[2]))
                    h_e.delete(0, tk.END); h_e.insert(0,str(bbox[3]))
            status.config(text=f'로드됨: {jp}')
        except Exception as e:
            status.config(text=f'파싱 오류: {e}')

    combo.bind('<<ComboboxSelected>>', on_select)

    def preview():
        # Draw a single bbox based on current input fields (left,top,width,height)
        try:
            lx = int(left_e.get()); ty = int(top_e.get()); ww = int(w_e.get()); hh = int(h_e.get())
        except Exception:
            messagebox.showerror('입력 오류','좌표를 정수로 입력하세요', parent=root)
            return
        try:
            ow, oh = img.size; dw, dh = disp.size
            sx = dw/ow; sy = dh/oh
        except Exception:
            messagebox.showerror('이미지 오류','이미지 크기 정보를 읽을 수 없습니다', parent=root); return

        rx = int(lx * sx); ry = int(ty * sy); rw = int(ww * sx); rh = int(hh * sy)
        canvas_img = disp.copy(); draw_img = ImageDraw.Draw(canvas_img)
        draw_img.rectangle([rx, ry, rx+rw, ry+rh], outline='green', width=3)
        label = dl_entry.get().strip() or 'Unknown'
        text_x = rx; text_y = max(0, ry - 14)
        tw = max(30, len(label) * 7)
        draw_img.rectangle([text_x, text_y, text_x + tw, text_y + 14], fill='black')
        draw_img.text((text_x + 2, text_y), label, fill='white')
        imgtk2 = ImageTk.PhotoImage(canvas_img)
        img_label.configure(image=imgtk2); img_label.image = imgtk2

    def draw_all():
        # Draw all boxes from json_entries at dialog open or when requested
        try:
            ow, oh = img.size; dw, dh = disp.size
            sx = dw/ow; sy = dh/oh
        except Exception:
            return
        canvas_img = disp.copy(); draw_img = ImageDraw.Draw(canvas_img)
        colors = ['red','lime','blue','yellow','magenta','cyan','orange','purple','white']
        sel = combo.get()
        selected_sd = None
        if sel and sel != '(없음)':
            selected_sd = sel.split(' - ')[0]
        for idx, (sd, jp, dl) in enumerate(json_entries):
            color = colors[idx % len(colors)]
            try:
                if not os.path.exists(jp):
                    continue
                with open(jp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'annotations' in data and len(data['annotations'])>0:
                    bbox = data['annotations'][0].get('bbox', [])
                    if len(bbox) != 4:
                        continue
                    lx, ty, ww, hh = bbox
                    rx = int(lx * sx); ry = int(ty * sy); rw = int(ww * sx); rh = int(hh * sy)
                    width = 4 if sd == selected_sd else 2
                    draw_img.rectangle([rx, ry, rx+rw, ry+rh], outline=color, width=width)
                    label = dl or 'Unknown'
                    text_x = rx; text_y = max(0, ry - 14)
                    tw = max(30, len(label) * 7)
                    draw_img.rectangle([text_x, text_y, text_x + tw, text_y + 14], fill='black')
                    draw_img.text((text_x + 2, text_y), label, fill='white')
            except Exception:
                continue
        imgtk2 = ImageTk.PhotoImage(canvas_img)
        img_label.configure(image=imgtk2); img_label.image = imgtk2

    def save():
        sel = combo.get()
        if sel == '(없음)':
            messagebox.showerror('선택 오류','수정할 JSON을 선택하세요', parent=root); return
        sd = sel.split(' - ')[0]
        target_dir = os.path.join(anno_dir, sd)
        os.makedirs(target_dir, exist_ok=True)
        jp = os.path.join(target_dir, f"{base_name}.json")
        dl = dl_entry.get().strip() or 'Unknown'
        try:
            lx = int(left_e.get()); ty = int(top_e.get()); ww = int(w_e.get()); hh = int(h_e.get())
        except Exception:
            messagebox.showerror('입력 오류','좌표는 정수여야 합니다', parent=root); return
        data = {'images':[{'file_name': os.path.basename(img_path), 'dl_name': dl}], 'annotations':[{'bbox':[lx,ty,ww,hh]}], 'categories':[]}
        with open(jp,'w',encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        status.config(text=f'저장됨: {jp}')
        # 갱신된 dl_name을 콤보에 반영
        idx = combo.get()
        combo['values'] = ['(없음)'] + [f"{sd} - {dl if sd==sd else ''}" for sd,_,dl in json_entries]

    def add_new():
        # 다음 순번 생성
        existing = [d for d in sorted(os.listdir(anno_dir)) if os.path.isdir(os.path.join(anno_dir,d))]
        next_idx = 1
        for d in existing:
            try:
                next_idx = max(next_idx, int(d)+1)
            except:
                pass
        new_sd = f"{next_idx:02d}"
        new_dir = os.path.join(anno_dir, new_sd); os.makedirs(new_dir, exist_ok=True)
        jp = os.path.join(new_dir, f"{base_name}.json")
        dl = dl_entry.get().strip() or 'Unknown'
        try:
            lx = int(left_e.get()); ty = int(top_e.get()); ww = int(w_e.get()); hh = int(h_e.get())
        except Exception:
            messagebox.showerror('입력 오류','좌표는 정수여야 합니다', parent=root); return
        data = {'images':[{'file_name': os.path.basename(img_path), 'dl_name': dl}], 'annotations':[{'bbox':[lx,ty,ww,hh]}], 'categories':[]}
        with open(jp,'w',encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        status.config(text=f'추가됨: {jp}')
        # 콤보 항목 갱신
        json_entries.append((new_sd, jp, dl))
        combo['values'] = ['(없음)'] + [f"{sd} - {dl}" for sd,_,dl in json_entries]
        combo.current(len(json_entries))

    # 버튼
    tk.Button(right, text='그리기', command=preview, bg='#4CAF50', fg='white').pack(fill=tk.X, padx=8, pady=6)
    tk.Button(right, text='저장', command=save, bg='#2196F3', fg='white').pack(fill=tk.X, padx=8, pady=6)
    tk.Button(right, text='추가(Json 추가)', command=add_new, bg='#9C27B0', fg='white').pack(fill=tk.X, padx=8, pady=6)
    tk.Button(right, text='닫기', command=root.destroy, bg='#f44336', fg='white').pack(fill=tk.X, padx=8, pady=6)

    # 초기 콤보 채우기
    combo['values'] = ['(없음)'] + [f"{sd} - {dl}" for sd,_,dl in json_entries]
    if len(json_entries)>0:
        combo.current(1); on_select()
    # 다이얼로그가 뜰 때 모든 박스를 먼저 그림
    try:
        draw_all()
    except Exception:
        pass

    root.mainloop()
    return True


def className_dlName(file_path, bNew = True):
    result_file = r"D:\01.project\EntryPrj\data\oraldrug\className_dl_Name.csv"
    OpLog(f"className_dlName 시작: path={file_path}, bNew={bNew}", bLines=True)
    # Collect unique (class_name, dl_name) pairs
    results = []
    classes_seen = set()
    for root, dirs, files in os.walk(file_path):
        # treat the immediate directory name as class_name
        class_name = os.path.basename(root)
        if class_name in classes_seen:
            # skip scanning this directory for efficiency
            continue
        for fname in files:
            if not fname.lower().endswith('.json'):
                continue
            jp = os.path.join(root, fname)
            try:
                with open(jp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                continue
            dl_name = None
            if 'images' in data and isinstance(data['images'], list) and len(data['images'])>0:
                dl_name = data['images'][0].get('dl_name', '')
                if isinstance(dl_name, str):
                    dl_name = dl_name.strip()
            if not dl_name:
                continue
            # if this class_name already recorded, skip (efficiency requirement)
            if class_name in classes_seen:
                break
            results.append((class_name, dl_name, jp))
            classes_seen.add(class_name)
            # once we found one json for this class, skip other files in this directory
            break

    # write results to CSV (bNew -> overwrite; otherwise merge/update by class_name)
    try:
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        if bNew or (not os.path.exists(result_file)):
            with open(result_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['class_name', 'dl_name'])
                for class_name, dl_name, jp in results:
                    writer.writerow([class_name, dl_name])
            OpLog(f"className_dlName: 새 파일로 저장됨 -> {result_file} (entries={len(results)})", bLines=True)
        else:
            # load existing
            existing = {}
            try:
                with open(result_file, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        cn = row.get('class_name','').strip()
                        existing[cn] = row.get('dl_name','').strip()
            except Exception:
                existing = {}

            added = 0; updated = 0
            for class_name, dl_name, jp in results:
                if class_name in existing:
                    if existing[class_name] != dl_name:
                        existing[class_name] = dl_name
                        updated += 1
                else:
                    existing[class_name] = dl_name
                    added += 1

            with open(result_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['class_name', 'dl_name'])
                for cn in sorted(existing.keys()):
                    writer.writerow([cn, existing[cn]])

            OpLog(f"className_dlName: 병합 저장 -> {result_file} (added={added}, updated={updated}, total={len(existing)})", bLines=True)
    except Exception as e:
        OpLog(f"className_dlName: 저장 실패 -> {e}", bLines=True)
    return results


def change_category_id(json_path):
    """
    주어진 json_path를 재귀적으로 검사하여 JSON 파일을 읽고,
    dl_idx와 category_id가 다르고 category_id=1이라면
    category_id를 dl_idx로 수정하고 저장.
    """
    OpLog(f"change_category_id 시작: path={json_path}", bLines=True)
    
    # json_path를 재귀적으로 검사하여 JSON 파일 수정
    updated_count = 0
    skipped_count = 0
    error_count = 0
    
    for root, dirs, files in os.walk(json_path):
        for fname in files:
            if not fname.lower().endswith('.json'):
                continue
            jp = os.path.join(root, fname)
            try:
                with open(jp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                OpLog(f"change_category_id: JSON 읽기 실패 -> {jp}: {e}")
                error_count += 1
                continue
            
            # dl_idx 추출
            dl_idx = None
            if 'images' in data and isinstance(data['images'], list) and len(data['images']) > 0:
                dl_idx = data['images'][0].get('dl_idx')
            
            if dl_idx is None:
                skipped_count += 1
                continue
            
            # annotations의 category_id 업데이트 (category_id=1이고 dl_idx와 다른 경우만)
            modified = False
            if 'annotations' in data and isinstance(data['annotations'], list):
                for ann in data['annotations']:
                    if isinstance(ann, dict):
                        category_id = ann.get('category_id')
                        # category_id가 1이고, dl_idx와 다른 경우에만 수정
                        if category_id == 1 and category_id != dl_idx:
                            ann['category_id'] = dl_idx
                            modified = True
            
            if modified:
                try:
                    with open(jp, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    updated_count += 1
                except Exception as e:
                    OpLog(f"change_category_id: JSON 저장 실패 -> {jp}: {e}")
                    error_count += 1
            else:
                skipped_count += 1
    
    OpLog(f"change_category_id 완료: updated={updated_count}, skipped={skipped_count}, errors={error_count}", bLines=True)
    return updated_count



def MoveImageAnnotaoin(sourceDir, targetDir):
    """
    sourceDir/train_images의 이미지와 sourceDir/train_annotations의 JSON을 targetDir로 복사합니다.
    이미지 파일명을 기준으로 매칭되는 JSON을 재귀 검색하여 함께 복사합니다.
    원본 파일은 sourceDir에 보존됩니다.
    
    Args:
        sourceDir: 베이스 디렉토리 (train_images, train_annotations 포함)
        targetDir: 이미지와 JSON을 복사할 대상 디렉토리
    """
    OpLog(f"MoveImageAnnotaoin 시작: source={sourceDir}, target={targetDir}", bLines=True)
    
    source_img_dir = os.path.join(sourceDir, "train_images")
    source_anno_dir = os.path.join(sourceDir, "train_annotations")
    
    if not os.path.exists(source_img_dir):
        OpLog(f"MoveImageAnnotaoin: 소스 이미지 디렉토리 없음 -> {source_img_dir}", bLines=True)
        return
    
    if not os.path.exists(source_anno_dir):
        OpLog(f"MoveImageAnnotaoin: 소스 어노테이션 디렉토리 없음 -> {source_anno_dir}", bLines=True)
        return
    
    # targetDir 생성 (train_images/train_annotations 구조 유지)
    os.makedirs(targetDir, exist_ok=True)
    target_img_dir = os.path.join(targetDir, "train_images")
    target_anno_dir = os.path.join(targetDir, "train_annotations")
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_anno_dir, exist_ok=True)
    
    copied_images = 0
    copied_jsons = 0
    no_annotation = 0
    
    # source_img_dir의 모든 이미지 파일 처리
    for img_file in os.listdir(source_img_dir):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        
        img_src = os.path.join(source_img_dir, img_file)
        img_dst = os.path.join(target_img_dir, img_file)
        
        # 해당 이미지의 JSON 파일 찾기 (source_anno_dir 재귀 검색)
        base_name = os.path.splitext(img_file)[0]
        json_name = f"{base_name}.json"
        found_jsons = []
        
        for root, dirs, files in os.walk(source_anno_dir):
            if json_name in files:
                found_jsons.append(os.path.join(root, json_name))
        
        if not found_jsons:
            no_annotation += 1
            OpLog(f"MoveImageAnnotaoin: JSON 없음 -> {img_file}")
            continue
        
        # JSON이 1개 이상 있을 때만 이미지 복사
        try:
            shutil.copy2(img_src, img_dst)
            copied_images += 1
        except Exception as e:
            OpLog(f"MoveImageAnnotaoin: 이미지 복사 실패 -> {img_file}: {e}")
            continue
        
        # 발견된 모든 JSON을 대상 디렉토리로 복사 (서브디렉토리 구조 유지)
        for json_src in found_jsons:
            # source_anno_dir 기준 상대경로 계산
            rel_path = os.path.relpath(json_src, source_anno_dir)
            json_dst = os.path.join(target_anno_dir, rel_path)
            os.makedirs(os.path.dirname(json_dst), exist_ok=True)
            
            try:
                shutil.copy2(json_src, json_dst)
                copied_jsons += 1
            except Exception as e:
                OpLog(f"MoveImageAnnotaoin: JSON 복사 실패 -> {json_src}: {e}")
    
    OpLog(f"MoveImageAnnotaoin 완료: images={copied_images}, jsons={copied_jsons}, no_anno={no_annotation}", bLines=True)
    return copied_images, copied_jsons


def MoveOnlyImage(sourceDir, targetDir):
    """
    sourceDir의 train_images에서 annotation이 하나도 없는 이미지를 찾아 targetDir/train_images로 이동합니다.
    train_annotations를 재귀적으로 검사하여 해당 이미지의 JSON이 존재하지 않는 경우만 이동합니다.
    
    Args:
        sourceDir: 베이스 디렉토리 (train_images, train_annotations 포함)
        targetDir: 이미지를 이동할 대상 디렉토리
    """
    OpLog(f"MoveOnlyImage 시작: source={sourceDir}, target={targetDir}", bLines=True)
    
    source_img_dir = os.path.join(sourceDir, "train_images")
    source_anno_dir = os.path.join(sourceDir, "train_annotations")
    
    if not os.path.exists(source_img_dir):
        OpLog(f"MoveOnlyImage: 소스 이미지 디렉토리 없음 -> {source_img_dir}", bLines=True)
        return
    
    if not os.path.exists(source_anno_dir):
        OpLog(f"MoveOnlyImage: 소스 어노테이션 디렉토리 없음 -> {source_anno_dir}", bLines=True)
        return
    
    # targetDir/train_images 생성
    os.makedirs(targetDir, exist_ok=True)
    target_img_dir = os.path.join(targetDir, "train_images")
    os.makedirs(target_img_dir, exist_ok=True)
    
    moved_count = 0
    has_annotation_count = 0
    
    # source_img_dir의 모든 이미지 파일 처리
    for img_file in os.listdir(source_img_dir):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        
        # 해당 이미지의 JSON 파일 찾기 (source_anno_dir 재귀 검색)
        base_name = os.path.splitext(img_file)[0]
        json_name = f"{base_name}.json"
        has_annotation = False
        
        for root, dirs, files in os.walk(source_anno_dir):
            if json_name in files:
                has_annotation = True
                break
        
        if has_annotation:
            has_annotation_count += 1
            continue
        
        # annotation이 없으면 복사 (원본 보존)
        img_src = os.path.join(source_img_dir, img_file)
        img_dst = os.path.join(target_img_dir, img_file)
        
        try:
            shutil.copy2(img_src, img_dst)
            moved_count += 1
        except Exception as e:
            OpLog(f"MoveOnlyImage: 이미지 복사 실패 -> {img_file}: {e}")
    
    OpLog(f"MoveOnlyImage 완료: copied={moved_count}, has_anno={has_annotation_count}", bLines=True)
    return moved_count

#  
def GetStatisticDrug(file_path):
    """
    file_path 밑의 모든 JSON 파일을 읽어서 dl_idx와 dl_name별 개수를 집계하여 반환합니다.
    
    Args:
        file_path: JSON 파일들이 있는 디렉토리 경로
    
    Returns:
        tuple: (stats, json_count)
            - stats: {(dl_idx, dl_name): count} 딕셔너리
            - json_count: 처리한 JSON 파일 개수
    """
    # dl_idx, dl_name별 카운트를 저장할 딕셔너리
    # key: (dl_idx, dl_name), value: count
    stats = {}
    
    # file_path 밑의 모든 JSON 파일 재귀적으로 검색
    json_count = 0
    for root, dirs, files in os.walk(file_path):
        for fname in files:
            if not fname.lower().endswith('.json'):
                continue
            
            json_path = os.path.join(root, fname)
            json_count += 1
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # images 배열에서 dl_idx와 dl_name 추출
                if 'images' in data and isinstance(data['images'], list):
                    for img_info in data['images']:
                        dl_idx = img_info.get('dl_idx', '')
                        dl_name = img_info.get('dl_name', '')
                        
                        if dl_idx != '' and dl_name:
                            key = (dl_idx, dl_name)
                            stats[key] = stats.get(key, 0) + 1
                            
            except Exception as e:
                OpLog(f"GetStatisticDrug: JSON 읽기 실패 -> {json_path}: {e}")
                continue
    
    return stats, json_count

def MergeStatisticDrug(*stats_list):
    """
    여러 GetStatisticDrug 결과를 병합하여 category_id별 총 개수를 반환합니다.
    
    Args:
        *stats_list: GetStatisticDrug이 반환한 stats 딕셔너리들
    
    Returns:
        dict: {category_id: (dl_name, count)} 형태의 딕셔너리
    """
    merged = {}  # {category_id: {'dl_name': str, 'count': int}}
    
    for stats in stats_list:
        for (dl_idx, dl_name), count in stats.items():
            try:
                category_id = int(dl_idx)
            except (ValueError, TypeError):
                OpLog(f"MergeStatisticDrug: dl_idx 변환 실패 -> {dl_idx}", bLines=False)
                continue
            
            if category_id not in merged:
                merged[category_id] = {'dl_name': dl_name, 'count': 0}
            
            merged[category_id]['count'] += count
            # dl_name이 다를 경우 첫 번째 것을 유지
            if not merged[category_id]['dl_name']:
                merged[category_id]['dl_name'] = dl_name
    
    # {category_id: (dl_name, count)} 형태로 변환
    result = {cat_id: (info['dl_name'], info['count']) for cat_id, info in merged.items()}
    
    return result

def Statistic_drug(file_path, csv_file_path):
    """
    file_path 밑의 모든 JSON 파일을 읽어서 dl_idx와 dl_name별 개수를 집계하고
    csv_file_path에 기록합니다.
    
    Args:
        file_path: JSON 파일들이 있는 디렉토리 경로
        csv_file_path: 결과를 저장할 CSV 파일 경로
    """
    OpLog(f"Statistic_drug 시작: file_path={file_path}", bLines=True)
    
    # GetStatisticDrug을 사용하여 통계 수집
    stats, json_count = GetStatisticDrug(file_path)
    
    # CSV 파일로 저장
    try:
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        
        # dl_idx 기준으로 정렬
        sorted_stats = sorted(stats.items(), key=lambda x: (x[0][0], x[0][1]))
        
        with open(csv_file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['dl_idx', 'dl_name', 'count'])
            
            for (dl_idx, dl_name), count in sorted_stats:
                writer.writerow([dl_idx, dl_name, count])
        
        OpLog(f"Statistic_drug 완료: json_files={json_count}, unique_drugs={len(stats)}, csv={csv_file_path}", bLines=True)
        
    except Exception as e:
        OpLog(f"Statistic_drug: CSV 저장 실패 -> {csv_file_path}: {e}")
        raise

def augment_balance_classes(anno_dir, img_dir, target_dir, method='flip'):
    """
    JSON을 모두 읽어서 dl_idx별로 개수를 세고,
    가장 높은 숫자의 dl_idx만큼 다른 클래스를 증강합니다.
    
    Args:
        anno_dir: annotation 디렉토리 경로
        img_dir: 이미지 디렉토리 경로
        target_dir: 증강된 이미지/annotation을 저장할 디렉토리
        method: 증강 방법 ('flip', 'rotate', 'brightness', 'all')
    
    Returns:
        증강된 이미지/annotation 개수
    """
    OpLog(f"augment_balance_classes 시작: anno_dir={anno_dir}, method={method}", bLines=True)
    
    # 1단계: dl_idx별 개수 파악
    class_stats = {}  # {(dl_idx, dl_name): [json_paths]}
    
    for root, dirs, files in os.walk(anno_dir):
        for fname in files:
            if not fname.lower().endswith('.json'):
                continue
            
            json_path = os.path.join(root, fname)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'images' in data and isinstance(data['images'], list) and len(data['images']) > 0:
                    img_info = data['images'][0]
                    dl_idx = img_info.get('dl_idx', '')
                    dl_name = img_info.get('dl_name', '')
                    
                    if dl_idx != '' and dl_name:
                        key = (dl_idx, dl_name)
                        if key not in class_stats:
                            class_stats[key] = []
                        class_stats[key].append(json_path)
            except Exception as e:
                OpLog(f"augment_balance_classes: JSON 읽기 실패 -> {json_path}: {e}")
                continue
    
    # 2단계: 최대 개수 찾기
    if not class_stats:
        OpLog("augment_balance_classes: 클래스를 찾을 수 없습니다.")
        return 0
    
    max_count = max(len(paths) for paths in class_stats.values())
    OpLog(f"augment_balance_classes: 최대 클래스 샘플 수 = {max_count}")
    
    # 3단계: 각 클래스별로 증강
    total_augmented = 0
    os.makedirs(target_dir, exist_ok=True)
    target_img_dir = os.path.join(target_dir, 'train_images')
    target_anno_dir = os.path.join(target_dir, 'train_annotations')
    os.makedirs(target_img_dir, exist_ok=True)
    
    for (dl_idx, dl_name), json_paths in class_stats.items():
        current_count = len(json_paths)
        needed = max_count - current_count
        
        if needed <= 0:
            OpLog(f"클래스 {dl_name} (idx={dl_idx}): 증강 불필요 (현재={current_count})")
            continue
        
        OpLog(f"클래스 {dl_name} (idx={dl_idx}): {needed}개 증강 필요 (현재={current_count}/{max_count})")
        
        # 증강할 이미지 순환 선택
        augmented = 0
        for i in range(needed):
            src_json_path = json_paths[i % current_count]
            
            try:
                # JSON 읽기
                with open(src_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 이미지 파일명 추출
                if 'images' not in data or len(data['images']) == 0:
                    continue
                
                img_filename = data['images'][0].get('file_name', '')
                if not img_filename:
                    continue
                
                # 원본 이미지 찾기
                src_img_path = None
                for root, dirs, files in os.walk(img_dir):
                    if img_filename in files:
                        src_img_path = os.path.join(root, img_filename)
                        break
                
                if not src_img_path or not os.path.exists(src_img_path):
                    OpLog(f"이미지 파일 없음: {img_filename}")
                    continue
                
                # 이미지 로드 및 증강
                img = Image.open(src_img_path)
                img_base, img_ext = os.path.splitext(img_filename)
                
                # 증강 방법 적용
                augmented_img = None
                aug_suffix = ""
                
                if method in ['flip', 'all']:
                    if i % 3 == 0:
                        augmented_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                        aug_suffix = "_flip"
                    elif i % 3 == 1 and method == 'all':
                        augmented_img = img.transpose(Image.FLIP_TOP_BOTTOM)
                        aug_suffix = "_vflip"
                
                if augmented_img is None and method in ['rotate', 'all']:
                    angles = [90, 180, 270]
                    angle = angles[i % len(angles)]
                    augmented_img = img.rotate(angle, expand=True)
                    aug_suffix = f"_rot{angle}"
                
                if augmented_img is None and method in ['brightness', 'all']:
                    from PIL import ImageEnhance
                    enhancer = ImageEnhance.Brightness(img)
                    factor = 0.7 + (i % 5) * 0.15  # 0.7 ~ 1.3
                    augmented_img = enhancer.enhance(factor)
                    aug_suffix = f"_bright{int(factor*100)}"
                
                if augmented_img is None:
                    # 기본: 좌우 반전
                    augmented_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    aug_suffix = "_flip"
                
                # 증강된 이미지 저장
                new_img_filename = f"{img_base}{aug_suffix}_{i:04d}{img_ext}"
                new_img_path = os.path.join(target_img_dir, new_img_filename)
                augmented_img.save(new_img_path)
                
                # 증강된 JSON 생성 (bbox 좌표 조정 필요)
                new_data = json.loads(json.dumps(data))  # deep copy
                new_data['images'][0]['file_name'] = new_img_filename
                
                # bbox 좌표 조정 (좌우반전의 경우)
                if 'flip' in aug_suffix and 'annotations' in new_data:
                    img_width = augmented_img.width
                    for anno in new_data['annotations']:
                        if 'bbox' in anno and len(anno['bbox']) == 4:
                            x, y, w, h = anno['bbox']
                            # 좌우 반전: x' = img_width - x - w
                            anno['bbox'] = [img_width - x - w, y, w, h]
                
                # JSON 저장 (원본과 동일한 하위 구조 유지)
                rel_path = os.path.relpath(os.path.dirname(src_json_path), anno_dir)
                new_anno_subdir = os.path.join(target_anno_dir, rel_path)
                os.makedirs(new_anno_subdir, exist_ok=True)
                
                new_json_filename = f"{img_base}{aug_suffix}_{i:04d}.json"
                new_json_path = os.path.join(new_anno_subdir, new_json_filename)
                
                with open(new_json_path, 'w', encoding='utf-8') as f:
                    json.dump(new_data, f, ensure_ascii=False, indent=2)
                
                augmented += 1
                total_augmented += 1
                
            except Exception as e:
                OpLog(f"증강 실패: {src_json_path}: {e}")
                continue
        
        OpLog(f"클래스 {dl_name}: {augmented}개 증강 완료")
    
    OpLog(f"augment_balance_classes 완료: total_augmented={total_augmented}", bLines=True)
    return total_augmented

def TotalStatistic():
    file_path = r"D:\01.project\EntryPrj\data\oraldrug\1.drug_Image_annotation_allOK\train_annotations"
    states1, _ = GetStatisticDrug(file_path)
    print("=== states1 ===")
    print(states1)
    
    file_path = r"D:\01.project\EntryPrj\data\oraldrug\2.drug_no_image_ok_Anno\train_annotations"
    states2, _ = GetStatisticDrug(file_path)
    print("\n=== states2 ===")
    print(states2)
    
    file_path = r"D:\01.project\EntryPrj\data\oraldrug\3.drug_ok_Image_no_Anno\train_annotations"
    states3, _ = GetStatisticDrug(file_path)
    print("\n=== states3 ===")
    print(states3)
    
    # 세 개의 stats 병합
    merged = MergeStatisticDrug(states1, states2, states3)
    print("\n=== Merged Statistics (category_id: (dl_name, count)) ===")
    for cat_id in sorted(merged.keys()):
        dl_name, count = merged[cat_id]
        print(f"category_id {cat_id}: {dl_name} - {count}개")

def TotalStatisticByName():
    """
    세 개의 stage에서 통계를 수집하여 dl_name 기준으로 병합하여 출력합니다.
    
    Returns:
        dict: {dl_name: count} 형태의 딕셔너리
    """
    file_path = r"D:\01.project\EntryPrj\data\oraldrug\1.drug_Image_annotation_allOK\train_annotations"
    states1, _ = GetStatisticDrug(file_path)
    
    file_path = r"D:\01.project\EntryPrj\data\oraldrug\2.drug_no_image_ok_Anno\train_annotations"
    states2, _ = GetStatisticDrug(file_path)
    
    file_path = r"D:\01.project\EntryPrj\data\oraldrug\3.drug_ok_Image_no_Anno\train_annotations"
    states3, _ = GetStatisticDrug(file_path)
    
    # dl_name 기준으로 병합
    merged_by_name = {}
    
    for stats in [states1, states2, states3]:
        for (dl_idx, dl_name), count in stats.items():
            if dl_name not in merged_by_name:
                merged_by_name[dl_name] = 0
            merged_by_name[dl_name] += count
    
    # 결과 출력
    print("\n=== Merged Statistics By dl_name ===")
    for dl_name in sorted(merged_by_name.keys()):
        count = merged_by_name[dl_name]
        print(f"{dl_name}: {count}개")
    
    return merged_by_name

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


def GetIdx_CatID(annotations_dir):
    """
    annotation 디렉토리에서 YOLO class index와 category_id 매핑을 생성
    
    Args:
        annotations_dir: annotation 디렉토리 경로
    
    Returns:
        dict: {index: category_id} 형태의 딕셔너리
              예: {0: 1899, 1: 2482, 2: 3350, ...}
    """
    class_info = {}  # {category_id: count}
    
    # os.walk()로 모든 하위 디렉토리 재귀적으로 탐색
    for root, dirs, files in os.walk(annotations_dir):
        for json_file in files:
            if json_file.endswith(".json"):
                json_path = os.path.join(root, json_file)
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if "annotations" in data and len(data["annotations"]) > 0:
                        for ann in data["annotations"]:
                            category_id = ann.get("category_id")
                            if category_id is not None:
                                class_info[category_id] = class_info.get(category_id, 0) + 1
                                break  # 한 JSON에서 category_id 하나만 찾으면 됨
                except Exception as e:
                    continue
    
    # category_id를 정렬하여 순차적 인덱스 매핑 생성
    sorted_category_ids = sorted(class_info.keys())
    idx_to_catid = {idx: cat_id for idx, cat_id in enumerate(sorted_category_ids)}
    
    OpLog(f"GetIdx_CatID: {len(idx_to_catid)}개 클래스 매핑 생성", bLines=False)
    
    return idx_to_catid

def update_category_id(annoationDir, csvFile):
    """
    CSV 파일의 category_id를 idx_to_catid 매핑에 따라 업데이트
    
    Args:
        annoationDir: annotation 디렉토리 경로
        csvFile: 업데이트할 CSV 파일 경로
    
    설명:
        idx_to_catid의 인덱스 번호와 CSV의 category_id가 같으면,
        idx_to_catid의 카테고리 번호로 변경하여 저장
        예: idx_to_catid = {0: 1899, 1: 2482, 2: 3350}
            CSV의 category_id가 0이면 -> 1899로 변경
            CSV의 category_id가 1이면 -> 2482로 변경
    """
    # 인덱스 -> category_id 매핑 가져오기
    idx_to_catid = GetIdx_CatID(annoationDir)
    
    OpLog(f"update_category_id 시작: csvFile={csvFile}", bLines=True)
    OpLog(f"매핑 정보: {len(idx_to_catid)}개 항목", bLines=False)
    
    # CSV 파일 존재 확인
    if not os.path.exists(csvFile):
        OpLog(f"CSV 파일을 찾을 수 없습니다: {csvFile}", bLines=True)
        return
    
    # CSV 파일 읽기
    try:
        import csv
        rows = []
        updated_count = 0
        
        with open(csvFile, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            for row in reader:
                # category_id 컬럼이 있는지 확인
                if 'category_id' in row:
                    try:
                        old_cat_id = int(row['category_id'])
                        
                        # idx_to_catid에서 인덱스로 category_id 찾기
                        if old_cat_id in idx_to_catid:
                            new_cat_id = idx_to_catid[old_cat_id]
                            row['category_id'] = str(new_cat_id)
                            updated_count += 1
                            OpLog(f"업데이트: {old_cat_id} -> {new_cat_id}", bLines=False)
                    except (ValueError, KeyError) as e:
                        OpLog(f"변환 실패: {row.get('category_id')} - {e}", bLines=False)
                
                rows.append(row)
        
        # CSV 파일 저장
        with open(csvFile, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        OpLog(f"update_category_id 완료: {updated_count}개 업데이트, 저장={csvFile}", bLines=True)
        
    except Exception as e:
        OpLog(f"update_category_id 오류: {e}", bLines=True)
        raise


def changesubmission(submission_file, yaml_file):
    """
    Submission CSV 파일의 category_id를 인덱스에서 실제 값으로 변경
    
    Args:
        submission_file: submission CSV 파일 경로 (category_id가 인덱스로 저장되어 있음)
        yaml_file: YOLO YAML 파일 경로 (names 리스트에 실제 category_id가 저장되어 있음)
    
    설명:
        submission_file의 category_id는 0, 1, 2, ... (인덱스)
        yaml_file의 names는 ['1899', '2482', '3350', ...] (실제 category_id)
        인덱스를 사용해서 실제 category_id로 변환하여 저장
    """
    import yaml
    import csv
    
    OpLog(f"changesubmission 시작: submission={submission_file}, yaml={yaml_file}", bLines=True)
    
    # YAML 파일 존재 확인
    if not os.path.exists(yaml_file):
        OpLog(f"YAML 파일을 찾을 수 없습니다: {yaml_file}", bLines=True)
        return False
    
    # CSV 파일 존재 확인
    if not os.path.exists(submission_file):
        OpLog(f"Submission 파일을 찾을 수 없습니다: {submission_file}", bLines=True)
        return False
    
    try:
        # YAML 파일에서 class names 읽기
        with open(yaml_file, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
            class_names = yaml_data.get('names', [])
        
        OpLog(f"YAML에서 {len(class_names)}개 클래스 로드", bLines=False)
        OpLog(f"클래스 목록: {class_names[:5]}..." if len(class_names) > 5 else f"클래스 목록: {class_names}", bLines=False)
        
        # CSV 파일 읽기
        rows = []
        updated_count = 0
        
        with open(submission_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            for row in reader:
                if 'category_id' in row:
                    try:
                        idx = int(row['category_id'])
                        
                        # 인덱스를 사용해서 실제 category_id 가져오기
                        if 0 <= idx < len(class_names):
                            actual_cat_id = int(class_names[idx])
                            old_cat_id = row['category_id']
                            row['category_id'] = str(actual_cat_id)
                            updated_count += 1
                            
                            if updated_count <= 5:  # 처음 5개만 로그 출력
                                OpLog(f"변환: idx {old_cat_id} -> category_id {actual_cat_id}", bLines=False)
                        else:
                            OpLog(f"Warning: 인덱스 범위 초과 {idx} (최대: {len(class_names)-1})", bLines=False)
                    except (ValueError, IndexError) as e:
                        OpLog(f"변환 실패: {row.get('category_id')} - {e}", bLines=False)
                
                rows.append(row)
        
        # CSV 파일 저장 (백업 생성)
        backup_file = submission_file.replace('.csv', '_backup.csv')
        shutil.copy(submission_file, backup_file)
        OpLog(f"백업 생성: {backup_file}", bLines=False)
        
        with open(submission_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        OpLog(f"changesubmission 완료: {updated_count}개 업데이트, 저장={submission_file}", bLines=True)
        return True
        
    except Exception as e:
        OpLog(f"changesubmission 오류: {e}", bLines=True)
        import traceback
        OpLog(traceback.format_exc(), bLines=False)
        return False

def CheckJwj(img_dir, anno_dir):
    # 이미지-JSON 매핑 분석 실행
    img_to_json, json_to_img = analyze_image_json_mapping(img_dir, anno_dir)
  
# ════════════════════════════════════════
# ▣ 04. 실행
# ════════════════════════════════

if __name__ == "__main__":
    # 이미지 및 어노테이션 디렉토리 경로 설정
    img_dir = r"D:\01.project\EntryPrj\data\jwj\4.drug_Augmentation\train_images"
    anno_dir = r"D:\01.project\EntryPrj\data\jwj\4.drug_Augmentation\train_annotations"
    #CheckJwj(img_dir, anno_dir)


    # 통계 시각화
    #visualize_mapping_statistics(img_to_json, json_to_img)
    
    # 중복 이미지 검사 (필요시 주석 해제)
    # method='checksum'  # 빠른 검사 (MD5 해시 기반)
    # method='pixel'     # 정확한 검사 (픽셀 비교, 느림)
    #duplicates = find_duplicate_images(method='checksum')
    #print(f"\n중복 이미지 {len(duplicates)}개 발견")
    
    # 중복 데이터 복사
    #if duplicates:
    #    copied_images, copied_jsons = copy_missing_data(duplicates)
    #    print(f"복사 완료: 이미지 {copied_images}개, JSON {copied_jsons}개")
    
    # DUPLICATE_RESULT 처리 (이미지 복사 + JSON 복사 + 바운딩 박스 시각화)
    #process_duplicate_result()
    
    # MISSING_ONLY 바운딩 박스 시각화 단독 실행
    # file_path = r"D:\01.project\EntryPrj\data\oraldrug\missingJson\only_annoataions_img\K-003351-013900-035206_0_2_0_2_90_000_200.png"
    # process_missing_only_visualization_Ex(file_path)
    # file_path = r"D:\01.project\EntryPrj\data\oraldrug\missingJson\only_annoataions_img\K-003351-016688-018357_0_2_0_2_90_000_200.png"
    # process_missing_only_visualization_Ex(file_path)
    # file_path = r"D:\01.project\EntryPrj\data\oraldrug\missingJson\only_annoataions_img\K-003351-016688-041768_0_2_0_2_75_000_200.png"
    # process_missing_only_visualization_Ex(file_path)
    # file_path = r"D:\01.project\EntryPrj\data\oraldrug\missingJson\only_annoataions_img\K-003351-018147-020238_0_2_0_2_90_000_200.png"
    # process_missing_only_visualization_Ex(file_path)
    # file_path = r"D:\01.project\EntryPrj\data\oraldrug\missingJson\only_annoataions_img\K-003351-016688-041768_0_2_0_2_75_000_200.png"
    # process_missing_only_visualization_Ex(file_path)
    # file_path = r"D:\01.project\EntryPrj\data\oraldrug\missingJson\only_annoataions_img\K-003351-020014-021325_0_2_0_2_70_000_200.png"
    # process_missing_only_visualization_Ex(file_path)
#    file_path = r"D:\01.project\EntryPrj\data\oraldrug\missingJson\only_annoataions_img\K-003351-003832-029667_0_2_0_2_90_000_200.png"
    #process_missing_only_visualization_Ex(file_path)
        
    #process_missing_only_visualization_new()
    
    # 단일 파일 바운딩 박스 시각화 예제
    #visualize_single_file_boxes("K-003351-016688-018357_0_2_0_2_90_000_200.png")
    
    #find_missing_images_with_json()

    #file_path = r"D:\01.project\EntryPrj\data\AI_HUB\DATA_01\1.Training\라벨링데이터"
    #className_dlName(file_path)
    #file_path = r" D:\01.project\EntryPrj\data\oraldrug\train_annotations"
    #className_dlName(file_path,False)
    file_path =    r"D:\01.project\EntryPrj\data\drug\missing_annotations"
    #change_category_id(file_path)
    #src_path = r"D:\01.project\EntryPrj\data\drug\drugAll"
    #target_apth = r"D:\01.project\EntryPrj\data\drug\drug_Image_annotation"
    # MoveImageAnnotaoin(src_path, target_apth)
    BASE_DIR = r"D:\01.project\EntryPrj\data\drug\drug_ok_Image_no_Anno"
    BASE_DIR = r"D:\01.project\EntryPrj\data\drug\drug_Image_annotation_allOK\drug_Image_annotation_allOK_renewed"
    #process_All_edit(f"{BASE_DIR}\\train_images", f"{BASE_DIR}\\train_annotations")
    #MoveOnlyImage(r"D:\01.project\EntryPrj\data\drug\drugAll",r"D:\01.project\EntryPrj\data\drug\drug_ok_Image_no_Anno")
    file_path = r"D:\01.project\EntryPrj\data\drug\drug_Image_annotation_allOK\train_annotations"
    file_path = r"D:\01.project\EntryPrj\data\drug\drug_ok_Image_no_Anno\train_annotations"
    
    # file_path = r"D:\01.project\EntryPrj\data\drug\drug_Image_annotation_allOK\train_annotations"
    # change_category_id(file_path)
    # file_path = r"D:\01.project\EntryPrj\data\drug\drug_no_image_ok_Anno\train_annotations"
    # change_category_id(file_path)
    # file_path = r"D:\01.project\EntryPrj\data\drug\drug_ok_Image_no_Anno\train_annotations"
    # change_category_id(file_path)
    #TotalStatistic()
    #TotalStatisticByName()
    
    # GetIdx_CatID 테스트
  #  annotations_dir = r"D:\01.project\EntryPrj\data\oraldrug\1.drug_Image_annotation_allOK\train_annotations"
   # idx_to_catid = GetIdx_CatID(annotations_dir)
    
    # print("\n=== YOLO Class Index to Category ID Mapping ===")
    # for idx, cat_id in sorted(idx_to_catid.items()):
    #     print(f"class {idx} = category_id {cat_id}")
    # update_category_id(annotations_dir,r"D:\01.project\EntryPrj\data\submission_yolov8n_conf0.25.csv")


    # changesubmission(r"D:\01.project\EntryPrj\data\submission\submission20251212112431\submission20251212112431.csv",
    #                 r"D:\01.project\EntryPrj\data\oraldrug\yolo_yaml.yaml")

    train_imge_dir = r"D:\01.project\EntryPrj\data\eda\org_data\train_images"
    train_annotation_dir = r"D:\01.project\EntryPrj\data\eda\org_data\train_annotations"
    MISSING_ONLY_ANNOTAIIONS_IMG = r"D:\01.project\EntryPrj\data\eda\noImage_okAnno\train_images"
    MISSING_ONLY_ANNOTATIONS_ADD = r"D:\01.project\EntryPrj\data\eda\noImage_okAnno\train_annotations"
    find_missing_images_with_json(train_imge_dir, train_annotation_dir)

