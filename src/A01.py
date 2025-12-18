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
# ════════════════════════════════════════
# ▣ 0. 디렉토리 및 유틸 함수 설정 
# ════════════════════════════════════════
BASE_DIR = r"D:\01.project\EntryPrj\data"
LOG_FILE = os.path.join(BASE_DIR, "operation.log")
ANNOTATION_DIR = os.path.join(BASE_DIR, "oraldrug", "train_annotations")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "oraldrug", "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "oraldrug", "test_images")


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
# ▣ 1. 클래스 수 계산
# ════════════════════════════════════════
# train_annotations에서 고유한 K-* 디렉토리 개수로 클래스 수 계산
def count_classes(annotations_dir):
    unique_classes = set()
    for subdir in os.listdir(annotations_dir):
        subdir_path = os.path.join(annotations_dir, subdir)
        if os.path.isdir(subdir_path):
            for class_dir in os.listdir(subdir_path):
                if class_dir.startswith('K-'):
                    unique_classes.add(class_dir)
    return len(unique_classes)

Lines(f"ANNOTATION_DIR: {ANNOTATION_DIR}")
num_classes = count_classes(ANNOTATION_DIR)
OpLog(f"총 클래스 수: {num_classes}", bLines=True)
 
# ════════════════════════════════════════
# ▣ 2. 데이터셋 및 데이터 증강 함수 정의
# ════════════════════════════════════════
# 다양한 데이터 증강(transform) 함수 정의
def GetTransform(transform_type="default"):
    if( transform_type == "default" ):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    if( transform_type == "A" ):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
        ])
    elif( transform_type == "B" ):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

# 커스텀 데이터셋 클래스 정의
class PillDataset(Dataset):
    def __init__(self, annotations_dir, img_dir, transform=None):
        """ annotations_dir: train_annotations 경로
        img_dir: train_images 경로
        """
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []  # (img_path, label_idx, class_name) 튜플 리스트
        self.class_to_idx = {}  # {class_name: idx}
        self.idx_to_class = {}  # {idx: class_name}
        
        # 모든 클래스(K-*) 디렉토리 수집
        class_dirs = []
        for subdir in os.listdir(annotations_dir):
            subdir_path = os.path.join(annotations_dir, subdir)
            if os.path.isdir(subdir_path):
                for class_dir in os.listdir(subdir_path):
                    if class_dir.startswith('K-'):
                        class_dir_path = os.path.join(subdir_path, class_dir)
                        if os.path.isdir(class_dir_path):
                            class_dirs.append((class_dir, class_dir_path))
        
        # 클래스 정렬 및 인덱스 매핑
        self._unique_classes = sorted(set([cls for cls, _ in class_dirs]))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self._unique_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # 각 클래스의 annotation 파일 읽기
        for class_name, class_dir_path in class_dirs:
            label_idx = self.class_to_idx[class_name]
            
            # 클래스 디렉토리 내 모든 JSON 파일 읽기
            for json_file in os.listdir(class_dir_path):
                if json_file.endswith('.json'):
                    json_path = os.path.join(class_dir_path, json_file)
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # images 정보 추출
                        if 'images' in data:
                            for img_info in data['images']:
                                img_filename = img_info['file_name']
                                img_path = os.path.join(self.img_dir, img_filename)
                                
                                # 이미지 파일이 실제로 존재하는지 확인
                                if os.path.exists(img_path):
                                    self.samples.append((img_path, label_idx, class_name))
                    except Exception as e:
                        OpLog(f"Error reading {json_path}: {e}", bLines=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, class_name = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, label

def GetDataset(annotations_dir, img_dir, transform_type="default"):
    transform = GetTransform(transform_type)
    dataset = PillDataset(annotations_dir, img_dir, transform)
    return dataset

def GetLoaders(annotations_dir, img_dir, batch_size=32, train_ratio=0.8, num_workers=4):
    """
    전체 데이터셋을 train/val로 분할하여 DataLoader 생성
    """
    from torch.utils.data import DataLoader, random_split
    
    # 전체 데이터셋 로드 (train용 augmentation)
    full_dataset = GetDataset(annotations_dir, img_dir, transform_type="A")
    # Train/Val 분할
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    # Validation 데이터셋에는 augmentation 없이 기본 transform만 적용
    val_dataset_plain = GetDataset(annotations_dir, img_dir, transform_type="default")
    val_indices = val_dataset.indices
    val_dataset = torch.utils.data.Subset(val_dataset_plain, val_indices)
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    OpLog(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}", bLines=False)
    OpLog(f"Total classes: {len(full_dataset.class_to_idx)}", bLines=False)
    return train_loader, val_loader

def TestLoader():
    train_loader, val_loader = GetLoaders(ANNOTATION_DIR, TRAIN_IMG_DIR, batch_size=16, train_ratio=0.8, num_workers=2)
    return train_loader, val_loader
TestLoader()



  

def test():   
    # num_classes = len(set([cat['id'] for cat in json.load(open("annotations.json"))['categories']]))

    model_resnet = models.resnet18(pretrained=True)
    model_resnet.fc = nn.Linear(model_resnet.fc.in_features, num_classes)


    model_effnet = EfficientNet.from_pretrained('efficientnet-b0')
    model_effnet._fc = nn.Linear(model_effnet._fc.in_features, num_classes)

    def train_model(model, train_loader, criterion, optimizer, device):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    def evaluate_model(model, val_loader, device):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy: {100 * correct / total:.2f}%')

    # bbox: [x, y, w, h] → YOLO format: [class, x_center, y_center, w, h] (normalized)
    def convert_to_yolo_format(annotation, img_width, img_height):
        x, y, w, h = annotation['bbox']
        x_center = (x + w/2) / img_width
        y_center = (y + h/2) / img_height
        w /= img_width
        h /= img_height
        return f"{annotation['category_id']} {x_center} {y_center} {w} {h}"

