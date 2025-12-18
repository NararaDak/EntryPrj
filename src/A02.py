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

# ════════════════════════════════════════
# ▣ 0. 디렉토리 및 유틸 함수 설정 
# ════════════════════════════════════════
BASE_DIR = r"D:\01.project\EntryPrj\data"
LOG_FILE = os.path.join(BASE_DIR, "operation.log")
ANNOTATION_DIR = os.path.join(BASE_DIR, "oraldrug", "train_annotations")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "oraldrug", "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "oraldrug", "test_images")
YAML_FILE   = os.path.join(BASE_DIR, "oraldrug", "yolo_yaml.yaml")
MODEL_FILES = os.path.join(BASE_DIR, "oraldrug", "models")
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


# ════════════════════════════════════════
# ▣ 3. 기본 모델 클래스 정의 
# ════════════════════════════════════════

class BaseModel(nn.Module):
    """모델의 기본 클래스 - save/load 등 공통 기능 제공"""
    def __init__(self):
        super(BaseModel, self).__init__()
    
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
        model_name = self.__class__.__name__
        
        # Best 모델 파일명
        if is_best:
            filename = os.path.join(save_dir, f"{model_name}_best_model.pth")
        else:
            filename = os.path.join(save_dir, f"{model_name}_epoch_{epoch_index}.pth")
        
        # 기본 저장 데이터
        checkpoint = {
            'epoch': epoch_index,
            'is_best': is_best,
            'model_name': model_name,
        }
        
        # kwargs로 전달된 추가 데이터 저장
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, filename)
        
        if is_best:
            print(f"  🏆 Best 모델 저장됨: {filename}")
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
        
        OpLog(f"모델 로드 완료: {model_file} (Epoch {checkpoint['epoch']})", bLines=True)
        return checkpoint

# ════════════════════════════════════════
# ▣ 4. EfficientNetModel 모델 정의 
# ════════════════════════════════════════

   
class EfficientNetModel(BaseModel):
    """
    알약 분류 모델 (이미지 분류용)
    주의: 실제 YOLO 객체 탐지 모델이 아닌 EfficientNet 기반 분류기입니다.
    """
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()
        self.num_classes = num_classes
        # EfficientNet-B0 모델 로드 (사전 학습된 가중치 사용)
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        # 분류기 레이어 교체
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Linear(in_features, num_classes)
        
        # 학습 이력 저장용
        self.train_losses = []
        self.train_accs = []
        self.val_accs = []
    @staticmethod
    def preJob():
        """전처리 작업: YOLO YAML 파일 및 클래스 매핑 생성 (없을 경우에만)"""
        import yaml
        
        class_mapping_file = os.path.join(BASE_DIR, "oraldrug", "class_mapping.json")
        
        # YAML 파일이 이미 존재하면 패스
        if os.path.exists(YAML_FILE):
            OpLog(f"YAML 파일이 이미 존재합니다: {YAML_FILE}", bLines=False)
            return
        
        OpLog("YOLO YAML 파일 생성 시작", bLines=True)
        
        # 모든 클래스(K-*) 수집 및 dl_name 매핑
        class_to_name = {}  # {K-code: dl_name}
        unique_classes = set()
        
        for subdir in os.listdir(ANNOTATION_DIR):
            subdir_path = os.path.join(ANNOTATION_DIR, subdir)
            if os.path.isdir(subdir_path):
                for class_dir in os.listdir(subdir_path):
                    if class_dir.startswith('K-'):
                        unique_classes.add(class_dir)
                        
                        # 해당 클래스 폴더의 첫 번째 JSON 파일에서 dl_name 추출
                        class_dir_path = os.path.join(subdir_path, class_dir)
                        if os.path.isdir(class_dir_path) and class_dir not in class_to_name:
                            for json_file in os.listdir(class_dir_path):
                                if json_file.endswith('.json'):
                                    json_path = os.path.join(class_dir_path, json_file)
                                    try:
                                        with open(json_path, 'r', encoding='utf-8') as f:
                                            data = json.load(f)
                                        if 'images' in data and len(data['images']) > 0:
                                            dl_name = data['images'][0].get('dl_name', class_dir)
                                            class_to_name[class_dir] = dl_name
                                            break
                                    except Exception as e:
                                        OpLog(f"Error reading {json_path}: {e}", bLines=False)
        
        # 클래스 정렬
        class_names = sorted(unique_classes)
        
        # 클래스 매핑 정보 저장 (K-code: {index, dl_name})
        class_mapping = {}
        for idx, cls in enumerate(class_names):
            class_mapping[cls] = {
                'index': idx,
                'dl_name': class_to_name.get(cls, cls)
            }
        
        # 클래스 매핑 JSON 파일 저장
        with open(class_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(class_mapping, f, ensure_ascii=False, indent=2)
        
        # YAML 데이터 구조 생성
        yaml_data = {
            'path': BASE_DIR,
            'train': 'oraldrug/train_images',
            'val': 'oraldrug/val_images',
            'test': 'oraldrug/test_images',
            'nc': len(class_names),
            'names': class_names
        }
        
        # YAML 파일 저장
        makedirs(os.path.dirname(YAML_FILE))
        with open(YAML_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        OpLog(f"YAML 파일 생성 완료: {YAML_FILE}", bLines=False)
        OpLog(f"클래스 매핑 파일 생성 완료: {class_mapping_file}", bLines=False)
        OpLog(f"총 클래스 수: {len(class_names)}", bLines=False)
        
    def forward(self, x):
        return self.backbone(x)

    def getOptimizers(self, lr, betas):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)
        return optimizer
    def getCriterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion


    def fit(self, train_loader, val_loader, epochs=50, lr=0.0002, device='cuda'):
        """
        현재 구현: 이미지 분류(Classification) 학습
        - EfficientNet 백본 사용
        - CrossEntropyLoss로 클래스 분류만 수행
        - bbox 정보는 사용하지 않음
        
        주의: 실제 YOLO는 객체 탐지 모델이며, bbox 예측 + 클래스 분류를 동시에 수행합니다.
        진정한 YOLO 학습을 원한다면 YOLOv5/YOLOv8 라이브러리 사용을 권장합니다.
        """
        self.train()
        optimizer = self.getOptimizers(lr, (0.5, 0.999))
        criterion = self.getCriterion()
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            # Training loop
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = self(images)  # forward 메서드 호출
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # 정확도 계산
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if batch_idx % 10 == 0:
                    OpLog(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}", bLines=False)
            
            # Epoch 결과 출력
            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100 * correct / total
            self.train_losses.append(avg_loss)
            self.train_accs.append(accuracy)
            
            OpLog(f"Epoch [{epoch+1}/{epochs}] 완료 - Avg Loss: {avg_loss:.4f}, "
                  f"Train Accuracy: {accuracy:.2f}%", bLines=True)
            
            # Validation
            if val_loader:
                val_acc = self.evaluate(val_loader, device)
                self.val_accs.append(val_acc)
                OpLog(f"Validation Accuracy: {val_acc:.2f}%", bLines=False)
                
                # Best 모델 저장
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_model(epoch + 1, is_best=True,
                                  model_state_dict=self.state_dict(),
                                  num_classes=self.num_classes,
                                  train_losses=self.train_losses,
                                  train_accs=self.train_accs,
                                  val_accs=self.val_accs)
                    OpLog(f"Best 모델 저장됨 (Epoch {epoch+1}, Val Acc: {val_acc:.2f}%)", bLines=False)
            
            # 주기적 저장 (10 에포크마다)
            if (epoch + 1) % 10 == 0:
                self.save_model(epoch + 1, is_best=False,
                              model_state_dict=self.state_dict(),
                              num_classes=self.num_classes,
                              train_losses=self.train_losses,
                              train_accs=self.train_accs,
                              val_accs=self.val_accs)
        
        # 학습 완료 후 최종 모델 저장
        self.save_model(epochs, is_best=False,
                       model_state_dict=self.state_dict(),
                       num_classes=self.num_classes,
                       train_losses=self.train_losses,
                       train_accs=self.train_accs,
                       val_accs=self.val_accs)
        OpLog(f"학습 완료! Best Validation Accuracy: {best_val_acc:.2f}%", bLines=True)
        
        # 학습 곡선 시각화
        self.plot_training_history()
    
    def evaluate(self, val_loader, device='cuda'):
        """검증 데이터셋에 대한 정확도 평가 (분류 모델용)"""
        self.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100 * correct / total
        self.train()  # 다시 학습 모드로 전환
        return accuracy
    
    def load_yolo_model(self, model_path):
        """YOLO 모델 전용 로드 함수"""
        checkpoint = self.load_model(model_path)
        if checkpoint is None:
            return False
        
        # YoloModel 전용 데이터 복원
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.train_accs = checkpoint.get('train_accs', [])
        self.val_accs = checkpoint.get('val_accs', [])
        
        return True
    
    def plot_training_history(self):
        """학습 이력 시각화"""
        if not self.train_losses:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss 그래프
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy 그래프
        ax2.plot(self.train_accs, label='Train Accuracy')
        if self.val_accs:
            ax2.plot(self.val_accs, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training & Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # 저장
        result_dir = os.path.join(BASE_DIR, "model_results")
        makedirs(result_dir)
        filename = os.path.join(result_dir, "training_history.png")
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        OpLog(f"학습 이력 그래프 저장됨: {filename}", bLines=False)
        
        plt.show(block=False)
        plt.pause(3)
        plt.close()

# ════════════════════════════════════════
# ▣ 5. YOLOv8 모델 정의 
# ════════════════════════════════════════

class YOLOv8Model(BaseModel):
    """
    YOLOv8 기반 객체 탐지 모델
    - Ultralytics YOLOv8 사용
    - 객체 탐지 및 분류 동시 수행
    """
    def __init__(self, model_size='n', num_classes=None):
        """
        Args:
            model_size: YOLOv8 모델 크기 ('n', 's', 'm', 'l', 'x')
            num_classes: 클래스 수 (None이면 자동 계산)
        """
        super(YOLOv8Model, self).__init__()
        self.model_size = model_size
        self.num_classes = num_classes if num_classes else count_classes(ANNOTATION_DIR)
        
        # YOLOv8 모델 초기화 (사전 학습된 가중치 사용)
        self.model = YOLO(f'yolov8{model_size}.pt')
        
        # 학습 이력 저장용
        self.train_losses = []
        self.val_metrics = []
    
    @staticmethod
    def preJob():
        """전처리 작업: YOLO YAML 파일 및 클래스 매핑 생성 (없을 경우에만)"""
        import yaml
        
        class_mapping_file = os.path.join(BASE_DIR, "oraldrug", "class_mapping.json")
        
        # YAML 파일이 이미 존재하면 패스
        if os.path.exists(YAML_FILE):
            OpLog(f"YAML 파일이 이미 존재합니다: {YAML_FILE}", bLines=False)
            return
        
        OpLog("YOLO YAML 파일 생성 시작", bLines=True)
        
        # 모든 클래스(K-*) 수집 및 dl_name 매핑
        class_to_name = {}  # {K-code: dl_name}
        unique_classes = set()
        
        for subdir in os.listdir(ANNOTATION_DIR):
            subdir_path = os.path.join(ANNOTATION_DIR, subdir)
            if os.path.isdir(subdir_path):
                for class_dir in os.listdir(subdir_path):
                    if class_dir.startswith('K-'):
                        unique_classes.add(class_dir)
                        
                        # 해당 클래스 폴더의 첫 번째 JSON 파일에서 dl_name 추출
                        class_dir_path = os.path.join(subdir_path, class_dir)
                        if os.path.isdir(class_dir_path) and class_dir not in class_to_name:
                            for json_file in os.listdir(class_dir_path):
                                if json_file.endswith('.json'):
                                    json_path = os.path.join(class_dir_path, json_file)
                                    try:
                                        with open(json_path, 'r', encoding='utf-8') as f:
                                            data = json.load(f)
                                        if 'images' in data and len(data['images']) > 0:
                                            dl_name = data['images'][0].get('dl_name', class_dir)
                                            class_to_name[class_dir] = dl_name
                                            break
                                    except Exception as e:
                                        OpLog(f"Error reading {json_path}: {e}", bLines=False)
        
        # 클래스 정렬
        class_names = sorted(unique_classes)
        
        # 클래스 매핑 정보 저장 (K-code: {index, dl_name})
        class_mapping = {}
        for idx, cls in enumerate(class_names):
            class_mapping[cls] = {
                'index': idx,
                'dl_name': class_to_name.get(cls, cls)
            }
        
        # 클래스 매핑 JSON 파일 저장
        with open(class_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(class_mapping, f, ensure_ascii=False, indent=2)
        
        # YAML 데이터 구조 생성
        yaml_data = {
            'path': BASE_DIR,
            'train': 'oraldrug/train_images',
            'val': 'oraldrug/val_images',
            'test': 'oraldrug/test_images',
            'nc': len(class_names),
            'names': class_names
        }
        
        # YAML 파일 저장
        makedirs(os.path.dirname(YAML_FILE))
        with open(YAML_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        OpLog(f"YAML 파일 생성 완료: {YAML_FILE}", bLines=False)
        OpLog(f"클래스 매핑 파일 생성 완료: {class_mapping_file}", bLines=False)
        OpLog(f"총 클래스 수: {len(class_names)}", bLines=False)
    
    def fit(self, epochs=50, imgsz=640, batch_size=16, device='cuda'):
        """
        YOLOv8 모델 학습 (BaseModel save_model 사용)
        
        Args:
            epochs: 학습 에포크 수
            imgsz: 입력 이미지 크기
            batch_size: 배치 크기
            device: 'cuda' 또는 'cpu'
        """
        from ultralytics.utils.callbacks import default_callbacks
        
        OpLog(f"YOLOv8{self.model_size} 모델 학습 시작", bLines=True)
        
        # YAML 파일 확인
        if not os.path.exists(YAML_FILE):
            self.preJob()
        
        # 매 epoch마다 BaseModel의 save_model을 호출하는 콜백 함수
        def on_epoch_end(trainer):
            epoch = trainer.epoch
            # BaseModel의 save_model 사용
            save_path = os.path.join(MODEL_DIR, f"yolov8{self.model_size}_epoch{epoch+1}.pt")
            self.save_model(
                filepath=save_path,
                epoch=epoch + 1,
                model_state=trainer.model.state_dict(),
                metrics={
                    'box_loss': float(trainer.loss_items[0]) if trainer.loss_items is not None else 0,
                    'cls_loss': float(trainer.loss_items[1]) if trainer.loss_items is not None else 0,
                    'dfl_loss': float(trainer.loss_items[2]) if trainer.loss_items is not None else 0,
                }
            )
            OpLog(f"Epoch {epoch+1} 모델 저장: {save_path}", bLines=False)
        
        # 콜백 추가
        self.model.add_callback("on_epoch_end", on_epoch_end)
        
        # YOLOv8 학습 시작
        results = self.model.train(
            data=YAML_FILE,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            project=os.path.join(BASE_DIR, "yolo_results"),
            name=f"yolov8{self.model_size}_train",
            exist_ok=True,
            patience=10,  # Early stopping
            save=True,  # 최종 모델 저장
            plots=True,
            verbose=True,
        )
        
        OpLog(f"YOLOv8 학습 완료!", bLines=True)
        
        # 학습 결과 시각화
        self.plot_results()
        
        return results
    
    def evaluate(self, data_yaml=None, device='cuda'):
        """
        검증 데이터셋에 대한 모델 평가
        
        Args:
            data_yaml: 데이터셋 YAML 파일 경로 (None이면 YAML_FILE 사용)
            device: 'cuda' 또는 'cpu'
        """
        if data_yaml is None:
            data_yaml = YAML_FILE
        
        OpLog("YOLOv8 모델 평가 시작", bLines=True)
        
        # 모델 검증
        metrics = self.model.val(
            data=data_yaml,
            device=device,
            split='val',
            plots=True,
        )
        
        # 주요 메트릭 출력
        OpLog(f"mAP50: {metrics.box.map50:.4f}", bLines=False)
        OpLog(f"mAP50-95: {metrics.box.map:.4f}", bLines=False)
        OpLog(f"Precision: {metrics.box.mp:.4f}", bLines=False)
        OpLog(f"Recall: {metrics.box.mr:.4f}", bLines=False)
        
        return metrics
    
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
        self.model.export(format='torchscript', dynamic=False)
        
        OpLog(f"YOLOv8 모델 저장됨: {save_path}", bLines=True)
        return save_path
    
    def plot_results(self):
        """학습 결과 시각화"""
        results_dir = os.path.join(BASE_DIR, "yolo_results", f"yolov8{self.model_size}_train")
        results_file = os.path.join(results_dir, "results.png")
        
        if os.path.exists(results_file):
            OpLog(f"학습 결과 그래프: {results_file}", bLines=False)
        else:
            OpLog("학습 결과 파일을 찾을 수 없습니다.", bLines=False)
        plt.close()