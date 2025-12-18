import os
from pathlib import Path

# train_annotations 경로
annotations_dir = Path(r"D:\01.project\EntryPrj\data\oraldrug\train_annotations")

# 모든 고유 클래스(K-로 시작하는 디렉토리) 수집
unique_classes = set()

# train_annotations 아래의 모든 하위 디렉토리 탐색
for subdir in annotations_dir.iterdir():
    if subdir.is_dir():
        # 각 하위 디렉토리 안의 K-로 시작하는 폴더들 확인
        for class_dir in subdir.iterdir():
            if class_dir.is_dir() and class_dir.name.startswith('K-'):
                unique_classes.add(class_dir.name)

# 결과 출력
print(f"총 클래스 수: {len(unique_classes)}")
print(f"\n클래스 목록:")
for cls in sorted(unique_classes):
    print(f"  - {cls}")
