import os
import yaml
import json

# ════════════════════════════════════════
# ▣ YOLO YAML 파일 생성
# ════════════════════════════════════════

BASE_DIR = r"D:\01.project\EntryPrj\data"
annotations_dir = os.path.join(BASE_DIR, "oraldrug", "train_annotations")
yaml_path = os.path.join(BASE_DIR, "oraldrug", "dataset.yaml")
class_mapping_path = os.path.join(BASE_DIR, "oraldrug", "class_mapping.json")

# YAML 파일이 이미 존재하면 스킵
if os.path.exists(yaml_path):
    print(f"YAML 파일이 이미 존재합니다: {yaml_path}")
    print("생성을 건너뜁니다.")
else:
    # 모든 클래스(K-*) 수집 및 dl_name 매핑
    class_to_name = {}  # {K-code: dl_name}
    unique_classes = set()
    
    for subdir in os.listdir(annotations_dir):
        subdir_path = os.path.join(annotations_dir, subdir)
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
                                    print(f"Error reading {json_path}: {e}")

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
    with open(class_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(class_mapping, f, ensure_ascii=False, indent=2)

    # YAML 데이터 구조 생성
    yaml_data = {
        'path': BASE_DIR,  # 데이터셋 루트 경로
        'train': 'oraldrug/train_images',  # 학습 이미지 경로
        'val': 'oraldrug/val_images',  # 검증 이미지 경로 (나중에 분할 필요)
        'test': 'oraldrug/test_images',  # 테스트 이미지 경로 (옵션)
        
        'nc': len(class_names),  # 클래스 수
        'names': class_names  # 클래스 이름 리스트
    }

    # YAML 파일 저장
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"YAML 파일 생성 완료: {yaml_path}")
    print(f"클래스 매핑 파일 생성 완료: {class_mapping_path}")
    print(f"총 클래스 수: {len(class_names)}")
    print(f"\n클래스 목록 (처음 10개):")
    for i, cls in enumerate(class_names[:10]):
        dl_name = class_to_name.get(cls, '정보없음')
        print(f"  {i}: {cls} -> {dl_name}")
    if len(class_names) > 10:
        print(f"  ... ({len(class_names) - 10}개 더)")
