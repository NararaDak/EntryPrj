import json
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import json
import os
from PIL import Image, ImageDraw
from dataclasses import dataclass

# 실행 방법: streamlit run D:\01.project\EntryPrj\src\streamit_edaexecute.py
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 한글 폰트
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
plt.style.use('fivethirtyeight')

@dataclass
class AnnotaionItem:
    image_file: str
    annotation_file: str
    bbox: list
    dl_name: str
    category_id: int
    dl_idx: str
  


class MyData:
  ANNOTAION_ITEMS = None
  image_dir = None
  annotation_dir = None
  draw_detail_type = 0
  
  # draw_detail_types : 0  목록 이미지 클릭, 1  전체 그리기 버튼, 2 선택 그리기 버튼,3 콤보박스 변경
  @staticmethod
  def get_Dtail_Type():
    return MyData.draw_detail_type
  @staticmethod
  def set_Dtail_Type(value):
    MyData.draw_detail_type = value

  @staticmethod
  def save_anotation_items(annotation_file,dl_name,dl_idx,category_id,box):
    #ANNOTAION_ITEMS의 box를 정보 수정한다.
    if MyData.ANNOTAION_ITEMS is None:
      return
    for item in MyData.ANNOTAION_ITEMS:
      if item.annotation_file == annotation_file:
        item.bbox = box
        item.dl_name = dl_name
        item.dl_idx = dl_idx
        item.category_id = category_id

        MyData.save_annotation_file(item)
        break



  
  #파일을 저장 한다.
  @staticmethod
  def save_annotation_file(item):
    file_path =  os.path.join(MyData.annotation_dir, item.annotation_file)
    #item.bbox를 저장한다.
    try:
      with open(file_path, encoding='utf-8') as f:
        ann_data = json.load(f)
      if 'annotations' in ann_data and isinstance(ann_data['annotations'], list):
        if len(ann_data['annotations'])>0:
          ann_data['annotations'][0]['bbox'] = item.bbox
          ann_data['annotations'][0]['dl_name'] = item.dl_name
          ann_data['annotations'][0]['dl_idx'] = item.dl_idx
          ann_data['annotations'][0]['category_id'] = item.category_id
      with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(ann_data, f, ensure_ascii=False, indent=2)
      st.success(f"Annotation 파일이 저장되었습니다: {file_path}")
    except Exception as e:
      st.error(f"Annotation 파일 저장 중 오류 발생: {e}")

  @staticmethod
  def add_annotation_item(image_file, bbox, dl_name, category_id, dl_idx):
    import re
    # image_file: 이미지 파일명 (상대경로)
    # annotation_dir 하위에 image_file명(확장자 제거) 폴더 생성 후, 01, 02, 03 등 순차적으로 폴더 생성
    # 그 하위에 json 파일 생성
    if MyData.annotation_dir is None:
      st.error("annotation_dir가 설정되지 않았습니다.")
      return
    base_name = os.path.splitext(os.path.basename(image_file))[0]
    base_dir = os.path.join(MyData.annotation_dir, base_name)
    os.makedirs(base_dir, exist_ok=True)
    # 01, 02, 03 등 하위 폴더 중 없는 번호 찾기
    idx = 1
    while True:
      subdir = f"{idx:02d}"
      subdir_path = os.path.join(base_dir, subdir)
      if not os.path.exists(subdir_path):
        os.makedirs(subdir_path)
        break
      idx += 1
      if idx > 99:
        st.error("하위 폴더가 너무 많습니다.")
        return
    # json 파일명
    json_filename = f"{base_name}.json"
    json_path = os.path.join(subdir_path, json_filename)
    # NoneType 방지 및 기본값 처리
    safe_category_id = int(category_id) if category_id is not None else -1
    safe_dl_idx = dl_idx if dl_idx is not None else "0" 
    # bbox 값도 float 변환, None/빈값 방지
    if bbox is None or len(bbox) != 4:
      safe_bbox = [0.0, 0.0, 0.0, 0.0]
    else:
      try:
        safe_bbox = [float(b) if b is not None else 0.0 for b in bbox]
      except Exception:
        safe_bbox = [0.0, 0.0, 0.0, 0.0]
    # coco style json 생성
    coco_json = {
      "images": [
        {
          "file_name": os.path.basename(image_file),
          "imgfile": os.path.basename(image_file),
          "dl_idx": str(safe_dl_idx),
          "dl_name": dl_name if dl_name is not None else "",
        }
      ],
      "type": "instances",
      "annotations": [
        {
          "area": int(safe_bbox[2] * safe_bbox[3]),
          "iscrowd": 0,
          "bbox": safe_bbox,
          "category_id": safe_category_id,
        }
      ],
    }
    try:
      with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_json, f, ensure_ascii=False, indent=2)
      # ANNOTAION_ITEMS에 추가
      rel_ann_path = os.path.relpath(json_path, MyData.annotation_dir)
      item = AnnotaionItem(
        image_file=image_file,
        annotation_file=rel_ann_path,
        bbox=safe_bbox,
        dl_name=dl_name if dl_name is not None else "",
        category_id=safe_category_id,
        dl_idx=safe_dl_idx
      )
      if MyData.ANNOTAION_ITEMS is None:
        MyData.ANNOTAION_ITEMS = []
      MyData.ANNOTAION_ITEMS.append(item)
      st.success(f"Annotation 파일이 추가되었습니다: {json_path}")
    except Exception as e:
      st.error(f"Annotation 파일 추가 중 오류 발생: {e}")

  @staticmethod
  def delete_annotation_item(annotation_file):
    if MyData.ANNOTAION_ITEMS is None:
      return
    
    # 파일 삭제.
    delete_path = os.path.join(MyData.annotation_dir, annotation_file)
    try:
      if os.path.exists(delete_path):
        os.remove(delete_path)
        st.success(f"Annotation 파일이 삭제되었습니다: {delete_path}")
        MyData.ANNOTAION_ITEMS = [item for item in MyData.ANNOTAION_ITEMS if item.annotation_file != annotation_file]
        # 삭제한 파일이 속한 01, 02 등 하위 폴더가 비어 있으면 디렉토리도 삭제
        parent_dir = os.path.dirname(delete_path)
        try:
          if os.path.isdir(parent_dir) and not os.listdir(parent_dir):
            os.rmdir(parent_dir)
            st.info(f"비어있는 폴더도 삭제됨: {parent_dir}")
        except Exception as e:
          st.warning(f"폴더 삭제 중 오류: {e}")
      else:
        st.warning(f"삭제할 Annotation 파일이 존재하지 않습니다: {delete_path}")
    except Exception as e:
      st.error(f"Annotation 파일 삭제 중 오류 발생: {e}")
    
    


  # annotation 로드
  @staticmethod
  def load_anotation(image_dir, annotation_dir):
    MyData.image_dir = image_dir
    MyData.annotation_dir = annotation_dir
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
                    item = AnnotaionItem(
                      image_file=img_rel_path,
                      annotation_file=ann_rel_path,
                      bbox=bbox,
                      dl_name=dl_name,
                      category_id=category_id,
                      dl_idx=ann_dl_idx_str
                    )
                    items.append(item)
                    print(f"A_dl_dix:{item.dl_idx}")
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
                  item = AnnotaionItem(
                    image_file=img_rel_path,
                    annotation_file=ann_rel_path,
                    bbox=bbox,
                    dl_name=dl_name,
                    category_id=category_id,
                    dl_idx=ann_dl_idx_str
                  )
                  print(f"B_dl_dix:{item.dl_idx}")
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
            item = AnnotaionItem(
              image_file=img_rel_path,
              annotation_file=ann_rel_path,
              bbox=bbox,
              dl_name=dl_name,
              category_id=category_id,
              dl_idx=dl_idx
            )
            items.append(item)
            print(f"C_dl_dix:{item.dl_idx}")
        except Exception as e:
          print(f"[load_anotation] Error reading {ann_rel_path}: {e}")
          pass
    MyData.ANNOTAION_ITEMS = items
    print(f"Loaded {len(items)} annotation items.")
    return items
  
  # image_file로 검색
  @staticmethod
  def search_Item_by_image(image_file):
    # image_file은 상대경로로 들어옴
    if MyData.ANNOTAION_ITEMS is None:
      return None
    items = []
    for item in MyData.ANNOTAION_ITEMS:
      if item.image_file == image_file:
        items.append(item)
    return items
  
  @staticmethod
  def search_Iteam_by_annotation(annotation_file):
    # annotation_file은 상대경로로 들어옴
    if MyData.ANNOTAION_ITEMS is None:
      return None
    items = []
    for item in MyData.ANNOTAION_ITEMS:
      if item.annotation_file == annotation_file:
        items.append(item)
    return items  
    
  @staticmethod
  def serach_imge_files(index, count):
    if MyData.ANNOTAION_ITEMS is None:
      return []
    unique_images = list(set([item.image_file for item in MyData.ANNOTAION_ITEMS]))
    unique_images.sort()
    return unique_images[index:index+count]
  @staticmethod
  def get_path_by_image_file(image_file):
    if MyData.image_dir is None:
      return None
    return os.path.join(MyData.image_dir, image_file)
  
  @staticmethod
  def get_path_by_annotation_file(annotation_file):
    if MyData.annotation_dir is None:
      return None
    return os.path.join(MyData.annotation_dir, annotation_file)

train_image_dir =   r"D:\01.project\EntryPrj\data\eda\noImage_okAnno\train_images" # 실제 경로로 변경하세요
train_annotation_dir = r"D:\01.project\EntryPrj\data\eda\noImage_okAnno\train_annotations" #
MyData.load_anotation(train_image_dir, train_annotation_dir)
