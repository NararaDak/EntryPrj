import json
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import json
import os
from PIL import Image, ImageDraw
import streamlit as st
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
    if MyData.ANNOTAION_ITEMS is not None and  10 < len(MyData.ANNOTAION_ITEMS):
      return MyData.ANNOTAION_ITEMS
    
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
  

     

def draw_box_with_override(img, ann_candidates, image_file, selected_ann=None, override_bbox=None, width=450, placeholder=None):
  img_draw = img.copy()
  draw = ImageDraw.Draw(img_draw)
  from PIL import ImageFont
  font = None
  font_candidates = [
    "NanumGothic.ttf",  # 나눔고딕 (윈도우/리눅스 설치시)
    "malgun.ttf",       # 맑은고딕 (Windows)
    "MalgunGothic.ttf",# 맑은고딕 (다른 표기)
    "arial.ttf"        # 영문 Arial
  ]
  for font_path in font_candidates:
    try:
      font = ImageFont.truetype(font_path, 28)
      break
    except Exception:
      font = None
  if font is None:
    font = ImageFont.load_default()
  for ann_item in ann_candidates:
    ann_path = os.path.join(MyData.annotation_dir, ann_item.annotation_file)
    try:
      with open(ann_path, encoding='utf-8') as f:
        ann_data = json.load(f)
      if 'annotations' in ann_data and isinstance(ann_data['annotations'], list):
        for idx, ann in enumerate(ann_data['annotations']):
          # 텍스트 정보 준비
          dl_name = ann_data.get('dl_name')
          if not dl_name and 'images' in ann_data and ann_data['images']:
            dl_name = ann_data['images'][0].get('dl_name')
          category_id = ann.get('category_id', None)
          label_text = f"{dl_name if dl_name else ''} | {category_id if category_id is not None else ''}"
          # 선택 그리기(파랑)
          if ann_item.annotation_file == (selected_ann.annotation_file if selected_ann else None) and override_bbox is not None and idx == 0:
            if isinstance(override_bbox, list) and len(override_bbox) == 4:
              try:
                x, y, w, h = [float(v) for v in override_bbox]
                draw.rectangle([x, y, x + w, y + h], outline='blue', width=4)
                # 텍스트 표시 (좌상단, 배경)
                text_x, text_y = x, max(0, y-28)
                text_size = draw.textbbox((text_x, text_y), label_text, font=font)
                draw.rectangle([text_size[0], text_size[1], text_size[2], text_size[3]], fill=(255,255,255,200))
                draw.text((text_x, text_y), label_text, fill='black', font=font)
              except Exception:
                pass
          # 전체 그리기(빨강)
          elif 'bbox' in ann and isinstance(ann['bbox'], list) and len(ann['bbox']) == 4:
            x, y, w, h = ann['bbox']
            draw.rectangle([x, y, x + w, y + h], outline='red', width=4)
            text_x, text_y = x, max(0, y-28)
            text_size = draw.textbbox((text_x, text_y), label_text, font=font)
            draw.rectangle([text_size[0], text_size[1], text_size[2], text_size[3]], fill=(255,255,255,200))
            draw.text((text_x, text_y), label_text, fill='black', font=font)
    except Exception:
      pass
  if placeholder is not None:
    placeholder.image(img_draw, caption=f"상세 보기: {os.path.basename(image_file)} (수정 bbox: 파랑, 기존: 빨강)", width=width)
  else:
    st.image(img_draw, caption=f"상세 보기: {os.path.basename(image_file)} (수정 bbox: 파랑, 기존: 빨강)", width=width)


# -------------------------------
# bbox 그리기
# -------------------------------
def draw_box(img, ann_candidates, image_file, width=450, placeholder=None):
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    from PIL import ImageFont
    font = None
    font_candidates = [
      "NanumGothic.ttf",  # 나눔고딕 (윈도우/리눅스 설치시)
      "malgun.ttf",       # 맑은고딕 (Windows)
      "MalgunGothic.ttf",# 맑은고딕 (다른 표기)
      "arial.ttf"        # 영문 Arial
    ]
    for font_path in font_candidates:
      try:
        font = ImageFont.truetype(font_path, 28)
        break
      except Exception:
        font = None
    if font is None:
      font = ImageFont.load_default()
    for ann_item in ann_candidates:
      ann_path = os.path.join(MyData.annotation_dir, ann_item.annotation_file)
      try:
        with open(ann_path, encoding='utf-8') as f:
          ann_data = json.load(f)
        if 'annotations' in ann_data and isinstance(ann_data['annotations'], list):
          for ann in ann_data['annotations']:
            dl_name = ann_data.get('dl_name')
            if not dl_name and 'images' in ann_data and ann_data['images']:
              dl_name = ann_data['images'][0].get('dl_name')
            category_id = ann.get('category_id', None)
            label_text = f"{dl_name if dl_name else ''} | {category_id if category_id is not None else ''}"
            if 'bbox' in ann and isinstance(ann['bbox'], list) and len(ann['bbox']) == 4:
              x, y, w, h = ann['bbox']
              draw.rectangle([x, y, x + w, y + h], outline='red', width=4)
              text_x, text_y = x, max(0, y-28)
              text_size = draw.textbbox((text_x, text_y), label_text, font=font)
              draw.rectangle([text_size[0], text_size[1], text_size[2], text_size[3]], fill=(255,255,255,200))
              draw.text((text_x, text_y), label_text, fill='black', font=font)
      except Exception:
        pass
    if placeholder is not None:
        placeholder.image(img_draw, caption=f"상세 보기: {os.path.basename(image_file)} (모든 bbox)", width=width)
    else:
        st.image(img_draw, caption=f"상세 보기: {os.path.basename(image_file)} (모든 bbox)", width=width)



def edit_annotation_byImage(image_file, annotation_dir=None):
    """ image_file: 상대경로 (MyData.image_dir 기준)
    annotation_dir: MyData.annotation_dir 사용
    """
    # 1. 컬럼 분할
    col_img, col_anno = st.columns([2, 2])

    # 2. image_file이 전체 경로로 들어오면 image_dir 기준 상대경로로 변환
    if os.path.isabs(image_file) and MyData.image_dir and image_file.startswith(MyData.image_dir):
        rel_image_file = os.path.relpath(image_file, MyData.image_dir)
    else:
        rel_image_file = image_file

    # 3. annotation 후보 리스트
    ann_candidates = MyData.search_Item_by_image(rel_image_file)
    img_path = MyData.get_path_by_image_file(rel_image_file)

    # 4. 이미지 영역
    with col_img:
      img_placeholder = st.empty()
      if not rel_image_file or not img_path or not os.path.exists(img_path):
        st.warning(f"이미지 파일을 찾을 수 없습니다: {img_path}")
        st.session_state['selected_train_image'] = None
      else:
        try:
          img = Image.open(img_path)
          detail_type = MyData.get_Dtail_Type()
          selected_ann = st.session_state.get(f'selected_ann_{image_file}', None)
          override_bbox = st.session_state.get(f'override_bbox_{image_file}', None)
          if detail_type == 1 and ann_candidates:
            draw_box_with_override(img, ann_candidates, image_file, selected_ann=selected_ann, override_bbox=override_bbox, width=450, placeholder=img_placeholder)
          elif detail_type == 2 and selected_ann and override_bbox:
            draw_box_with_override(img, [selected_ann], image_file, selected_ann=selected_ann, override_bbox=override_bbox, width=450, placeholder=img_placeholder)
          else:
            if ann_candidates:
              draw_box(img, ann_candidates, image_file, width=450, placeholder=img_placeholder)
            else:
              img_placeholder.image(img, caption=f"상세 보기: {os.path.basename(image_file)}", width=450)
          st.session_state['detail_image_draw'] = False
        except Exception as e:
          img_placeholder.error(f"상세 보기 오류: {os.path.basename(image_file)}\n{img_path}\n{e}")
          st.session_state['selected_train_image'] = None

    # 5. annotation 영역
    with col_anno:
        if ann_candidates:
            ann_candidates_sorted = sorted(ann_candidates, key=lambda x: x.annotation_file)
            ann_labels = [ann.annotation_file for ann in ann_candidates_sorted]
            prev_selected_ann = st.session_state.get(f'selected_ann_{image_file}', None)
            try:
                if prev_selected_ann:
                    default_index = next((i for i, ann in enumerate(ann_candidates_sorted) if ann.annotation_file == prev_selected_ann.annotation_file), 0) if isinstance(prev_selected_ann, AnnotaionItem) else 0
                else:
                    default_index = 0
            except Exception:
                default_index = 0
            selected_label = st.selectbox("Annotation 파일 선택", ann_labels, index=default_index, key=f"anno_combo_{image_file}")
            try:
                # search_Iteam_by_annotation을 사용하여 annotation 정보를 가져옴
                ann_items = MyData.search_Iteam_by_annotation(selected_label)
                if ann_items and len(ann_items) > 0:
                    selected_ann = ann_items[0]
                    st.session_state[f'selected_ann_{image_file}'] = selected_ann
                    # ann_items[0]의 값을 직접 사용
                    dl_name = selected_ann.dl_name if hasattr(selected_ann, 'dl_name') else ''
                    dl_idx = selected_ann.dl_idx if hasattr(selected_ann, 'dl_idx') else ''
                    category_id = selected_ann.category_id if hasattr(selected_ann, 'category_id') else ''
                    bbox = selected_ann.bbox if hasattr(selected_ann, 'bbox') and isinstance(selected_ann.bbox, list) and len(selected_ann.bbox) == 4 else None
                    if not dl_name:
                        st.warning("선택한 annotation에 dl_name 항목이 없습니다.")
                        st.warning(f"annotation_file: {selected_ann.annotation_file}")
                    # dl_name, dl_idx, category_id 입력란 준비
                    # 기본값 세팅
                    dl_name_val = dl_name if dl_name is not None else ''
                    dl_idx_val = dl_idx if dl_idx is not None else ''
                    category_id_val = category_id if category_id is not None else ''
                    # 첫 줄: dl_name, dl_idx, category_id 제목+입력
                    st.markdown('<div style="font-weight:bold; display:flex; gap:8px;">'
                          '<div style="width:120px;display:inline-block;">dl_name</div>'
                          '<div style="width:80px;display:inline-block;">dl_idx</div>'
                          '<div style="width:100px;display:inline-block;">category_id</div>'
                          '</div>', unsafe_allow_html=True)
                    cols1 = st.columns([2, 1, 1.2])
                    dl_name_input = cols1[0].text_input('dl_name', value=str(dl_name_val), key=f'dlname_{selected_ann.annotation_file}')
                    dl_idx_input = cols1[1].text_input('dl_idx', value=str(dl_idx_val), key=f'dlidx_{selected_ann.annotation_file}')
                    category_id_input = cols1[2].text_input('category_id', value=str(category_id_val), key=f'catid_{selected_ann.annotation_file}')
                    # 두 번째 줄: left, top, width, height 제목+입력
                    st.markdown('<div style="font-weight:bold; display:flex; gap:8px; margin-top:8px;">'
                          '<div style="width:80px;display:inline-block;">left</div>'
                          '<div style="width:80px;display:inline-block;">top</div>'
                          '<div style="width:80px;display:inline-block;">width</div>'
                          '<div style="width:80px;display:inline-block;">height</div>'
                          '</div>', unsafe_allow_html=True)
                    cols2 = st.columns([1, 1, 1, 1])
                    if not bbox:
                      left = cols2[0].text_input('left', value='', key=f'bbox0_{selected_ann.annotation_file}')
                      top = cols2[1].text_input('top', value='', key=f'bbox1_{selected_ann.annotation_file}')
                      width_ = cols2[2].text_input('width', value='', key=f'bbox2_{selected_ann.annotation_file}')
                      height = cols2[3].text_input('height', value='', key=f'bbox3_{selected_ann.annotation_file}')
                      st.warning('해당 annotation에 bbox 정보가 없거나 올바르지 않습니다.')
                    else:
                      b1, b2, b3, b4 = bbox
                      left = cols2[0].text_input('left', value=str(b1), key=f'bbox0_{selected_ann.annotation_file}')
                      top = cols2[1].text_input('top', value=str(b2), key=f'bbox1_{selected_ann.annotation_file}')
                      width_ = cols2[2].text_input('width', value=str(b3), key=f'bbox2_{selected_ann.annotation_file}')
                      height = cols2[3].text_input('height', value=str(b4), key=f'bbox3_{selected_ann.annotation_file}')
                    st.session_state[f'override_bbox_{image_file}'] = [left, top, width_, height]
                    # 버튼 행
                    btn1, btn2, btn3, btn4, btn5 = st.columns([1, 1, 1, 1, 1])
                    draw_all_clicked = btn1.button('전체 그리기', key=f'draw_{selected_ann.annotation_file}')
                    draw_selected_clicked = btn2.button('선택 그리기', key=f'draw_selected_{selected_ann.annotation_file}')
                    save_clicked = btn3.button('저장', key=f'save_{selected_ann.annotation_file}')
                    add_clicked = btn4.button('추가', key=f'add_{selected_ann.annotation_file}')
                    delete_clicked = btn5.button('삭제', key=f'delete_{selected_ann.annotation_file}')
                    if draw_all_clicked:
                        st.session_state[f'draw_clicked_{image_file}'] = True
                        MyData.set_Dtail_Type(1)  # 전체 그리기 버튼
                    elif draw_selected_clicked:
                        st.session_state[f'draw_clicked_{image_file}'] = False
                        MyData.set_Dtail_Type(2)  # 선택 그리기 버튼
                        try:
                            img = Image.open(img_path)
                            draw_box_with_override(
                                img,
                                [selected_ann],
                                image_file,
                                selected_ann=selected_ann,
                                override_bbox=[left, top, width_, height],
                                width=450,
                                placeholder=img_placeholder
                            )
                            ann_path = MyData.get_path_by_annotation_file(selected_ann.annotation_file)
                            with open(ann_path, encoding='utf-8') as f:
                                anno_data = json.load(f)
                            dl_name = anno_data.get('dl_name')
                            if not dl_name and 'images' in anno_data and anno_data['images']:
                                dl_name = anno_data['images'][0].get('dl_name')
                            category_id = None
                            if 'annotations' in anno_data and anno_data['annotations']:
                                ann0 = anno_data['annotations'][0]
                                category_id = ann0.get('category_id', None)
                            st.info(f"[선택 그리기] dl_name: {dl_name}")
                            st.info(f"[선택 그리기] category_id: {category_id}")
                        except Exception as e:
                            st.error(f"선택 그리기 오류: {e}")
                    elif save_clicked:
                        try:
                            left_f = float(left)
                            top_f = float(top)
                            width_f = float(width_)
                            height_f = float(height)
                            new_bbox = [left_f, top_f, width_f, height_f]
                            # 입력값 반영
                            new_dl_name = dl_name_input
                            new_dl_idx = dl_idx_input
                            new_category_id = category_id_input
                            # annotation 파일 저장 시 dl_name, dl_idx, category_id도 반영
                            # 기존 save_anotation_items는 bbox만 저장하므로, 직접 파일 수정
                            ann_path = MyData.get_path_by_annotation_file(selected_ann.annotation_file)
                            with open(ann_path, encoding='utf-8') as f:
                                ann_data = json.load(f)
                            # images[0]에 반영
                            if 'images' in ann_data and isinstance(ann_data['images'], list) and ann_data['images']:
                                ann_data['images'][0]['dl_name'] = new_dl_name
                                ann_data['images'][0]['dl_idx'] = new_dl_idx
                            # annotations[0]에 반영
                            if 'annotations' in ann_data and isinstance(ann_data['annotations'], list) and ann_data['annotations']:
                                ann_data['annotations'][0]['bbox'] = new_bbox
                                ann_data['annotations'][0]['category_id'] = int(new_category_id) if str(new_category_id).isdigit() else -1
                                ann_data['annotations'][0]['dl_idx'] = new_dl_idx
                            with open(ann_path, 'w', encoding='utf-8') as f:
                                json.dump(ann_data, f, ensure_ascii=False, indent=2)
                            st.success(f"Annotation 파일이 저장되었습니다: {ann_path}")
                            MyData.set_Dtail_Type(1)  # 저장 후 전체 그리기
                        except Exception as e:
                            st.error(f"저장 오류: {e}")
                    elif add_clicked:
                        try:
                            left_f = float(left)
                            top_f = float(top)
                            width_f = float(width_)
                            height_f = float(height)
                            new_bbox = [left_f, top_f, width_f, height_f]
                            # 입력값 반영
                            new_dl_name = dl_name_input
                            new_dl_idx = dl_idx_input
                            new_category_id = category_id_input
                            MyData.add_annotation_item(
                                rel_image_file,
                                new_bbox,
                                new_dl_name,
                                new_category_id,
                                new_dl_idx
                            )
                            MyData.set_Dtail_Type(1)  # 추가 후 전체 그리기
                            # annotation 목록 업데이트
                            ann_candidates = MyData.search_Item_by_image(rel_image_file)
                            if ann_candidates:
                                selected_ann = ann_candidates[-1]  # 마지막 추가된 항목 선택
                                st.session_state[f'selected_ann_{image_file}'] = selected_ann
                        except Exception as e:
                            st.error(f"추가 오류: {e}")
                    elif delete_clicked:
                      MyData.delete_annotation_item(selected_ann.annotation_file)
                      MyData.set_Dtail_Type(1)  # 삭제 후 전체 그리기
                      st.session_state[f'selected_ann_{image_file}'] = None
                      st.warning(f"Annotation 파일이 삭제되었습니다: {selected_ann.annotation_file}")
                      # annotation 목록 업데이트 및 콤보박스 강제 초기화
                      ann_candidates = MyData.search_Item_by_image(rel_image_file)
                      st.session_state[f"anno_combo_{image_file}"] = None  # 콤보박스 강제 초기화
                      if ann_candidates:
                        selected_ann = ann_candidates[0]
                        st.session_state[f'selected_ann_{image_file}'] = selected_ann
                      else:
                        st.session_state[f'selected_ann_{image_file}'] = None
            except Exception as e:
                st.error(f"annotation 파일 읽기 오류: {e}\nann_labels: {ann_labels}\nselected_label: {selected_label}")
        else:
            st.warning("해당 이미지에 매칭되는 annotation 파일이 없습니다.")



def display_train_image(rows=2, cols=2):
  # 학습 이미지 디렉토리의 이미지들을 표시하는 함수
  # rows: 행 개수, cols: 열 개수 그리드로 표시
  # Previous/Next 버튼을 통해 이미지 페이지 전환 가능
  # 상세 이미지 영역을 미리 자리 선점 (placeholder 사용)
    print("display_train_image called")
    detail_placeholder = st.empty()
    selected_image = st.session_state.get('selected_train_image', None)
    # 상세 이미지가 있으면 먼저 그려줌
    with detail_placeholder:
        if selected_image:
            MyData.set_Dtail_Type(0)  # 목록 이미지 클릭
            edit_annotation_byImage(selected_image)
        else:
            st.markdown('<div style="min-height:120px;"></div>', unsafe_allow_html=True)
    print("display_train_image called")

    st.write("### 학습 이미지 샘플02")
    images_per_page = rows * cols
    page = st.session_state.get('train_image_page', 0)
    show_files = MyData.serach_imge_files(page * images_per_page, images_per_page)
    total_images = len(set([item.image_file for item in MyData.ANNOTAION_ITEMS])) if MyData.ANNOTAION_ITEMS else 0
    total_pages = (total_images + images_per_page - 1) // images_per_page if images_per_page > 0 else 1
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
      if st.button("◀ 이전", key="prev_train_img"):
        page = max(0, page - 1)
    with col3:
      if st.button("다음 ▶", key="next_train_img"):
        page = min(total_pages - 1, page + 1)
    st.session_state['train_image_page'] = page
    thumb_size = (80, 80) # 작은 아이콘 크기
    for r in range(rows):
      cols_list = st.columns(cols)
      for c in range(cols):
        idx = r * cols + c
        if idx < len(show_files):
          img_rel_path = show_files[idx]
          img_file = os.path.basename(img_rel_path)
          img_path = os.path.join(MyData.image_dir, img_rel_path)
          try:
            img = Image.open(img_path)
            img = img.copy()
            img.thumbnail(thumb_size)
            with cols_list[c]:
              st.image(img, caption=None, width=80)
              # 파일명 클릭 시 상세 이미지 표시
              if st.button(img_file, key=f"filebtn_{page}_{idx}"):
                st.session_state['selected_train_image'] = img_rel_path
                st.session_state['detail_image_draw'] = True
                MyData.set_Dtail_Type(0)  # 목록 이미지 클릭
                with detail_placeholder:
                    edit_annotation_byImage(img_rel_path)
          except Exception:
            cols_list[c].empty()
        else:
          cols_list[c].empty()

def display_eda():
  train_image_dir =   r"D:\01.project\EntryPrj\data\eda\noImage_okAnno\train_images" # 실제 경로로 변경하세요
  train_annotation_dir = r"D:\01.project\EntryPrj\data\eda\noImage_okAnno\train_annotations" #
  MyData.load_anotation(train_image_dir, train_annotation_dir)
  rows, cols = 3, 10
  display_train_image(rows=rows, cols=cols)

