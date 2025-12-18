def draw_box_with_override(img, ann_candidates, image_file, selected_ann=None, override_bbox=None, width=450, placeholder=None):
  img_draw = img.copy()
  draw = ImageDraw.Draw(img_draw)
  for ann_path in ann_candidates:
    try:
      with open(ann_path, encoding='utf-8') as f:
        ann_data = json.load(f)
      if 'annotations' in ann_data and isinstance(ann_data['annotations'], list):
        for idx, ann in enumerate(ann_data['annotations']):
          if ann_path == selected_ann and override_bbox is not None and idx == 0:
            # 첫 annotation만 입력값으로 그림
            if isinstance(override_bbox, list) and len(override_bbox) == 4:
              try:
                x, y, w, h = [float(v) for v in override_bbox]
                draw.rectangle([x, y, x + w, y + h], outline='blue', width=4)
              except Exception:
                pass
          elif 'bbox' in ann and isinstance(ann['bbox'], list) and len(ann['bbox']) == 4:
            x, y, w, h = ann['bbox']
            draw.rectangle([x, y, x + w, y + h], outline='red', width=4)
    except Exception:
      pass
  if placeholder is not None:
    placeholder.image(img_draw, caption=f"상세 보기: {os.path.basename(image_file)} (수정 bbox: 파랑, 기존: 빨강)", width=width)
  else:
    st.image(img_draw, caption=f"상세 보기: {os.path.basename(image_file)} (수정 bbox: 파랑, 기존: 빨강)", width=width)

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

# 실행 방법: streamlit run D:\01.project\EntryPrj\src\streamit_edaexecute.py


# -------------------------------
# bbox 그리기
# -------------------------------
def draw_box(img, ann_candidates, image_file, width=450, placeholder=None):
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    for ann_path in ann_candidates:
        try:
            with open(ann_path, encoding='utf-8') as f:
                ann_data = json.load(f)
            if 'annotations' in ann_data and isinstance(ann_data['annotations'], list):
                for ann in ann_data['annotations']:
                    if 'bbox' in ann and isinstance(ann['bbox'], list) and len(ann['bbox']) == 4:
                        x, y, w, h = ann['bbox']
                        draw.rectangle([x, y, x + w, y + h], outline='red', width=4)
        except Exception:
            pass
    if placeholder is not None:
        placeholder.image(img_draw, caption=f"상세 보기: {os.path.basename(image_file)} (모든 bbox)", width=width)
    else:
        st.image(img_draw, caption=f"상세 보기: {os.path.basename(image_file)} (모든 bbox)", width=width)



def edit_annotation_byImage(image_file, annotation_dir=None):
    if not image_file or not os.path.exists(image_file):
        st.warning(f"이미지 파일을 찾을 수 없습니다: {image_file}")
        st.session_state['selected_train_image'] = None
        return

    base_name = os.path.splitext(os.path.basename(image_file))[0]
    ann_candidates = []
    if annotation_dir and os.path.isdir(annotation_dir):
        for root, _, files in os.walk(annotation_dir):
            for f in files:
                if f.endswith('.json') and base_name in f:
                    ann_candidates.append(os.path.join(root, f))

    col_img, col_anno = st.columns([1, 2])
    with col_img:
        img_placeholder = st.empty()
        try:
            img = Image.open(image_file)
            draw_clicked_key = f'draw_clicked_{image_file}'
            draw_clicked = st.session_state.get(draw_clicked_key, False)
            selected_ann = None
            override_bbox = None
            if ann_candidates:
                if draw_clicked:
                    selected_ann = st.session_state.get(f'selected_ann_{image_file}', None)
                    override_bbox = st.session_state.get(f'override_bbox_{image_file}', None)
                    draw_box_with_override(img, ann_candidates, image_file, selected_ann=selected_ann, override_bbox=override_bbox, width=450, placeholder=img_placeholder)
                    st.session_state[draw_clicked_key] = False
                else:
                    draw_box(img, ann_candidates, image_file, width=450, placeholder=img_placeholder)
            else:
                img_placeholder.image(img, caption=f"상세 보기: {os.path.basename(image_file)}", width=450)
        except Exception as e:
            img_placeholder.error(f"상세 보기 오류: {os.path.basename(image_file)}\n{image_file}\n{e}")
            st.session_state['selected_train_image'] = None

    with col_anno:
        if ann_candidates:
            ann_candidates.sort()
            ann_labels = [os.path.relpath(f, annotation_dir) for f in ann_candidates]
            selected_label = st.selectbox("Annotation 파일 선택", ann_labels, index=0, key=f"anno_combo_{image_file}")
            selected_ann = os.path.normpath(os.path.join(annotation_dir, selected_label))
            st.session_state[f'selected_ann_{image_file}'] = selected_ann
            try:
                with open(selected_ann, encoding='utf-8') as f:
                    anno_data = json.load(f)
                dl_name = anno_data.get('dl_name')
                if not dl_name and 'images' in anno_data and anno_data['images']:
                    dl_name = anno_data['images'][0].get('dl_name')
                if dl_name:
                    st.info(f"dl_name: {dl_name}")
                else:
                    st.warning("선택한 annotation에 dl_name 항목이 없습니다.")
                    st.warning(f"실제 annotation 파일 전체 경로: {selected_ann}")

                bbox = None
                if 'annotations' in anno_data and anno_data['annotations']:
                    ann0 = anno_data['annotations'][0]
                    if 'bbox' in ann0 and isinstance(ann0['bbox'], list) and len(ann0['bbox']) == 4:
                        bbox = ann0['bbox']
                if not bbox:
                    st.warning('해당 annotation에 bbox 정보가 없거나 올바르지 않습니다.')
                else:
                    b1, b2, b3, b4 = bbox
                    cols = st.columns(4)
                    left = cols[0].text_input('left', value=str(b1), key=f'bbox0_{selected_ann}')
                    top = cols[1].text_input('top', value=str(b2), key=f'bbox1_{selected_ann}')
                    width_ = cols[2].text_input('width', value=str(b3), key=f'bbox2_{selected_ann}')
                    height = cols[3].text_input('height', value=str(b4), key=f'bbox3_{selected_ann}')
                    st.session_state[f'override_bbox_{image_file}'] = [left, top, width_, height]
                    btn1, btn2, btn3 = st.columns([1, 1, 1])
                    draw_clicked = btn1.button('전체 그리기', key=f'draw_{selected_ann}')
                    save_clicked = btn2.button('저장', key=f'save_{selected_ann}')
                    add_clicked = btn3.button('추가', key=f'add_{selected_ann}')
                    if draw_clicked:
                        # flag를 True로 유지 → rerun 후에도 draw_box 실행
                        st.session_state[f'draw_clicked_{image_file}'] = True
            except Exception as e:
                st.error(f"annotation 파일 읽기 오류: {e}")
        else:
            st.warning("해당 이미지에 매칭되는 annotation 파일이 없습니다.")



def display_train_image(train_image_dir,train_annotation_dir, rows=2, cols=2):
  # 학습 이미지 디렉토리의 이미지들을 표시하는 함수
  # rows: 행 개수, cols: 열 개수 그리드로 표시
  # Previous/Next 버튼을 통해 이미지 페이지 전환 가능
  # 상세 이미지 영역을 미리 자리 선점 (placeholder 사용)
  detail_placeholder = st.empty()
  selected_image = st.session_state.get('selected_train_image', None)
  # 상세 이미지가 있으면 먼저 그려줌
  # 항상 상세 이미지 영역을 먼저 렌더링 (selected_image가 있으면 표시, 없으면 빈 영역)
  with detail_placeholder:
    if selected_image:
      edit_annotation_byImage(selected_image, train_annotation_dir)
    else:
      st.markdown('<div style="min-height:120px;"></div>', unsafe_allow_html=True)
  st.write("### 학습 이미지 샘플02")
  if not os.path.exists(train_image_dir):
    st.error(f"디렉토리 없음: {train_image_dir}")
    return
  image_files = [f for f in os.listdir(train_image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]
  if not image_files:
    st.warning("이미지 파일이 없습니다.")
    return
  image_files.sort()
  images_per_page = rows * cols
  total_pages = (len(image_files) + images_per_page - 1) // images_per_page
  page = st.session_state.get('train_image_page', 0)
  col1, col2, col3 = st.columns([1,2,1])
  with col1:
    if st.button("◀ 이전", key="prev_train_img"):
      page = max(0, page - 1)
  with col3:
    if st.button("다음 ▶", key="next_train_img"):
      page = min(total_pages - 1, page + 1)
  st.session_state['train_image_page'] = page
  start_idx = page * images_per_page
  end_idx = min(start_idx + images_per_page, len(image_files))
  show_files = image_files[start_idx:end_idx]
  thumb_size = (80, 80) # 작은 아이콘 크기
  for r in range(rows):
    cols_list = st.columns(cols)
    for c in range(cols):
      idx = r * cols + c
      if idx < len(show_files):
        img_file = show_files[idx]
        img_path = os.path.join(train_image_dir, img_file)
        try:
          img = Image.open(img_path)
          img = img.copy()
          img.thumbnail(thumb_size)
          with cols_list[c]:
            st.image(img, caption=None, width=80)
            # 파일명 클릭 시 상세 이미지 표시
            if st.button(img_file, key=f"filebtn_{page}_{idx}"):
              st.session_state['selected_train_image'] = img_path
              # 클릭 즉시 상세 이미지 갱신
              with detail_placeholder:
                edit_annotation_byImage(img_path, train_annotation_dir)
        except Exception:
          cols_list[c].empty()
      else:
        cols_list[c].empty()
  st.caption(f"페이지 {page+1} / {total_pages} (총 {len(image_files)}장)")

def excute_execute_eda():
  train_image_dir =   r"D:\01.project\EntryPrj\data\eda\noImage_okAnno\train_images" # 실제 경로로 변경하세요
  train_annotation_dir = r"D:\01.project\EntryPrj\data\eda\noImage_okAnno\train_annotations" #
  MyData.
  rows, cols = 3, 10
  display_train_image(train_image_dir, train_annotation_dir,rows=rows, cols=cols)

