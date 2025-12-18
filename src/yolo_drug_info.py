"""
식품의약품안전처 의약품개요정보 OpenAPI를 사용하여
yolo_class_mapping.json의 item_seq 기반으로 약물 정보를 조회하는 스크립트

API 문서: https://www.data.go.kr/data/15075057/openapi.do
"""

import json
import requests
import time
from typing import Dict, List, Optional
from urllib.parse import unquote

# 설정
BASE_URL = "http://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList"
SERVICE_KEY = "d4f215cfeaa74bb71b94270e1bc07cbe217bde62d91a4b0c094d2f059b2dec9c"# 실제 키로 교체 필요

# API 호출 설정
REQUEST_DELAY = 0.5  # 초 단위 (API 과부하 방지)
MAX_RETRIES = 3  # 최대 재시도 횟수
TIMEOUT = 10  # 초 단위


def load_item_seq_list(json_path: str = "yolo_class_mapping.json") -> List[str]:
    """
    yolo_class_mapping.json에서 item_seq 목록을 추출

    Args:
        json_path: JSON 파일 경로

    Returns:
        item_seq 문자열 리스트
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        item_seq_list = [
            str(v["item_seq"])
            for v in data.values()
            if "item_seq" in v
        ]

        print(f"✓ {len(item_seq_list)}개의 item_seq를 로드했습니다.")
        return item_seq_list

    except FileNotFoundError:
        print(f"✗ 오류: '{json_path}' 파일을 찾을 수 없습니다.")
        return []
    except json.JSONDecodeError:
        print(f"✗ 오류: '{json_path}' JSON 파싱 실패")
        return []
    except Exception as e:
        print(f"✗ 예상치 못한 오류: {e}")
        return []


def fetch_drug_info(item_seq: str, service_key: str, retry_count: int = 0) -> Optional[Dict]:
    """
    단일 item_seq에 대한 약물 정보를 API로 조회

    Args:
        item_seq: 품목기준코드
        service_key: API 서비스 키
        retry_count: 현재 재시도 횟수

    Returns:
        약물 정보 딕셔너리 또는 None
    """
    # 서비스 키 디코딩 (URL 인코딩된 키인 경우)
    decoded_key = unquote(service_key)

    params = {
        "serviceKey": decoded_key,
        "itemSeq": item_seq,
        "type": "json",
        "pageNo": "1",
        "numOfRows": "1"
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=TIMEOUT)

        # HTTP 상태 코드 확인
        if response.status_code != 200:
            print(f"  ✗ HTTP {response.status_code} 오류 (item_seq: {item_seq})")

            # 재시도 로직
            if retry_count < MAX_RETRIES:
                print(f"    → {retry_count + 1}번째 재시도 중...")
                time.sleep(REQUEST_DELAY * 2)  # 재시도 시 더 긴 딜레이
                return fetch_drug_info(item_seq, service_key, retry_count + 1)
            return None

        # JSON 파싱
        data = response.json()

        # API 응답 오류 코드 확인
        if "header" in data:
            result_code = data["header"].get("resultCode", "")
            result_msg = data["header"].get("resultMsg", "")

            if result_code != "00":
                print(f"  ✗ API 오류: [{result_code}] {result_msg} (item_seq: {item_seq})")
                return None

        # 실제 데이터 추출 (body > items)
        if "body" in data and "items" in data["body"]:
            items = data["body"]["items"]

            if items and len(items) > 0:
                # item_seq는 고유하므로 첫 번째 결과만 사용
                drug_info = items[0]
                print(f"  ✓ 성공: {drug_info.get('itemName', 'N/A')} (item_seq: {item_seq})")
                return drug_info
            else:
                print(f"  ✗ 데이터 없음 (item_seq: {item_seq})")
                return None
        else:
            print(f"  ✗ 응답 구조 오류 (item_seq: {item_seq})")
            return None

    except requests.exceptions.Timeout:
        print(f"  ✗ 타임아웃 (item_seq: {item_seq})")

        if retry_count < MAX_RETRIES:
            print(f"    → {retry_count + 1}번째 재시도 중...")
            time.sleep(REQUEST_DELAY * 2)
            return fetch_drug_info(item_seq, service_key, retry_count + 1)
        return None

    except requests.exceptions.RequestException as e:
        print(f"  ✗ 네트워크 오류: {e} (item_seq: {item_seq})")
        return None

    except Exception as e:
        print(f"  ✗ 예상치 못한 오류: {e} (item_seq: {item_seq})")
        return None


def batch_fetch_drug_info(item_seq_list: List[str], service_key: str) -> List[Dict]:
    """
    여러 item_seq에 대한 약물 정보를 일괄 조회

    Args:
        item_seq_list: item_seq 문자열 리스트
        service_key: API 서비스 키

    Returns:
        약물 정보 딕셔너리 리스트
    """
    results = []
    total = len(item_seq_list)
    success_count = 0

    print(f"\n{'='*60}")
    print(f"약물 정보 조회 시작 (총 {total}개)")
    print(f"{'='*60}\n")

    for idx, item_seq in enumerate(item_seq_list, 1):
        print(f"[{idx}/{total}] item_seq: {item_seq}")

        # API 호출
        drug_info = fetch_drug_info(item_seq, service_key)

        if drug_info:
            results.append(drug_info)
            success_count += 1

        # API 과부하 방지를 위한 딜레이 (마지막 요청 제외)
        if idx < total:
            time.sleep(REQUEST_DELAY)

    print(f"\n{'='*60}")
    print(f"조회 완료: 성공 {success_count}/{total}, 실패 {total - success_count}/{total}")
    print(f"{'='*60}\n")

    return results


def save_results(results: List[Dict], output_path: str = "drug_info_filtered.json"):
    """
    조회 결과를 JSON 파일로 저장

    Args:
        results: 약물 정보 딕셔너리 리스트
        output_path: 출력 파일 경로
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"✓ 결과를 '{output_path}'에 저장했습니다. (총 {len(results)}개)")

        # 샘플 데이터 출력
        if results:
            print(f"\n{'='*60}")
            print("샘플 데이터 (첫 번째 항목):")
            print(f"{'='*60}")
            print(json.dumps(results[0], ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"✗ 파일 저장 오류: {e}")


def main():
    """메인 실행 함수"""

    # 1. 서비스 키 확인
    if SERVICE_KEY == "여기에_발급받은_서비스키를_입력하세요":
        print("✗ 오류: SERVICE_KEY를 설정해주세요!")
        print("\n발급 방법:")
        print("1. https://www.data.go.kr 접속")
        print("2. '의약품개요정보(e약은요)조회서비스' 활용신청")
        print("3. 승인 후 일반 인증키(Decoding) 복사")
        print("4. 이 스크립트의 SERVICE_KEY 변수에 붙여넣기")
        return

    # 2. item_seq 목록 로드
    item_seq_list = load_item_seq_list("yolo_class_mapping.json")

    if not item_seq_list:
        print("✗ 조회할 item_seq가 없습니다.")
        return

    # 3. API 호출 시작
    results = batch_fetch_drug_info(item_seq_list, SERVICE_KEY)

    # 4. 결과 저장
    if results:
        save_results(results, "drug_info_filtered.json")
    else:
        print("✗ 저장할 데이터가 없습니다.")


if __name__ == "__main__":
    main()
