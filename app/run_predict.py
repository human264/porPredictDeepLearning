# app/run_predict.py

"""
헤더별 예측 트리거 스크립트
- POST /ml/mr/predict_header
- POST /ml/act/predict_header

기본 동작:
  python run_predict.py
  → 두 엔드포인트(mr, act) 모두 호출 (환경변수 또는 기본값 사용)

옵션 예:
  python run_predict.py --only mr
  python run_predict.py --base-url http://0.0.0.0:8000 --wait 60 --topk 10
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, Tuple

import requests


def wait_for_server(base_url: str, wait_seconds: int) -> None:
    """FastAPI 서버가 응답할 때까지 /openapi.json을 폴링."""
    if wait_seconds <= 0:
        return
    url = base_url.rstrip("/") + "/openapi.json"
    deadline = time.time() + wait_seconds
    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.ok:
                return
        except Exception as e:
            last_err = e
        time.sleep(1.0)
    raise RuntimeError(f"Server not ready within {wait_seconds}s. Last error: {last_err}")


def post_json(url: str, payload: Dict[str, Any], timeout: Tuple[float, float] = (5.0, 120.0)) -> Dict[str, Any]:
    """POST 호출하고 JSON 응답을 dict로 반환. 실패 시 예외."""
    headers = {"Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        data = r.json()
    except Exception:
        data = {"non_json_response": r.text}

    if not r.ok:
        msg = data.get("detail") if isinstance(data, dict) else r.text
        raise RuntimeError(f"POST {url} failed ({r.status_code}): {msg}")
    return data


def predict_mr(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/ml/mr/predict_header"
    return post_json(url, payload)


def predict_act(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/ml/act/predict_header"
    return post_json(url, payload)


def pretty_print(title: str, obj: Dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def clamp_topk(v: int) -> int:
    return max(1, min(50, v))


def main() -> None:
    # 인자 없이 실행 가능하도록 전부 기본값/환경변수로 세팅
    parser = argparse.ArgumentParser(description="Trigger header prediction endpoints (defaults to calling both).")
    parser.add_argument("--base-url", default=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--only", choices=["mr", "act", "all"], default=os.getenv("ONLY", "all"))
    parser.add_argument("--wait", type=int, default=int(os.getenv("WAIT", "15")))

    # HeaderReq 필드: 환경변수 우선, 없으면 예시값
    parser.add_argument("--pjtno", default=os.getenv("PJTNO", "P1"))
    parser.add_argument("--porser", default=os.getenv("PORSER", "S1"))
    parser.add_argument("--porseq", default=os.getenv("PORSEQ", "Q1"))
    parser.add_argument("--revno", default=os.getenv("REVNO", "R1"))
    parser.add_argument("--topk", type=int, default=int(os.getenv("TOPK", "5")))

    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    # 서버 준비 대기(옵션)
    try:
        wait_for_server(base_url, args.wait)
    except Exception as e:
        print(f"[WARN] Server readiness check failed: {e}", file=sys.stderr)

    payload = {
        "pjtno": args.pjtno,
        "porser": args.porser,
        "porseq": args.porseq,
        "revno": args.revno,
        "topk": clamp_topk(args.topk),
    }

    print(f"[INFO] base_url={base_url} only={args.only} wait={args.wait} payload={payload}")

    exit_code = 0

    if args.only in ("mr", "all"):
        try:
            res = predict_mr(base_url, payload)
            pretty_print("MRNO Predict Result", res)
        except Exception as e:
            exit_code = 1
            print(f"[ERROR] MRNO predict failed: {e}", file=sys.stderr)

    if args.only in ("act", "all"):
        try:
            res = predict_act(base_url, payload)
            pretty_print("ACT_CODE:ACT_NO Predict Result", res)
        except Exception as e:
            exit_code = 1
            print(f"[ERROR] ACT predict failed: {e}", file=sys.stderr)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()


# # 두 엔드포인트 모두 호출
# python run_predict.py --pjtno P1 --porser S1 --porseq Q1 --revno R1 --topk 5
#
# # MR만
# python run_predict.py --only mr  --pjtno P1 --porser S1 --porseq Q1 --revno R1
#
# # ACT만
# python run_predict.py --only act --pjtno P1 --porser S1 --porseq Q1 --revno R1
#
# # 서버 base-url 변경 + 서버 준비 최대 60초 대기
# python run_predict.py --base-url http://0.0.0.0:8000 --wait 60 --pjtno P1 --porser S1 --porseq Q1 --revno R1
