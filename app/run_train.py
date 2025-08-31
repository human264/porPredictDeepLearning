# app/run_train.py
"""
FastAPI 학습 엔드포인트 트리거 스크립트.
- 기본 호출 대상:
  POST /ml/mr/train_db
  POST /ml/act/train_db

사용 예:
  python run_train.py                 # 두 모델 모두 학습
  python run_train.py --only mr       # MRNO만 학습
  python run_train.py --only act      # ACT_CODE:ACT_NO만 학습
  python run_train.py --base-url http://0.0.0.0:8000 --wait 60

환경변수(있으면 우선 적용):
  API_BASE_URL  (기본: http://127.0.0.1:8000)
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


def post_json(url: str, payload: Dict[str, Any] | None = None, timeout: Tuple[float, float] = (5.0, 120.0)) -> Dict[str, Any]:
    """POST 호출하고 JSON 응답을 dict로 반환. 실패 시 예외."""
    headers = {"Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    # FastAPI 에러는 보통 {"detail": "..."} 형태
    try:
        data = r.json()
    except Exception:
        data = {"non_json_response": r.text}

    if not r.ok:
        msg = data.get("detail") if isinstance(data, dict) else r.text
        raise RuntimeError(f"POST {url} failed ({r.status_code}): {msg}")
    return data


def train_mr(base_url: str) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/ml/mr/train_db"
    return post_json(url)


def train_act(base_url: str) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/ml/act/train_db"
    return post_json(url)


def pretty_print(title: str, obj: Dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Trigger FastAPI training endpoints.")
    parser.add_argument("--base-url", default=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"),
                        help="API base URL (default: env API_BASE_URL or http://127.0.0.1:8000)")
    parser.add_argument("--only", choices=["mr", "act", "all"], default="all",
                        help="Which training to run (default: all)")
    parser.add_argument("--wait", type=int, default=30,
                        help="Seconds to wait for server readiness via /openapi.json (default: 30, 0=skip)")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    try:
        wait_for_server(base_url, args.wait)
    except Exception as e:
        print(f"[WARN] Server readiness check failed: {e}", file=sys.stderr)

    exit_code = 0

    if args.only in ("mr", "all"):
        try:
            res = train_mr(base_url)
            pretty_print("MRNO Train Result", res)
        except Exception as e:
            exit_code = 1
            print(f"[ERROR] MRNO training failed: {e}", file=sys.stderr)

    if args.only in ("act", "all"):
        try:
            res = train_act(base_url)
            pretty_print("ACT_CODE:ACT_NO Train Result", res)
        except Exception as e:
            exit_code = 1
            print(f"[ERROR] ACT training failed: {e}", file=sys.stderr)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
