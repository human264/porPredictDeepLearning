# app/start.py
import os
import uvicorn

def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    # 개발 중엔 기본 True, 운영에서는 False 권장
    reload = os.getenv("RELOAD", "true").lower() in ("1", "true", "yes")

    # 문자열 경로("app.main:app")를 쓰면 reload 동작이 가장 안정적입니다.
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=1,  # reload와 workers>1은 같이 못 씁니다.
    )

if __name__ == "__main__":
    main()
