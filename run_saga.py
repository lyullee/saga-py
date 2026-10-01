"""PyCharm에서 이 파일을 실행하면 SAGA 웹 서버가 시작됩니다."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python"))

import uvicorn  # noqa: E402

from saga.config import get_settings  # noqa: E402


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("saga.api:app", host=settings.host, port=settings.port, reload=False)
