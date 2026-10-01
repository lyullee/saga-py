from __future__ import annotations

import base64

import pymupdf
from openai import OpenAI


class ServiceHubPageOcr:
    """OCR fallback for PDFs whose embedded Korean font map is broken."""

    def __init__(self, api_key: str, model: str, base_url: str):
        if not api_key:
            raise ValueError("Service Hub OCR에는 OPEN_AI_SERVICE_HUB_API_KEY가 필요합니다.")
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=120.0,
            max_retries=1,
        )
        self.model = model

    def __call__(self, page: pymupdf.Page) -> str:
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False, colorspace=pymupdf.csRGB)
        image = base64.b64encode(pixmap.tobytes("jpeg", jpg_quality=82)).decode("ascii")
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "이 한국어 규정 PDF 페이지를 OCR 하세요. 보이는 모든 문자를 읽는 순서대로 정확히 전사하고, "
                                "조항 번호·수치·단위·표의 내용을 보존하세요. 해설이나 마크다운 없이 전사 텍스트만 출력하세요."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=6000,
        )
        return (completion.choices[0].message.content or "").strip()

    def close(self) -> None:
        self.client.close()
