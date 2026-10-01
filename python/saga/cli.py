from __future__ import annotations

import argparse
import asyncio
import sys

import uvicorn

from .config import get_settings
from .database import Database
from .indexer import PdfIndexer
from .law_api import LAW_CATALOG, LawApiIngestor
from .service_hub_client import ServiceHubReasoner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="saga", description="SAGA Python/Service Hub 관리 도구")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve", help="웹 서버 실행")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    index = subparsers.add_parser("index", help="saga-uploads의 PDF 색인")
    index.add_argument("--force", action="store_true", help="변경되지 않은 PDF도 다시 색인")
    index.add_argument("--service-hub-ocr", action="store_true", help="깨진 한글 페이지를 Service Hub 비전 OCR로 복구")
    law_sync = subparsers.add_parser("law-sync", help="국가법령정보센터 법령을 PDF로 저장하고 색인")
    law_sync.add_argument("keys", nargs="*", choices=sorted(LAW_CATALOG), help="법령 키(생략하면 전체 핵심 목록)")
    law_sync.add_argument("--force", action="store_true", help="오늘 생성된 PDF도 다시 받아서 색인")
    subparsers.add_parser("doctor", help="설정과 데이터 상태 점검")
    subparsers.add_parser("models", help="Service Hub에서 사용 가능한 모델 목록 확인")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()

    if args.command == "models":
        if not settings.service_hub_api_key:
            raise SystemExit("OPEN_AI_SERVICE_HUB_API_KEY 환경변수를 먼저 설정하세요.")
        reasoner = ServiceHubReasoner(settings)

        async def list_models() -> list[str]:
            try:
                return await reasoner.model_ids()
            finally:
                await reasoner.close()

        for model_id in asyncio.run(list_models()):
            print(model_id)
        return

    database = Database(settings.database_path)
    database.initialize()

    if args.command == "serve":
        uvicorn.run(
            "saga.api:app",
            host=args.host or settings.host,
            port=args.port or settings.port,
            reload=False,
        )
        return
    if args.command == "index":
        ocr_page = None
        if args.service_hub_ocr:
            from .service_hub_ocr import ServiceHubPageOcr

            ocr_page = ServiceHubPageOcr(
                settings.service_hub_api_key,
                settings.service_hub_vision_model,
                settings.service_hub_base_url,
            )
        indexer = PdfIndexer(database, ocr_page=ocr_page)

        def progress(current: int, total: int, result) -> None:
            print(f"[{current:>3}/{total}] {result.status:<9} {result.filename}")

        results = indexer.index_directory(settings.upload_dir, force=args.force, progress=progress)
        failures = [item for item in results if item.status == "failed"]
        print(
            f"완료: 전체 {len(results)}, 색인 {sum(item.status == 'indexed' for item in results)}, "
            f"OCR 필요 {sum(item.status == 'needs_ocr' for item in results)}, 실패 {len(failures)}"
        )
        raise SystemExit(1 if failures else 0)
    if args.command == "law-sync":
        if not settings.law_api_oc:
            raise SystemExit("SAGA_LAW_API_OC 환경변수를 먼저 설정하세요.")
        ingestor = LawApiIngestor(settings.law_api_oc, settings.law_api_base_url, settings.law_api_timeout)
        keys = args.keys or list(LAW_CATALOG)
        indexer = PdfIndexer(database)
        failures = 0
        for key in keys:
            try:
                rendered = ingestor.fetch_and_render(key, settings.law_pdf_dir, force=args.force)
                result = indexer.index_pdf(rendered.pdf_path, force=args.force)
                print(f"{key:<8} {result.status:<9} {rendered.name} · {rendered.pdf_path}")
                failures += result.status == "failed"
            except Exception as exc:
                failures += 1
                print(f"{key:<8} failed    {exc}")
        raise SystemExit(1 if failures else 0)
    if args.command == "doctor":
        stats = database.stats()
        print(f"Python: {sys.version.split()[0]}")
        print(f"Service Hub API key: {'설정됨' if settings.service_hub_api_key else '없음'}")
        print(f"Service Hub URL: {settings.service_hub_base_url}")
        print(f"Answer model: {settings.service_hub_model}")
        print(f"Fast model: {settings.service_hub_fast_model}")
        print(f"Upload directory: {settings.upload_dir} ({len(list(settings.upload_dir.glob('*.pdf')))} PDFs)")
        print(f"Database: {settings.database_path} ({stats['documents']} documents, {stats['chunks']} chunks)")
        if not settings.service_hub_api_key:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
