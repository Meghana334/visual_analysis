from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import sys
from crawl.crawler import AsyncImageCrawler
from ocr.text_detector import ImageTextDetector

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("text_detection.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI()

class TextDetectionRequest(BaseModel):
    url: str = "https://www.bluecaffeine.com/"
    max_depth: int = 0
    include_data_uris: bool = False
    include_invisible: bool = False

# Crawl
async def run_crawler(data: TextDetectionRequest, output_dir: str):
    crawler = AsyncImageCrawler(
        base_url=data.url,
        output_dir=output_dir,
        include_data_uris=data.include_data_uris,
        include_invisible=data.include_invisible
    )

    await crawler.crawl_page(
        url=data.url,
        max_depth=data.max_depth
    )

    crawler.save_results()
    return crawler.output_dir

@app.post("/detect-text")
async def detect_text(data: TextDetectionRequest):
    logger.info(f"Text detection started for {data.url}")

    try:
        # Crawl images
        crawl_dir = await run_crawler(data, "crawled_images")

        # Text detection 
        detector = ImageTextDetector(source_directory=crawl_dir)
        detector.scan_directory()
        detector.save_reports()

        summary = {
            "detected_images": sum(1 for r in detector.results if r.has_text),
            "informational_text": sum(1 for r in detector.results if r.category == "informational_text"),
            "logo_text": sum(1 for r in detector.results if r.category == "logo_text"),
            "with_text": sum(1 for r in detector.results if r.category == "with_text")
        }

        return {
            "status": "success",
            "url": data.url,
            "total_images_scanned": len(detector.results),
            "summary": summary,
            "output_dir": detector.text_detected_dir,
            "report": f"{detector.text_detected_dir}/text_detection_report.json"
        }

    except Exception as e:
        logger.error(str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Text detection failed"
        )   