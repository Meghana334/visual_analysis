from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import sys
from playwright.async_api import async_playwright
from crawl.crawler import AsyncImageCrawler
from ocr.text_detector import ImageTextDetector


# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI()

class CrawlRequest(BaseModel):
    url: str = "https://www.bluecaffeine.com/"
    max_depth: int = 0
    include_data_uris: bool = False
    include_invisible: bool = False

@app.post("/run_crawler")
async def run_crawler(data: CrawlRequest):
    try:
        crawler = AsyncImageCrawler(
            base_url=data.url,
            output_dir="crawled_images",  
            include_data_uris=True,
            include_invisible=True
        )

        await crawler.crawl_page(url=data.url, max_depth=0)

        total_img_tags = len(crawler.images_data)

        hidden_images = sum(
            1 for img in crawler.images_data 
            if getattr(img, "is_visible", True) is False
        )

        data_uris = sum(
            1 for img in crawler.images_data
             if getattr(img, "is_data_uri", False)
        )

        crawler.save_results()

        return {
            "status": "success",
            "url": data.url,
            "pages_crawled": len(crawler.visited_urls),
            "total_img_tags": total_img_tags,
            "hidden_images": hidden_images,
            "data_uris": data_uris,
            "output_dir": crawler.output_dir
        }

    except Exception as e:
        logger.error(str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/classify-images")
async def classify_images(data: CrawlRequest):
    try:
        crawler = AsyncImageCrawler(
            base_url=data.url,
            output_dir="classified_images",
            include_data_uris=data.include_data_uris,
            include_invisible=data.include_invisible
        )

        await crawler.crawl_page(url=data.url, max_depth=data.max_depth)
        crawler.save_results()

        summary = {
            "informative": 0,
            "decorative": 0,
            "functional": 0
        }

        for img in crawler.images_data:
            if img.classification in summary:
                summary[img.classification] += 1

        return {
            "status": "success",
            "url": data.url,
            "pages_crawled": len(crawler.visited_urls),
            "total_images": len(crawler.images_data),
            "classification_summary": summary,
            "output_dir": crawler.output_dir
        }

    except Exception as e:
        logger.error(str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Image classification failed")


@app.post("/detect-text")
async def detect_text(data: CrawlRequest):
    try:
        crawler = AsyncImageCrawler(
            base_url=data.url,
            output_dir="crawled_images",
            include_data_uris=data.include_data_uris,
            include_invisible=data.include_invisible
        )

        await crawler.crawl_page(url=data.url, max_depth=data.max_depth)
        crawler.save_results()

        detector = ImageTextDetector(source_directory=crawler.output_dir)
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
        raise HTTPException(status_code=500, detail="Text detection failed")