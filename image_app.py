from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import sys
from crawl.crawler import AsyncImageCrawler

# Configure logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("crawler.log"),
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

#Crawl 
async def run_crawler(data: CrawlRequest, output_folder: str):
    crawler = AsyncImageCrawler(
        base_url=data.url,
        output_dir=output_folder,
        include_data_uris=data.include_data_uris,
        include_invisible=data.include_invisible
    )

    await crawler.crawl_page(
        url=data.url,
        max_depth=data.max_depth
    )

    crawler.save_results()
    return crawler

@app.post("/classify-images")
async def classify_images(data: CrawlRequest):
    logger.info(f"Image crawl + classification started for {data.url}")

    try:
        crawler = await run_crawler(data, "classified_images")

        classification_summary = {
            "informative": 0,
            "decorative": 0,
            "functional": 0
        }

        for img in crawler.images_data:
            if img.classification in classification_summary:
                classification_summary[img.classification] += 1

        return {
            "status": "success",
            "url": data.url,
            "pages_crawled": len(crawler.visited_urls),
            "total_images": len(crawler.images_data),
            "classification_summary": classification_summary,
            "output_dir": crawler.output_dir
        }

    except Exception as e:
        logger.error(str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Image crawl and classification failed"
        )
