from fastapi import FastAPI
from pydantic import BaseModel
import requests
from lxml import html
from urllib.parse import urljoin
import logging
import sys
import aiohttp
from crawl.crawler import AsyncImageCrawler

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawler_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

class crawlRequest(BaseModel):
    url:str = "https://www.bluecaffeine.com/"
    max_depth:int = 0

@app.post("/crawl")
async def crawl(data:crawlRequest):
    output_dir:str = "crawled_images"
    logger.info(f"Configuration:")
    logger.info(f"  website_url: {data.url}")
    logger.info(f"  output_directory: {output_dir}")
    logger.info(f"  max_crawl_depth: {data.max_depth}")
    logger.info("Creating AsyncImageCrawler instance")
    crawler = AsyncImageCrawler(
        base_url=data.url,
        output_dir=output_dir
    )
    logger.info("Crawler instance created")
        # Start crawling
    try:
        logger.info("Starting crawl")
        await crawler.crawl_page(
            url=data.url,
            max_depth=data.max_depth
        )
        logger.info("Crawl completed successfully")
        return "Crawled Successfully"
    except KeyboardInterrupt:
        logger.warning("Crawl interrupted by user (KeyboardInterrupt)")
        print("\n\n⚠ Crawl interrupted by user")
    except Exception as e:
        logger.error(f"Crawl failed with error: {str(e)}")
        print(f"\n\n✗ Crawl failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return "error"







