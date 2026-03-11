#!/usr/bin/env python3

import asyncio
import sys
from dotenv import load_dotenv

load_dotenv()

from crawl.crawler import AsyncImageCrawler
from ocr.text_detector import OCRPreprocessing, TextClassification
from config.logger import setup_logger
from utils.helper_modules.config_helper import load_config


# Load config
logger = setup_logger(name="KAC", tag="main")
logger.info("Logger initialized")
config = load_config()
logger.info("Configuration loaded successfully")



async def run_crawler(url: str, max_depth: int = 0):
    logger.info("Starting crawler step")
    logger.info(f"URL: {url}")
    logger.info(f"Max depth: {max_depth}")

    crawler = AsyncImageCrawler(
        base_url=url,
        max_depth=max_depth
    )

    await crawler.crawl_page()
    crawler.save_results()
    logger.info("Crawler step completed successfully")
    return crawler.output_dir


def run_text_detector(source_dir: str):

    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: TEXT DETECTION & CONTRAST ANALYSIS (EasyOCR)")
    logger.info("=" * 60 + "\n")

    detector = OCRPreprocessing(source_directory=source_dir)
    detector.scan_directory()

    save = TextClassification(source_directory=source_dir)
    save.results = detector.results
    save.save_reports()
    logger.info("OCR step completed successfully")

    return detector.text_detected_dir


async def main():
    logger.info("=" * 80)
    logger.info("INTEGRATED IMAGE CRAWLER & TEXT DETECTOR STARTED")
    logger.info("=" * 80)

    website_url = config["input"]["url"]
    max_crawl_depth = config["input"]["max_depth"]
    output_dir = config["input"]["output_dir"]

    logger.info("Configuration values:")
    logger.info(f"  URL: {website_url}")
    logger.info(f"  Max depth: {max_crawl_depth}")
    logger.info(f"  Output dir: {output_dir}")

    if not website_url:
        logger.error("URL missing in config.yaml under 'input'")
        print("✗ No URL found in config.yaml under 'input'")
        sys.exit(1)


    # STEP 1: Crawl
    crawl_output_dir   = await run_crawler(
        website_url,
        max_depth=max_crawl_depth
    )

    if not crawl_output_dir:
        logger.error("Crawling failed — exiting workflow")
        sys.exit(1)

    logger.info(f"Crawling complete. Output: {crawl_output_dir}")

    # STEP 2: OCR
    text_detection_dir = run_text_detector(
        crawl_output_dir
    )

    if not text_detection_dir:
        logger.error("Text detection failed")
        sys.exit(1)

    logger.info(f"OCR complete. Output: {text_detection_dir}")

    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE ✅")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  1. Crawled files: {crawl_output_dir}/")
    print(f"  2. Text detection & Contrast: {text_detection_dir}/")
    print(f"  3. Crawl report: {crawl_output_dir}/images_report.json")
    print(f"  4. Detection report: {text_detection_dir}/text_detection_report.json")
    print("=" * 80 + "\n")
    logger.info("Workflow completed successfully")


if __name__ == "__main__":
    asyncio.run(main())