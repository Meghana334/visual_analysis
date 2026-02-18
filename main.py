#!/usr/bin/env python3
"""
Integrated Image Crawler and Text Detector (EasyOCR Version)
Combines web image crawling with EasyOCR for text detection and WCAG contrast analysis
"""

import asyncio
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file (if needed for other things)
load_dotenv()

from crawl.crawler import AsyncImageCrawler
from ocr.text_detector import ImageTextDetector



async def run_crawler(url: str, max_depth: int = 0):
    """Run the async image crawler"""
    print("\n" + "=" * 60)
    print("STEP 1: WEB IMAGE CRAWLER")
    print("=" * 60 + "\n")

    crawler = AsyncImageCrawler(
        base_url=url,
        output_dir="crawled_images",
        include_data_uris=False,
        include_invisible=False
    )

    await crawler.crawl_page(url=url, max_depth=max_depth)
    crawler.save_results()
    return crawler.output_dir


def run_text_detector(source_dir: str):
    """Run the EasyOCR text detector on crawled images"""
    print("\n" + "=" * 60)
    print("STEP 2: TEXT DETECTION & CONTRAST ANALYSIS (EasyOCR)")
    print("=" * 60 + "\n")

    detector = ImageTextDetector(
        source_directory=source_dir
    )

    # Scan directory
    detector.scan_directory()
    detector.save_reports()
    return detector.text_detected_dir


async def main():
    """Main integrated workflow"""
    print("\n" + "=" * 80)
    print("INTEGRATED IMAGE CRAWLER & TEXT DETECTOR (EasyOCR + WCAG)")
    print("=" * 80 + "\n")

    # Configuration
    parser = argparse.ArgumentParser(description="Integrated Image Crawler and Text Detector")
    parser.add_argument("url", nargs="?", default="https://www.kao.com/global/en/", help="Target URL to crawl")
    parser.add_argument("--depth", type=int, default=0, help="Max crawl depth (0=single page)")
    args = parser.parse_args()

    website_url = args.url
    max_crawl_depth = args.depth

    print(f"Configuration:")
    print(f"  Target URL: {website_url}")
    print(f"  Max Crawl Depth: {max_crawl_depth}")
    print(f"  OCR Provider: EasyOCR")
    print()

    # Step 1: Crawl website for images
    crawl_output_dir = await run_crawler(website_url, max_crawl_depth)

    if not crawl_output_dir:
        print("\n✗ Crawling failed. Exiting.")
        sys.exit(1)

    print(f"\n✓ Crawling complete. Files saved to: {crawl_output_dir}")

    # Step 2: Detect text in crawled images using EasyOCR
    text_detection_dir = run_text_detector(crawl_output_dir)

    if not text_detection_dir:
        print("\n✗ Text detection failed.")
        sys.exit(1)

    # Final summary
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE ✅")
    print("=" * 80)
    print("\nOutputs:")
    print(f"  1. Crawled files: {crawl_output_dir}/")
    print(f"  2. Text detection & Contrast: {text_detection_dir}/")
    print(f"  3. Crawl report: {crawl_output_dir}/images_report.json")
    print(f"  4. Detection report: {text_detection_dir}/text_detection_report.json")
    print("\nLogs:")
    print(f"  - Text detector log: text_detection.log")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())