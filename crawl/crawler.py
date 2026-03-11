import os
import json
import csv
import hashlib
import asyncio
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright
from pathlib import Path
import time
from typing import Optional, List, Set
from pydantic import BaseModel, Field
from datetime import datetime
import aiohttp
from crawl.models import ImageData
from config.logger import setup_logger
from image_processing.classify_assets import ClassifyAssets
from utils.helper_modules.config_helper import load_config
CONFIG = load_config()


logger = setup_logger(name="KAC", tag="crawler")
logger.info("Logger initialized")


class CrawlSummary(BaseModel):
    """Summary statistics for the crawl"""
    total_images: int = 0
    informative: int = 0
    decorative: int = 0
    functional: int = 0
    complex: int = 0
    text_images: int = 0
    functional_buttons: int = 0
    functional_icons: int = 0
    functional_logos: int = 0
    functional_images: int = 0
    pages_crawled: int = 0


class CrawlReport(BaseModel):
    """Complete crawl report"""
    base_url: str
    crawl_date: str
    summary: CrawlSummary
    sub_type_breakdown: dict[str, int] = Field(default_factory=dict)
    images: List[ImageData] = Field(default_factory=list)

class _ClassificationResult:
    """Adapter to convert classify_image dict to attribute-style access"""
    def __init__(self, d: dict):
        self.type = d["classification"]
        self.sub_type = d.get("sub_type")
        self.is_text_image = d.get("is_text_image", False)
        self.is_functional = d.get("is_functional", False)
        self.is_decorative = d.get("is_decorative", False)
        self.is_complex = d.get("is_complex", False)
        self.is_logo = d.get("is_logo", False)
        self.is_icon = d.get("is_icon", False)
        self.is_button = d.get("is_button", False)
        self.file_format = d.get("file_format")


class AsyncImageCrawler:
    def __init__(
        self,
        base_url: str,
        max_depth:int):
        self.base_url = base_url
        self.max_depth = max_depth
        self.include_invisible = CONFIG["crawler"]["include_invisible"]
        self.images_data: List[ImageData] = []
        self.visited_urls: Set[str] = set()


        ### Creating unique output directory with domain and timestamp
        base_output_dir = CONFIG["input"]["output_dir"]   # crawled_images
        domain = urlparse(base_url).netloc.replace('www.', '').replace('.', '_')
        timestamp = time.strftime('%m%d_%H%M')
        self.output_dir = f"{base_output_dir}/{domain}_{timestamp}"

        ### Initializing image classifier
        self.classifier = ClassifyAssets(output_dir=self.output_dir)

        # Create directory structure
        self._create_directories()
        self.file_handler = None  # basic handler not needed as we handle paths locally
        logger.info(f"Initializing AsyncImageCrawler with base_url={base_url}")


    def _create_directories(self):
        """Create all necessary output directories"""

        base_dir = Path(self.output_dir)
        directories = [base_dir] + [
            base_dir / subdir for subdir in CONFIG["directories"]
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Successfully created :  {directories} directories")


    async def trigger_lazy_loading(self, page):
        """More aggressive lazy loading trigger"""
        logger.info("Starting lazy loading trigger")

        # Multiple scroll passes with different patterns
        scroll_passes = CONFIG["crawler"]["scroll_passes"]
        for scroll_pass in range(scroll_passes):
            logger.info(f"Starting scroll pass {scroll_pass + 1}/{scroll_passes}")

            # Scroll down in steps
            logger.info("Executing scroll down in steps JavaScript")
            await page.evaluate('''() => {
                const scrollHeight = document.body.scrollHeight;
                const steps = 5;
                for (let i = 0; i <= steps; i++) {
                    setTimeout(() => {
                        window.scrollTo(0, (scrollHeight / steps) * i);
                    }, i * 200);
                }
            }''')
            await page.wait_for_timeout(2000)

            # Scroll to middle
            await page.evaluate('''() => {
                window.scrollTo(0, document.body.scrollHeight / 2);
            }''')
            await page.wait_for_timeout(1000)
            logger.info("Middle scroll complete")

        # Scroll back to top
        logger.info("Scrolling back to top")
        await page.evaluate('window.scrollTo(0, 0)')
        await page.wait_for_timeout(1000)

        # Trigger intersection observers by scrolling images into view
        logger.info("Triggering intersection observers for lazy-loaded images")
        await page.evaluate('''() => {
            document.querySelectorAll('img[data-src], img[loading="lazy"]').forEach(img => {
                img.scrollIntoView({ behavior: 'instant', block: 'center' });
            });
        }''')
        await page.wait_for_timeout(1500)
        logger.info("Lazy loading trigger complete")


    def save_results(self):
        """Save results to JSON file using Pydantic models"""
        logger.info("Starting save_results")
        output_file = f"{self.output_dir}/images_report.json"
        logger.debug(f"Output file: {output_file}")

        # --- Create summary ---
        logger.info("Creating crawl summary")
        summary = CrawlSummary(
            total_images=len(self.images_data),
            informative=sum(1 for img in self.images_data if img.classification == 'informative'),
            decorative=sum(1 for img in self.images_data if img.classification == 'decorative'),
            functional=sum(1 for img in self.images_data if img.classification == 'functional'),
            complex=sum(1 for img in self.images_data if img.classification == 'complex'),
            text_images=sum(1 for img in self.images_data if img.is_text_image),
            functional_buttons=sum(
                1 for img in self.images_data
                if img.classification == 'functional' and img.sub_type == 'buttons'
            ),
            functional_icons=sum(
                1 for img in self.images_data
                if img.classification == 'functional' and img.sub_type == 'icons'
            ),
            functional_logos=sum(
                1 for img in self.images_data
                if img.classification == 'functional' and img.sub_type == 'logos'
            ),
            functional_images=sum(
                1 for img in self.images_data
                if img.classification == 'functional' and img.sub_type == 'images'
            ),
            pages_crawled=len(self.visited_urls)
        )
        logger.debug(f"Summary: {summary}")

        # --- Sub-type breakdown ---
        logger.info("Creating sub-type breakdown")
        sub_type_breakdown: dict[str, int] = {}
        for img in self.images_data:
            if img.sub_type:
                sub_type_breakdown[img.sub_type] = sub_type_breakdown.get(img.sub_type, 0) + 1
        logger.debug(f"Sub-type breakdown: {sub_type_breakdown}")

        # --- Final report ---
        logger.info("Creating final crawl report")
        report = CrawlReport(
            base_url=self.base_url,
            crawl_date=datetime.utcnow().isoformat(),
            summary=summary,
            sub_type_breakdown=sub_type_breakdown,
            images=self.images_data
        )

        # --- Write JSON ---
        logger.info(f"Writing report to {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Report saved successfully to {output_file}")

        print("\n" + "=" * 60)
        print("CRAWL COMPLETE ✅")
        print(f"Images captured: {summary.total_images}")
        print(f"Pages crawled: {summary.pages_crawled}")
        print(f"Report saved to: {output_file}")
        print(f"Debug log saved to: crawler_debug.log")
        print("=" * 60)

        # Export images with alt text to CSV
        logger.info("Exporting images with alt text to CSV")
        self.export_alt_text_csv()

    def export_alt_text_csv(self):
        """Export images with alt text to a CSV file"""
        logger.info("Starting export_alt_text_csv")
        csv_file = f"{self.output_dir}/images_with_alt_text.csv"
        logger.debug(f"CSV output file: {csv_file}")

        # Export images with alt text to CSV
        images_to_export = [img for img in self.images_data if img.alt_text and img.alt_text.strip()]
        logger.info(f"Preparing to export {len(images_to_export)} images to CSV")

        if len(images_to_export) == 0:
            logger.warning("No images found, skipping CSV export")
            print("⚠ No images found")
            return

        # Define CSV headers
        headers = [
            'Image Filename',
            'Image URL',
            'Image Path',
            'Alt Text',
            'Title',
            'Classification',
            'Sub Type',
            'Is Logo',
            'Is Icon',
            'Is Button',
            'Is Functional',
            'Is Decorative',
            'Is Text Image',
            'File Format'
        ]

        # Write CSV
        logger.info(f"Writing CSV to {csv_file}")
        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()

                for img in images_to_export:
                    writer.writerow({
                        'Image Filename': img.filename,
                        'Image URL': img.url,
                        'Image Path': img.screenshot_path,
                        'Alt Text': img.alt_text,
                        'Title': img.title,
                        'Classification': img.classification,
                        'Sub Type': img.sub_type or '',
                        'Is Logo': 'Yes' if img.is_logo else 'No',
                        'Is Icon': 'Yes' if img.is_icon else 'No',
                        'Is Button': 'Yes' if img.is_button else 'No',
                        'Is Functional': 'Yes' if img.is_functional else 'No',
                        'Is Decorative': 'Yes' if img.is_decorative else 'No',
                        'Is Text Image': 'Yes' if img.is_text_image else 'No',
                        'File Format': img.file_format or 'N/A'
                    })

            logger.info(f"✓ CSV export successful: {csv_file}")
            print(f"\n✓ Exported {len(images_to_export)} images to: {csv_file}")

        except Exception as e:
            logger.error(f"Error writing CSV: {str(e)}")
            print(f"✗ Error exporting CSV: {str(e)}")

    async def reveal_hidden_images(self, page):
        """Click tabs, accordions, etc. to reveal hidden images"""
        logger.info("Starting reveal_hidden_images")

        revealed_count = 0

        # Click all tabs
        logger.info("Attempting to expand click tabs")
        try:
            tabs = await page.locator('[role="tab"], .tab, [data-toggle="tab"], .nav-link').all() #TODO: it is  ststic see all possibility
            logger.info(f"Found {len(tabs)} tab elements")

            for idx, tab in enumerate(tabs[:8]):  # Limit to avoid infinite loops
                logger.debug(f"Processing tab {idx + 1}/8")
                try:
                    if await tab.is_visible(timeout=1000):
                        await tab.click(timeout=2000)
                        await page.wait_for_timeout(500)
                        revealed_count += 1
                        logger.info(f"Successfully clicked tab {idx + 1}")
                    else:
                        logger.info(f"Tab {idx + 1} not visible, skipping")
                except Exception as e:
                    logger.info(f"Failed to click tab {idx + 1}: {str(e)}")
                    pass
        except Exception as e:
            logger.warning(f"Error processing tabs: {str(e)}")
            pass

        # Expand accordions
        logger.info("Attempting to expand accordions")
        try:
            accordions = await page.locator('[data-toggle="collapse"], .accordion-toggle, .accordion-button').all()
            logger.info(f"Found {len(accordions)} accordion elements")

            for idx, accordion in enumerate(accordions[:8]):
                try:
                    if await accordion.is_visible(timeout=1000):
                        await accordion.click(timeout=2000)
                        await page.wait_for_timeout(500)
                        revealed_count += 1
                        logger.info(f"Successfully clicked accordion {idx + 1}")
                    else:
                        logger.info(f"Accordion {idx + 1} not visible, skipping")
                except Exception as e:
                    logger.info(f"Failed to click accordion {idx + 1}: {str(e)}")
                    pass
        except Exception as e:
            logger.warning(f"Error processing accordions: {str(e)}")
            pass

        # Click carousel/slider controls
        logger.info("Attempting to click carousel controls")
        try:
            carousel_btns = await page.locator(
                '.carousel-control, .slider-next, .slick-next, [data-slide="next"]').all()
            logger.info(f"Found {len(carousel_btns)} carousel control elements")

            for idx, btn in enumerate(carousel_btns[:5]):
                try:
                    if await btn.is_visible(timeout=1000):
                        await btn.click(timeout=2000)
                        await page.wait_for_timeout(500)
                        revealed_count += 1
                        logger.info(f"Successfully clicked carousel button {idx + 1}")
                    else:
                        logger.info(f"Carousel button {idx + 1} not visible, skipping")
                except Exception as e:
                    logger.debug(f"Failed to click carousel button {idx + 1}: {str(e)}")
                    pass
        except Exception as e:
            logger.warning(f"Error processing carousel controls: {str(e)}")
            pass

        ### Total count
        logger.info(f"reveal_hidden_images complete. Revealed {revealed_count} elements")

    async def is_actually_visible(self, element, page):
        """More comprehensive visibility check"""
        logger.debug("Checking element visibility")
        try:
            visibility = await element.evaluate('''el => {
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                const tag = el.tagName.toLowerCase();
                // Only require src for <img> elements
                const srcOk = tag !== 'img' || !!(
                    el.src || el.getAttribute('data-src') || el.getAttribute('data-lazy-src')
                );
                return {
                    hasSize: rect.width > 0 && rect.height > 0,
                    isDisplayed: style.display !== 'none',
                    isVisible: style.visibility !== 'hidden',
                    hasOpacity: parseFloat(style.opacity) > 0,
                    srcOk: srcOk,
                };
            }''')
            logger.info(f"Visibility check result: {visibility}")

            is_visible = (
                    visibility['hasSize'] and
                    visibility['isDisplayed'] and
                    visibility['isVisible'] and
                    visibility['hasOpacity'] and
                    visibility['srcOk']
            )
            logger.info(f"Element visibility final result: {is_visible}")
            return is_visible

        except Exception as e:
            logger.warning(f"Error checking visibility: {str(e)}")
            return False

    async def crawl_page(self, current_depth: int = 0):
        """Crawl a single page and extract images"""
        logger.info(f"Starting crawl_page: url={self.base_url}, max_depth={self.max_depth}, current_depth={current_depth}")

        if self.base_url in self.visited_urls or current_depth > self.max_depth:
            logger.info(f"Skipping {self.base_url}: already visited or max depth exceeded")
            return

        logger.info(f"Adding {self.base_url} to visited_urls")
        self.visited_urls.add(self.base_url)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Crawling: {self.base_url}")
        logger.info(f"Depth: {current_depth}/{self.max_depth}")
        logger.info(f"{'=' * 60}")

        logger.info("Initializing Playwright")
        async with async_playwright() as p:
            logger.info("Launching Chromium browser")
            browser = await p.chromium.launch(headless=True)
            logger.info("Browser launched successfully")

            logger.info("Creating browser context")
            context = await browser.new_context(
                viewport={'width': CONFIG['crawl_browser']['width'], 'height': CONFIG['crawl_browser']['HEIGHT']},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            page = await context.new_page()
            logger.info("Browser context and Page created")


            # Set longer timeout for slow pages
            logger.info("Setting default timeout to 60000ms")
            page.set_default_timeout(60000)

            try:
                # Try different wait strategies
                logger.info(f"Attempting to load page: {self.base_url}")
                try:
                    await page.goto(self.base_url, wait_until='domcontentloaded', timeout=60000)
                    logger.info("✓ DOM loaded successfully")
                except Exception as e:
                    logger.info(f"domcontentloaded failed: {str(e)}")
                    logger.info("Trying 'load' wait strategy")

                    await page.goto(self.base_url, wait_until='load', timeout=60000)
                    logger.info("✓ Page loaded with 'load' strategy")

                # Wait for images to load
                logger.info("Waiting 3000ms for images to load")
                await page.wait_for_timeout(3000)
                logger.info("Initial wait complete")

                # Better lazy loading ( Scrolling webpage of the url )
                await self.trigger_lazy_loading(page)

                # Reveal hidden content
                await self.reveal_hidden_images(page)

                # Wait for new images to load
                logger.info("Waiting 2000ms for newly revealed images")
                await page.wait_for_timeout(2000)
                logger.info("Final wait complete")


                # Find all images
                logger.info("Locating all <img> elements on page")
                images = await page.locator('img').all()
                logger.info(f"✓ Found {len(images)} <img> elements")

                if len(images) == 0:
                    logger.info("No images found on page")

                # Process each urlimage
                skipped_invisible = 0
                skipped_no_src = 0










                logger.info(f"Starting to process {len(images)} images")
                for idx, img in enumerate(images):
                    logger.info(f"\n{'=' * 40}")
                    logger.info(f"Processing image {idx + 1}/{len(images)}")
                    try:
                        # === IMPROVED VISIBILITY CHECK ===
                        if not self.include_invisible:
                            logger.info("Checking image visibility (include_invisible=False)")
                            try:
                                is_visible = await self.is_actually_visible(img, page)
                                if not is_visible:
                                    logger.info(f"Image {idx + 1} is not visible, skipping")
                                    skipped_invisible += 1
                                    continue
                                logger.info(f"Image {idx + 1} is visible")
                            except Exception as visibility_error:
                                logger.warning(f"Visibility check failed for image {idx + 1}: {str(visibility_error)}")
                                skipped_invisible += 1
                                continue

                        # Get image properties
                        src = await img.get_attribute('src')
                        if not src:
                            logger.info("No src attribute, checking data-src and data-lazy-src")
                            # Try data-src for lazy loaded images
                            src = await img.get_attribute('data-src') or await img.get_attribute('data-lazy-src')

                        if not src:
                            logger.info(f"Image {idx + 1} has no src, skipping")
                            skipped_no_src += 1
                            continue

                        # Make absolute URL
                        absolute_src = urljoin(self.base_url, src)

                        # Get image info
                        alt_text = await img.get_attribute('alt') or ''
                        title = await img.get_attribute('title') or ''

                        # Classify image
                        classification_dict = await self.classifier.classify_image(img)
                        classification = _ClassificationResult(classification_dict)

                        # Generate unique filename
                        img_hash = self.classifier.get_image_hash(absolute_src)
                        filename = f"img_{img_hash}.png"

                        sub_path = None

                        if classification.type == "functional":
                            sub_path = f"functional/{classification.sub_type or 'images'}"

                        elif classification.type == "complex":
                            sub_path = f"complex/{classification.sub_type or 'charts'}"

                        else:
                            sub_path = classification.type  # informative / decorative

                        img_dir = os.path.join(self.output_dir, sub_path)

                        # Ensure directory exists (only if declared in config)
                        allowed_dirs = set(CONFIG["directories"])

                        if sub_path not in allowed_dirs:
                            img_dir = os.path.join(self.output_dir, classification.type)
                        os.makedirs(img_dir, exist_ok=True)

                        screenshot_path = f"{img_dir}/{filename}"
                        logger.info(f"Full screenshot path: {screenshot_path}")

                        # === CHECK FOR OVERLAY CONTAINER ===
                        logger.info("Checking for overlay container")
                        is_overlay_container = False
                        container = img  # default to img itself

                        try:
                            container_handle = await self.classifier.get_visual_container(img, page)
                            is_overlay_container = await page.evaluate(
                                '(args) => args[0] !== args[1]',
                                [container_handle, img]
                            )
                            if is_overlay_container:
                                container = container_handle
                                logger.info("Overlay container detected — will screenshot container instead of image")

                        except Exception as e:
                            logger.warning(f"Error checking for overlay container: {e}")

                        # === SCREENSHOT vs DOWNLOAD DECISION ===
                        # Priority order (icons/buttons checked BEFORE overlay):
                        #   Icons         → always download original (screenshot = blank tiny PNG)
                        #   Button-images → download original (<img> acting as button)
                        #   Logos         → screenshot (preserves brand context)
                        #   Overlays      → screenshot container (only for plain images)
                        #   Plain         → download original file

                        try:
                            if classification.is_icon:
                                # ── ICON: download original first — MUST be checked before overlay ──
                                # Icons are inside <a> links so is_overlay_container fires first otherwise.
                                # element.screenshot() on a ~40px transparent icon → blank ~121B PNG.
                                original_ext = os.path.splitext(urlparse(absolute_src).path)[1] or '.png'
                                if not filename.endswith(original_ext):
                                    filename = f"img_{img_hash}{original_ext}"
                                    screenshot_path = f"{img_dir}/{filename}"

                                async with aiohttp.ClientSession() as session:
                                    downloaded = await self.classifier._download_file(
                                        session, absolute_src, screenshot_path
                                    )

                                if downloaded:
                                    logger.info(f"✓ Icon downloaded: {screenshot_path}")
                                else:
                                    # Fallback: screenshot the nearest visible parent container
                                    logger.warning(f"Icon download failed, falling back to parent screenshot")
                                    screenshot_path = f"{img_dir}/img_{img_hash}.png"
                                    filename = f"img_{img_hash}.png"
                                    try:
                                        parent_handle = await img.evaluate_handle(
                                            'el => el.closest("a, button, li, td, div") || el.parentElement'
                                        )
                                        await parent_handle.screenshot(path=screenshot_path)
                                        logger.info(f"✓ Parent screenshot taken: {screenshot_path}")
                                    except Exception as pe:
                                        logger.error(f"Icon parent screenshot failed: {pe}")
                                        continue

                            elif classification.is_button:
                                # ── BUTTON-as-IMAGE (<img> classified as button) → download original ──
                                # The element is an <img> acting as a button; download preserves quality.
                                original_ext = os.path.splitext(urlparse(absolute_src).path)[1] or '.png'
                                if not filename.endswith(original_ext):
                                    filename = f"img_{img_hash}{original_ext}"
                                    screenshot_path = f"{img_dir}/{filename}"

                                async with aiohttp.ClientSession() as session:
                                    downloaded = await self.classifier._download_file(
                                        session, absolute_src, screenshot_path
                                    )

                                if downloaded:
                                    logger.info(f"✓ Button-image downloaded: {screenshot_path}")
                                else:
                                    # Fallback: screenshot the img itself
                                    logger.warning(f"Button-image download failed, falling back to screenshot")
                                    screenshot_path = f"{img_dir}/img_{img_hash}.png"
                                    filename = f"img_{img_hash}.png"
                                    await img.screenshot(path=screenshot_path)
                                    logger.info(f"✓ Button-image screenshot (fallback): {screenshot_path}")

                            elif classification.is_logo:
                                # Screenshot logos in page context
                                await img.screenshot(path=screenshot_path)
                                logger.info(f"✓ Screenshot taken (logo): {screenshot_path}")

                            elif is_overlay_container:
                                # Plain image with overlay text → screenshot the container
                                await container.screenshot(path=screenshot_path)
                                logger.info(f"✓ Screenshot taken (overlay): {screenshot_path}")

                            else:
                                # Plain image — download original file, preserve extension
                                original_ext = os.path.splitext(urlparse(absolute_src).path)[1]
                                if original_ext and not filename.endswith(original_ext):
                                    filename = f"img_{img_hash}{original_ext}"
                                    screenshot_path = f"{img_dir}/{filename}"

                                async with aiohttp.ClientSession() as session:
                                    logger.debug(f"Downloading original image to {screenshot_path}")
                                    success = await self.classifier._download_file(
                                        session, absolute_src, screenshot_path
                                    )

                                if success:
                                    logger.info(f"✓ Image downloaded: {screenshot_path}")
                                else:
                                    logger.info(f"Failed to download {absolute_src}, skipping")
                                    continue  # skip storing this image

                            # Print classification details
                            logger.info(f"    Type: {classification.type}")
                            if classification.sub_type:
                                logger.info(f"    Sub-type: {classification.sub_type}")
                            if classification.is_logo:
                                logger.info(f"    Logo: Yes")
                            if classification.is_icon:
                                logger.info(f"    Icon: Yes")
                            if classification.is_button:
                                logger.info(f"    Button Image: Yes")
                            if alt_text:
                                logger.info(f"    Alt text: {alt_text[:60]}{'...' if len(alt_text) > 60 else ''}")

                        except Exception as e:
                            logger.error(f"Failed to capture image {idx + 1}: {str(e)}")
                            continue

                        # Store image data using Pydantic model
                        logger.info("Creating ImageData object")
                        image_data = ImageData(
                            url=self.base_url,
                            src=self.base_url,
                            alt_text=alt_text,
                            title=title,
                            classification=classification.type,
                            sub_type=classification.sub_type,
                            is_functional=classification.is_functional,
                            is_decorative=classification.is_decorative,
                            is_complex=classification.is_complex,
                            is_text_image=classification.is_text_image,
                            is_logo=classification.is_logo,
                            is_icon=classification.is_icon,
                            is_button=classification.is_button,
                            file_format=classification.file_format,
                            screenshot_path=screenshot_path,
                            filename=filename
                        )

                        self.images_data.append(image_data)
                        logger.info(f"✓ Successfully processed image {idx + 1}")

                    except Exception as e:
                        logger.error(f"Error processing image {idx}: {str(e)}")
                        continue






































                # # Print skip statistics
                # logger.info("Image processing complete, printing summary")
                # print(f"\n{'=' * 60}")
                # print(f"IMAGE PROCESSING SUMMARY:")
                # print(f"  Total found: {len(images)}")
                # print(f"  Successfully captured: {len([img for img in self.images_data if img.url == url])}")
                # if self.include_data_uris:
                #     print(f"    - Data URIs: {data_uri_count}")
                #     print(
                #         f"    - Regular images: {len([img for img in self.images_data if img.url == url]) - data_uri_count}")
                # print(f"  Skipped - Invisible: {skipped_invisible}")
                # print(f"  Skipped - No dimensions: {skipped_no_dimensions}")
                # print(f"  Skipped - No src: {skipped_no_src}")
                # if not self.include_data_uris:
                #     print(f"  Skipped - Data URI: {skipped_data_uri}")
                # print(f"{'=' * 60}")

                logger.info(
                    f"Summary - Total: {len(images)}, Captured: {len([img for img in self.images_data if img.url == self.base_url])}, Skipped: {skipped_invisible + skipped_no_src }")

                # ═══════════════════════════════════════════════════════
                # BUTTON EXTRACTION PASS
                # The <img> loop above never sees <button>, <input type=submit>,
                # or [role=button] elements. This pass captures them separately.
                # ═══════════════════════════════════════════════════════
                logger.info("Starting button extraction pass")
                print(f"\n{'=' * 60}")
                print("Extracting standalone button elements...")

                btn_selector = (
                    'button, '
                    'input[type="button"], input[type="submit"], input[type="reset"], '
                    '[role="button"]'
                )
                btn_elements = await page.locator(btn_selector).all()
                logger.info(f"Found {len(btn_elements)} button elements")
                print(f"✓ Found {len(btn_elements)} button elements")

                btn_dir = f"{self.output_dir}/functional/buttons"
                os.makedirs(btn_dir, exist_ok=True)
                captured_btns = 0
                seen_btn_hashes: set[str] = set()

                for btn_idx, btn_el in enumerate(btn_elements):
                    try:
                        # Visibility check
                        is_visible = await self.is_actually_visible(btn_el, page)
                        if not is_visible:
                            logger.debug(f"Button {btn_idx+1} not visible, skipping")
                            continue

                        # Get accessible text for identification
                        btn_info = await btn_el.evaluate('''el => ({
                            tag: el.tagName.toLowerCase(),
                            text: (el.textContent || el.value || el.getAttribute("aria-label") || "").trim().slice(0, 80),
                            type: el.getAttribute("type") || "",
                            cls:  (el.className || "").slice(0, 60)
                        })''')

                        # Deduplicate by outer-HTML hash
                        btn_html = await btn_el.evaluate('el => el.outerHTML.slice(0, 200)')
                        btn_hash = hashlib.md5(btn_html.encode()).hexdigest()[:12]
                        if btn_hash in seen_btn_hashes:
                            logger.debug(f"Button {btn_idx+1} duplicate, skipping")
                            continue
                        seen_btn_hashes.add(btn_hash)

                        btn_filename = f"btn_{btn_hash}.png"
                        btn_path = f"{btn_dir}/{btn_filename}"

                        logger.debug(f"Screenshotting button {btn_idx+1}: {btn_info}")
                        await btn_el.screenshot(path=btn_path)

                        # Verify screenshot has real content:
                        #   1. File must be > 200 bytes (rejects totally blank PNGs)
                        #   2. Pixel dims must be >= 20x10 (rejects visually collapsed 10x11px buttons)
                        file_size = os.path.getsize(btn_path)
                        if file_size < 200:
                            logger.warning(f"Button screenshot too small ({file_size}B), skipping")
                            os.remove(btn_path)
                            continue

                        # Read PNG dimensions from header (no PIL required)
                        try:
                            import struct
                            with open(btn_path, 'rb') as _f:
                                _f.read(16)  # skip PNG sig + length + type
                                _w = struct.unpack('>I', _f.read(4))[0]
                                _h = struct.unpack('>I', _f.read(4))[0]
                            if _w < 20 or _h < 10:
                                logger.warning(f"Button too small ({_w}x{_h}px), skipping")
                                os.remove(btn_path)
                                continue
                        except Exception:
                            pass  # if we can't read dims, allow through

                        captured_btns += 1
                        btn_label = btn_info['text'] or f"<{btn_info['tag']}>"
                        print(f"  ✓ Button captured: {btn_filename} — \"{btn_label}\"")
                        logger.info(f"✓ Button screenshot: {btn_path} ({btn_label})")

                        # Store in images_data
                        image_data = ImageData(
                            url=self.base_url,
                            src=self.base_url,  # buttons have no src
                            alt_text=btn_info['text'],
                            title=btn_info['text'],
                            classification='functional',
                            sub_type='buttons',
                            is_functional=True,
                            is_decorative=False,
                            is_complex=False,
                            is_text_image=False,
                            is_logo=False,
                            is_icon=False,
                            is_button=True,
                            file_format='png',
                            screenshot_path=btn_path,
                            filename=btn_filename
                        )
                        self.images_data.append(image_data)

                    except Exception as e:
                        logger.debug(f"Failed to capture button {btn_idx+1}: {e}")
                        continue
                logger.info(f"Button extraction complete: {captured_btns} captured")

                # Find links for crawling (optional)
                if current_depth < self.max_depth:
                    logger.info(f"Current depth {current_depth} < max depth {self.max_depth}, finding links")
                    print(f"\n{'=' * 60}")
                    print(f"Finding links for deeper crawling...")

                    links = await page.locator('a[href]').all()
                    logger.info(f"Found {len(links)} links")

                    link_count = 0
                    for link_idx, link in enumerate(links[:10]):  # Limit to 10 links per page
                        logger.debug(f"Processing link {link_idx + 1}/10")
                        try:
                            logger.debug("Getting href attribute")
                            href = await link.get_attribute('href')
                            if href:
                                logger.debug(f"Link href: {href}")
                                absolute_url = urljoin(self.base_url, href)
                                logger.debug(f"Absolute URL: {absolute_url}")

                                # Only crawl same domain
                                logger.debug("Checking if link is same domain")
                                if urlparse(absolute_url).netloc == urlparse(self.base_url).netloc:
                                    if absolute_url not in self.visited_urls:
                                        logger.info(f"Found new link to crawl: {absolute_url}")
                                        link_count += 1

                                        logger.debug("Closing context and browser for recursive crawl")
                                        await context.close()
                                        await browser.close()

                                        logger.info(f"Recursively crawling: {absolute_url}")
                                        await self.crawl_page(absolute_url, self.max_depth, current_depth + 1)

                                        logger.debug("Relaunching browser after recursive crawl")
                                        browser = await p.chromium.launch(headless=True)
                                        context = await browser.new_context(
                                            viewport={'width': 1920, 'height': 1080},
                                            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                                        )
                                        page = await context.new_page()
                                        page.set_default_timeout(60000)
                                        await page.goto(self.base_url, wait_until='domcontentloaded')
                                    else:
                                        logger.info(f"Link already visited: {absolute_url}")
                                else:
                                    logger.info(f"Link is different domain, skipping: {absolute_url}")
                        except Exception as e:
                            logger.warning(f"Error processing link {link_idx + 1}: {str(e)}")
                            continue

                    logger.info(f"Found {link_count} new links to crawl")

            except Exception as e:
                logger.error(f"Error crawling {self.base_url}: {str(e)}")
                import traceback
                traceback.print_exc()

            finally:
                logger.info("Closing browser context")
                await context.close()
                logger.info("Closing browser")
                await browser.close()
                logger.info(f"Finished crawling {self.base_url}")


async def main():
    logger.info("=" * 60)
    logger.info("ASYNC IMAGE CRAWLER - STARTING")
    logger.info("=" * 60)

    # Configuration
    website_url = "https://www.bluecaffeine.com/"
    output_directory = "crawled_images"
    max_crawl_depth = 0  # 0 = single page, 1 = page + linked pages, etc.

    logger.info(f"Configuration:")
    logger.info(f"  website_url: {website_url}")
    logger.info(f"  output_directory: {output_directory}")
    logger.info(f"  max_crawl_depth: {max_crawl_depth}")

    print(f"\n{'=' * 60}")
    print(f"ASYNC IMAGE CRAWLER - STARTING")
    print(f"{'=' * 60}")
    print(f"Target URL: {website_url}")
    print(f"Output Directory: {output_directory}")
    print(f"Max Depth: {max_crawl_depth}")
    print(f"{'=' * 60}\n")

    # Create crawler instance
    logger.info("Creating AsyncImageCrawler instance")
    crawler = AsyncImageCrawler(
        base_url=website_url,
        output_dir=output_directory
    )
    logger.info("Crawler instance created")

    # Start crawling
    try:
        logger.info("Starting crawl")
        await crawler.crawl_page(
            url=website_url,
            max_depth=max_crawl_depth
        )
        logger.info("Crawl completed successfully")
    except KeyboardInterrupt:
        logger.warning("Crawl interrupted by user (KeyboardInterrupt)")
        print("\n\n⚠ Crawl interrupted by user")
    except Exception as e:
        logger.error(f"Crawl failed with error: {str(e)}")
        print(f"\n\n✗ Crawl failed with error: {str(e)}")
        import traceback
        traceback.print_exc()



#
#
# if __name__ == "__main__":
#     logger.info("Starting main program")
#     asyncio.run(main())
#     logger.info("Program finished")


