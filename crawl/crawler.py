import os
import json
import csv
import hashlib
import asyncio
import base64
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright
from pathlib import Path
import time
from typing import Optional, List, Set
from pydantic import BaseModel, Field
from datetime import datetime
import logging
import sys
import aiohttp

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


class ImageData(BaseModel):
    """Data structure for a single image"""
    url: str
    src: str
    alt_text: str = ""
    title: str = ""
    classification: str
    sub_type: Optional[str] = None
    is_functional: bool = False
    is_decorative: bool = False
    is_complex: bool = False
    is_text_image: bool = False
    is_logo: bool = False
    is_icon: bool = False
    is_button: bool = False
    file_format: Optional[str] = None
    screenshot_path: str
    filename: str


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


class AsyncImageCrawler:
    def __init__(self, base_url: str, output_dir: str = "crawled_images",
                 include_data_uris: bool = False,
                 include_invisible: bool = False):
        logger.info(f"Initializing AsyncImageCrawler with base_url={base_url}")
        logger.debug(
            f"output_dir={output_dir}, include_data_uris={include_data_uris}, include_invisible={include_invisible}")

        self.base_url = base_url
        logger.debug(f"Set self.base_url to {base_url}")

        self.include_data_uris = include_data_uris
        logger.debug(f"Set self.include_data_uris to {include_data_uris}")

        self.include_invisible = include_invisible
        logger.debug(f"Set self.include_invisible to {include_invisible}")

        # Create unique output directory based on domain
        logger.info("Creating unique output directory based on domain")
        domain = urlparse(base_url).netloc.replace('www.', '').replace('.', '_')
        logger.debug(f"Extracted domain: {domain}")

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        logger.debug(f"Generated timestamp: {timestamp}")

        self.output_dir = f"{output_dir}/{domain}_{timestamp}"
        logger.info(f"Output directory will be: {self.output_dir}")

        self.images_data: List[ImageData] = []
        logger.debug("Initialized empty images_data list")

        self.visited_urls: Set[str] = set()
        logger.debug("Initialized empty visited_urls set")

        # Create directory structure
        logger.info("Calling _create_directories()")
        self._create_directories()

        self.file_handler = None  # basic handler not needed as we handle paths locally

    def _create_directories(self):
        """Create all necessary output directories"""
        logger.info("Starting directory creation")

        directories = [
            self.output_dir,
            f"{self.output_dir}/informative",
            f"{self.output_dir}/decorative",
            f"{self.output_dir}/functional",
            f"{self.output_dir}/functional/buttons",
            f"{self.output_dir}/functional/icons",
            f"{self.output_dir}/functional/logos",
            f"{self.output_dir}/functional/images",
            f"{self.output_dir}/text_images",
        ]
        logger.debug(f"Directory list prepared: {len(directories)} directories")

        for directory in directories:
            logger.debug(f"Creating directory: {directory}")
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"✓ Directory created/verified: {directory}")

        logger.info(f"Successfully created/verified all {len(directories)} directories")

    async def trigger_lazy_loading(self, page):
        """More aggressive lazy loading trigger"""
        logger.info("Starting lazy loading trigger")
        print(f"Triggering lazy image loading...")

        # Multiple scroll passes with different patterns
        for scroll_pass in range(3):
            logger.info(f"Starting scroll pass {scroll_pass + 1}/3")
            print(f"  Scroll pass {scroll_pass + 1}/3...")

            # Scroll down in steps
            logger.debug("Executing scroll down in steps JavaScript")
            await page.evaluate('''() => {
                const scrollHeight = document.body.scrollHeight;
                const steps = 5;
                for (let i = 0; i <= steps; i++) {
                    setTimeout(() => {
                        window.scrollTo(0, (scrollHeight / steps) * i);
                    }, i * 200);
                }
            }''')
            logger.debug("Scroll JavaScript executed, waiting 2000ms")
            await page.wait_for_timeout(2000)
            logger.debug("Wait complete after scroll")

            # Scroll to middle
            logger.debug("Scrolling to middle of page")
            await page.evaluate('''() => {
                window.scrollTo(0, document.body.scrollHeight / 2);
            }''')
            logger.debug("Waiting 1000ms after middle scroll")
            await page.wait_for_timeout(1000)
            logger.debug("Middle scroll complete")

        # Scroll back to top
        logger.info("Scrolling back to top")
        await page.evaluate('window.scrollTo(0, 0)')
        logger.debug("Waiting 1000ms after scroll to top")
        await page.wait_for_timeout(1000)
        logger.debug("Scroll to top complete")

        # Trigger intersection observers by scrolling images into view
        logger.info("Triggering intersection observers for lazy-loaded images")
        await page.evaluate('''() => {
            document.querySelectorAll('img[data-src], img[loading="lazy"]').forEach(img => {
                img.scrollIntoView({ behavior: 'instant', block: 'center' });
            });
        }''')
        logger.debug("Waiting 1500ms for intersection observers")
        await page.wait_for_timeout(1500)
        logger.info("Lazy loading trigger complete")

        print(f"  ✓ Lazy loading complete")

    async def reveal_hidden_images(self, page):
        """Click tabs, accordions, etc. to reveal hidden images"""
        logger.info("Starting reveal_hidden_images")
        print("Attempting to reveal hidden images...")

        revealed_count = 0

        # Click all tabs
        logger.debug("Attempting to click tabs")
        try:
            logger.debug("Locating tab elements")
            tabs = await page.locator('[role="tab"], .tab, [data-toggle="tab"], .nav-link').all()
            logger.info(f"Found {len(tabs)} tab elements")

            for idx, tab in enumerate(tabs[:8]):  # Limit to avoid infinite loops
                logger.debug(f"Processing tab {idx + 1}/8")
                try:
                    logger.debug(f"Checking if tab {idx + 1} is visible")
                    if await tab.is_visible(timeout=1000):
                        logger.debug(f"Tab {idx + 1} is visible, clicking")
                        await tab.click(timeout=2000)
                        logger.debug(f"Tab {idx + 1} clicked, waiting 500ms")
                        await page.wait_for_timeout(500)
                        revealed_count += 1
                        logger.info(f"Successfully clicked tab {idx + 1}")
                    else:
                        logger.debug(f"Tab {idx + 1} not visible, skipping")
                except Exception as e:
                    logger.debug(f"Failed to click tab {idx + 1}: {str(e)}")
                    pass
        except Exception as e:
            logger.warning(f"Error processing tabs: {str(e)}")
            pass

        # Expand accordions
        logger.debug("Attempting to expand accordions")
        try:
            logger.debug("Locating accordion elements")
            accordions = await page.locator('[data-toggle="collapse"], .accordion-toggle, .accordion-button').all()
            logger.info(f"Found {len(accordions)} accordion elements")

            for idx, accordion in enumerate(accordions[:8]):
                logger.debug(f"Processing accordion {idx + 1}/8")
                try:
                    logger.debug(f"Checking if accordion {idx + 1} is visible")
                    if await accordion.is_visible(timeout=1000):
                        logger.debug(f"Accordion {idx + 1} is visible, clicking")
                        await accordion.click(timeout=2000)
                        logger.debug(f"Accordion {idx + 1} clicked, waiting 500ms")
                        await page.wait_for_timeout(500)
                        revealed_count += 1
                        logger.info(f"Successfully clicked accordion {idx + 1}")
                    else:
                        logger.debug(f"Accordion {idx + 1} not visible, skipping")
                except Exception as e:
                    logger.debug(f"Failed to click accordion {idx + 1}: {str(e)}")
                    pass
        except Exception as e:
            logger.warning(f"Error processing accordions: {str(e)}")
            pass

        # Click carousel/slider controls
        logger.debug("Attempting to click carousel controls")
        try:
            logger.debug("Locating carousel button elements")
            carousel_btns = await page.locator(
                '.carousel-control, .slider-next, .slick-next, [data-slide="next"]').all()
            logger.info(f"Found {len(carousel_btns)} carousel control elements")

            for idx, btn in enumerate(carousel_btns[:5]):
                logger.debug(f"Processing carousel button {idx + 1}/5")
                try:
                    logger.debug(f"Checking if carousel button {idx + 1} is visible")
                    if await btn.is_visible(timeout=1000):
                        logger.debug(f"Carousel button {idx + 1} is visible, clicking")
                        await btn.click(timeout=2000)
                        logger.debug(f"Carousel button {idx + 1} clicked, waiting 500ms")
                        await page.wait_for_timeout(500)
                        revealed_count += 1
                        logger.info(f"Successfully clicked carousel button {idx + 1}")
                    else:
                        logger.debug(f"Carousel button {idx + 1} not visible, skipping")
                except Exception as e:
                    logger.debug(f"Failed to click carousel button {idx + 1}: {str(e)}")
                    pass
        except Exception as e:
            logger.warning(f"Error processing carousel controls: {str(e)}")
            pass

        logger.info(f"reveal_hidden_images complete. Revealed {revealed_count} elements")
        print(f"  ✓ Clicked {revealed_count} interactive elements")

    async def is_actually_visible(self, element, page):
        """More comprehensive visibility check"""
        logger.debug("Checking element visibility")
        try:
            # Check if element is in viewport and has dimensions
            logger.debug("Evaluating element visibility properties")
            visibility = await element.evaluate('''el => {
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);

                return {
                    hasSize: rect.width > 0 && rect.height > 0,
                    isDisplayed: style.display !== 'none',
                    isVisible: style.visibility !== 'hidden',
                    hasOpacity: parseFloat(style.opacity) > 0,
                    src: el.src || el.getAttribute('data-src') || el.getAttribute('data-lazy-src') || '',
                };
            }''')
            logger.debug(f"Visibility check result: {visibility}")

            # Image must meet all conditions
            is_visible = (visibility['hasSize'] and
                          visibility['isDisplayed'] and
                          visibility['isVisible'] and
                          visibility['hasOpacity'] and
                          visibility['src'])  # Must have source

            logger.debug(f"Element visibility final result: {is_visible}")
            return is_visible

        except Exception as e:
            logger.warning(f"Error checking visibility: {str(e)}")
            return False

    async def crawl_page(self, url: str, max_depth: int = 2, current_depth: int = 0):
        """Crawl a single page and extract images"""
        logger.info(f"Starting crawl_page: url={url}, max_depth={max_depth}, current_depth={current_depth}")

        if url in self.visited_urls or current_depth > max_depth:
            logger.warning(f"Skipping {url}: already visited or max depth exceeded")
            return

        logger.info(f"Adding {url} to visited_urls")
        self.visited_urls.add(url)

        print(f"\n{'=' * 60}")
        print(f"Crawling: {url}")
        print(f"Depth: {current_depth}/{max_depth}")
        print(f"{'=' * 60}")

        logger.info("Initializing Playwright")
        async with async_playwright() as p:
            logger.debug("Launching Chromium browser")
            browser = await p.chromium.launch(headless=True)
            logger.debug("Browser launched successfully")

            logger.debug("Creating browser context")
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            logger.debug("Browser context created")

            logger.debug("Creating new page")
            page = await context.new_page()
            logger.debug("Page created")

            # Set longer timeout for slow pages
            logger.debug("Setting default timeout to 60000ms")
            page.set_default_timeout(60000)

            try:
                # Try different wait strategies
                print(f"Loading page...")
                logger.info(f"Attempting to load page: {url}")
                try:
                    logger.debug("Trying domcontentloaded wait strategy")
                    await page.goto(url, wait_until='domcontentloaded', timeout=60000)
                    logger.info("✓ DOM loaded successfully")
                    print(f"✓ DOM loaded")
                except Exception as e:
                    logger.warning(f"domcontentloaded failed: {str(e)}")
                    print(f"Warning: {str(e)}")
                    print(f"Trying alternative loading strategy...")
                    logger.debug("Trying 'load' wait strategy")
                    await page.goto(url, wait_until='load', timeout=60000)
                    logger.info("✓ Page loaded with 'load' strategy")

                # Wait for images to load
                logger.debug("Waiting 3000ms for images to load")
                await page.wait_for_timeout(3000)
                logger.debug("Initial wait complete")

                # Better lazy loading
                logger.info("Triggering lazy loading")
                await self.trigger_lazy_loading(page)

                # Reveal hidden content
                logger.info("Revealing hidden images")
                await self.reveal_hidden_images(page)

                # Wait for new images to load
                logger.debug("Waiting 2000ms for newly revealed images")
                await page.wait_for_timeout(2000)
                logger.debug("Final wait complete")

                # Find all images
                logger.info("Locating all <img> elements on page")
                images = await page.locator('img').all()
                logger.info(f"✓ Found {len(images)} <img> elements")
                print(f"\n✓ Found {len(images)} <img> elements")

                if len(images) == 0:
                    logger.warning("No images found on page")
                    print(f"⚠ No images found. Page may use background images or lazy loading.")

                # Process each image
                skipped_invisible = 0
                skipped_no_src = 0
                skipped_data_uri = 0
                skipped_no_dimensions = 0
                data_uri_count = 0

                logger.info(f"Starting to process {len(images)} images")
                for idx, img in enumerate(images):
                    logger.debug(f"\n{'=' * 40}")
                    logger.debug(f"Processing image {idx + 1}/{len(images)}")
                    try:
                        # === IMPROVED VISIBILITY CHECK ===
                        if not self.include_invisible:
                            logger.debug("Checking image visibility (include_invisible=False)")
                            try:
                                is_visible = await self.is_actually_visible(img, page)
                                if not is_visible:
                                    logger.debug(f"Image {idx + 1} is not visible, skipping")
                                    skipped_invisible += 1
                                    continue
                                logger.debug(f"Image {idx + 1} is visible")
                            except Exception as visibility_error:
                                logger.warning(f"Visibility check failed for image {idx + 1}: {str(visibility_error)}")
                                skipped_invisible += 1
                                continue

                        # Get image properties
                        logger.debug("Getting image src attribute")
                        src = await img.get_attribute('src')
                        if not src:
                            logger.debug("No src attribute, checking data-src and data-lazy-src")
                            # Try data-src for lazy loaded images
                            src = await img.get_attribute('data-src') or await img.get_attribute('data-lazy-src')

                        if not src:
                            logger.debug(f"Image {idx + 1} has no src, skipping")
                            skipped_no_src += 1
                            continue

                        logger.debug(f"Image src: {src[:50]}...")

                        # Handle data URIs
                        if src.startswith('data:'):
                            if not self.include_data_uris:
                                logger.debug(f"Image {idx + 1} is a data URI - SKIPPING per user request")
                                skipped_data_uri += 1
                                continue
                            else:
                                logger.debug(f"Image {idx + 1} is a data URI - Including")
                                data_uri_count += 1

                        # Make absolute URL
                        logger.debug("Converting to absolute URL")
                        absolute_src = urljoin(url, src)
                        logger.debug(f"Absolute URL: {absolute_src[:50]}...")

                        # Get image info
                        logger.debug("Getting alt text and title")
                        alt_text = await img.get_attribute('alt') or ''
                        title = await img.get_attribute('title') or ''

                        # TODO: Add classification logic and saving here
                        logger.debug(f"Image {idx + 1} processed: src={absolute_src[:50]}, alt={alt_text[:30]}")

                    except Exception as e:
                        logger.warning(f"Error processing image {idx + 1}: {str(e)}")
                        continue

                # Print skip statistics
                logger.info("Image processing complete, printing summary")
                print(f"\n{'=' * 60}")
                print(f"IMAGE PROCESSING SUMMARY:")
                print(f"  Total found: {len(images)}")
                print(f"  Successfully captured: {len([img for img in self.images_data if img.url == url])}")
                if self.include_data_uris:
                    print(f"    - Data URIs: {data_uri_count}")
                    print(
                        f"    - Regular images: {len([img for img in self.images_data if img.url == url]) - data_uri_count}")
                print(f"  Skipped - Invisible: {skipped_invisible}")
                print(f"  Skipped - No dimensions: {skipped_no_dimensions}")
                print(f"  Skipped - No src: {skipped_no_src}")
                if not self.include_data_uris:
                    print(f"  Skipped - Data URI: {skipped_data_uri}")
                print(f"{'=' * 60}")

                logger.info(
                    f"Summary - Total: {len(images)}, Captured: {len([img for img in self.images_data if img.url == url])}, Skipped: {skipped_invisible + skipped_no_src + skipped_data_uri}")

                # Find links for crawling (optional)
                if current_depth < max_depth:
                    logger.info(f"Current depth {current_depth} < max depth {max_depth}, finding links")
                    print(f"\n{'=' * 60}")
                    print(f"Finding links for deeper crawling...")

                    logger.debug("Locating all <a> elements")
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
                                absolute_url = urljoin(url, href)
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
                                        await self.crawl_page(absolute_url, max_depth, current_depth + 1)

                                        logger.debug("Relaunching browser after recursive crawl")
                                        browser = await p.chromium.launch(headless=True)
                                        context = await browser.new_context(
                                            viewport={'width': 1920, 'height': 1080},
                                            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                                        )
                                        page = await context.new_page()
                                        page.set_default_timeout(60000)
                                        await page.goto(url, wait_until='domcontentloaded')
                                    else:
                                        logger.debug(f"Link already visited: {absolute_url}")
                                else:
                                    logger.debug(f"Link is different domain, skipping: {absolute_url}")
                        except Exception as e:
                            logger.warning(f"Error processing link {link_idx + 1}: {str(e)}")
                            continue

                    logger.info(f"Found {link_count} new links to crawl")
                    print(f"Found {link_count} new links to crawl")

            except Exception as e:
                logger.error(f"Error crawling {url}: {str(e)}")
                print(f"\n✗ Error crawling {url}: {str(e)}")
                import traceback
                traceback.print_exc()

            finally:
                logger.debug("Closing browser context")
                await context.close()
                logger.debug("Closing browser")
                await browser.close()
                logger.info(f"Finished crawling {url}")


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


if __name__ == "__main__":
    logger.info("Starting main program")
    asyncio.run(main())
    logger.info("Program finished")