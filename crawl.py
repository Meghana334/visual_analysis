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


class ImageClassification(BaseModel):
    """Classification details for an image"""
    type: str = 'informative'
    sub_type: Optional[str] = None
    is_text_image: bool = False
    is_functional: bool = False
    is_decorative: bool = False
    is_complex: bool = False
    is_logo: bool = False
    is_icon: bool = False
    is_button: bool = False
    file_format: Optional[str] = None


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

        self.file_handler = None # basic handler not needed as we handle paths locally

    def _get_extension_from_type(self, asset_type: str) -> str:
        """Get file extension from asset type."""
        extensions = {
            'css': 'css',
            'js': 'js',
            'images': 'jpg',
            'fonts': 'woff2',
            'videos': 'mp4',
            'audios': 'mp3',
        }
        return extensions.get(asset_type, 'bin')

    async def _download_file(self, session: aiohttp.ClientSession, url: str, dest_path: str) -> bool:
        """Download a single file to destination path."""
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    logger.warning(f"Failed to download {url}: status {response.status}")
                    return False
                content = await response.read()
                with open(dest_path, 'wb') as f:
                    f.write(content)
                return True
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return False



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

    async def is_logo(self, src: str, alt_text: str, element) -> bool:
        """Detect if image is a logo"""
        logger.debug(f"Checking if image is logo: src={src[:50]}..., alt={alt_text[:50]}...")

        logo_patterns = ['logo', 'brand', 'header-img', 'site-logo', 'company-logo']
        logger.debug(f"Logo patterns to check: {logo_patterns}")

        src_lower = src.lower()
        alt_lower = alt_text.lower()
        logger.debug(f"Lowercased src and alt for comparison")

        # Check src and alt for logo keywords
        if any(pattern in src_lower for pattern in logo_patterns):
            logger.info(f"Logo detected in src: {src[:50]}...")
            return True
        if any(pattern in alt_lower for pattern in logo_patterns):
            logger.info(f"Logo detected in alt text: {alt_text[:50]}...")
            return True

        # Check class names
        try:
            logger.debug("Checking element class names for logo patterns")
            class_name = await element.get_attribute('class') or ''
            logger.debug(f"Element class: {class_name}")
            if any(pattern in class_name.lower() for pattern in logo_patterns):
                logger.info(f"Logo detected in class name: {class_name}")
                return True
        except Exception as e:
            logger.debug(f"Error checking class name: {str(e)}")
            pass

        # Check if in header/nav
        try:
            logger.debug("Checking if element is in header/nav")
            in_header = await element.evaluate('el => el.closest("header") !== null || el.closest("nav") !== null')
            logger.debug(f"Element in header/nav: {in_header}")

            if in_header and (alt_lower or src_lower):
                logger.debug("Element is in header/nav, checking size")
                # Small images in header are likely logos
                size = await element.evaluate('''el => {
                    const rect = el.getBoundingClientRect();
                    return {width: rect.width, height: rect.height};
                }''')
                logger.debug(f"Element size: {size}")
                if size['height'] < 100 and size['width'] < 300:
                    logger.info(f"Logo detected by size in header: {size}")
                    return True
        except Exception as e:
            logger.debug(f"Error checking header/nav: {str(e)}")
            pass

        logger.debug("Image is not a logo")
        return False

    async def is_icon(self, element, src: str, alt_text: str) -> bool:
        """Detect if image is an icon"""
        logger.debug(f"Checking if image is icon: src={src[:50]}..., alt={alt_text[:50]}...")

        icon_patterns = ['icon', 'ico', 'symbol', 'glyph', 'sprite']
        logger.debug(f"Icon patterns to check: {icon_patterns}")

        src_lower = src.lower()
        alt_lower = alt_text.lower()

        # Check src for icon keywords
        if any(pattern in src_lower for pattern in icon_patterns):
            logger.info(f"Icon detected in src: {src[:50]}...")
            return True

        # Check class names
        try:
            logger.debug("Checking element class names for icon patterns")
            class_name = await element.get_attribute('class') or ''
            logger.debug(f"Element class: {class_name}")
            if any(pattern in class_name.lower() for pattern in icon_patterns):
                logger.info(f"Icon detected in class name: {class_name}")
                return True
        except Exception as e:
            logger.debug(f"Error checking class name: {str(e)}")
            pass

        # Small square images are likely icons
        try:
            logger.debug("Checking element size for icon detection")
            size = await element.evaluate('''el => {
                const rect = el.getBoundingClientRect();
                return {width: rect.width, height: rect.height};
            }''')
            logger.debug(f"Element size: {size}")

            if size['width'] <= 64 and size['height'] <= 64:
                aspect_ratio = size['width'] / max(size['height'], 1)
                logger.debug(f"Small element aspect ratio: {aspect_ratio}")
                if 0.8 <= aspect_ratio <= 1.2:  # Nearly square
                    logger.info(f"Icon detected by size: {size}")
                    return True
        except Exception as e:
            logger.debug(f"Error checking size: {str(e)}")
            pass

        logger.debug("Image is not an icon")
        return False

    async def is_button_image(self, element) -> bool:
        """Detect if image is inside a button"""
        logger.debug("Checking if image is inside a button")
        try:
            # Check if inside button element
            logger.debug("Checking if element is inside <button> tag")
            in_button = await element.evaluate('''el => {
                const button = el.closest("button");
                return button !== null;
            }''')
            logger.debug(f"Element in button tag: {in_button}")

            if in_button:
                logger.info("Button image detected (inside <button> tag)")
                return True

            # Check if parent has button-like attributes
            logger.debug("Checking parent element for button attributes")
            parent_info = await element.evaluate('''el => {
                const parent = el.parentElement;
                if (!parent) return null;
                const role = parent.getAttribute('role');
                const type = parent.getAttribute('type');
                const className = parent.className || '';
                return {role, type, className};
            }''')
            logger.debug(f"Parent info: {parent_info}")

            if parent_info:
                if parent_info.get('role') == 'button':
                    logger.info("Button image detected (parent has role='button')")
                    return True
                if parent_info.get('type') == 'button':
                    logger.info("Button image detected (parent has type='button')")
                    return True
                if 'btn' in parent_info.get('className', '').lower():
                    logger.info("Button image detected (parent has 'btn' class)")
                    return True
        except Exception as e:
            logger.debug(f"Error checking button image: {str(e)}")
            pass

        logger.debug("Image is not a button image")
        return False

    async def classify_image(self, img_element, page) -> ImageClassification:
        """Classify image based on its attributes and context"""
        logger.info("Starting image classification")

        # Get basic attributes
        logger.debug("Getting image basic attributes")
        alt_text = await img_element.get_attribute('alt') or ''
        logger.debug(f"Alt text: {alt_text[:50]}...")

        src = await img_element.get_attribute('src') or ''
        logger.debug(f"Src: {src[:50]}...")

        role = await img_element.get_attribute('role') or ''
        logger.debug(f"Role: {role}")

        aria_hidden = await img_element.get_attribute('aria-hidden') or ''
        logger.debug(f"Aria-hidden: {aria_hidden}")

        aria_label = await img_element.get_attribute('aria-label') or ''
        logger.debug(f"Aria-label: {aria_label[:50]}...")

        title = await img_element.get_attribute('title') or ''
        logger.debug(f"Title: {title[:50]}...")

        classification = ImageClassification()
        logger.debug("Created ImageClassification instance")

        # Get context info
        try:
            logger.debug("Getting image context information")
            context_info = await img_element.evaluate('''el => {
                const styles = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                const parent = el.parentElement;

                return {
                    width: rect.width,
                    height: rect.height,
                    position: styles.position,
                    parentTag: parent ? parent.tagName : null,
                    inLink: el.closest("a") !== null,
                    inButton: el.closest("button") !== null,
                    linkHref: el.closest("a") ? el.closest("a").href : null,
                    linkText: el.closest("a") ? el.closest("a").textContent.trim() : null,
                    hasClickHandler: el.onclick !== null || el.parentElement?.onclick !== null
                };
            }''')
            logger.debug(f"Context info: {context_info}")
        except Exception as e:
            logger.warning(f"Could not get context info: {str(e)}")
            print(f"    Warning: Could not get context info: {str(e)}")
            context_info = {'width': 0, 'height': 0, 'inLink': False, 'inButton': False}

        # === FUNCTIONAL IMAGE CLASSIFICATION ===
        logger.debug("Starting functional image checks")

        # Example 5: Image used in a button
        logger.debug("Checking if image is a button image")
        if await self.is_button_image(img_element) or context_info.get('inButton'):
            logger.info("Classified as functional button image")
            classification.is_functional = True
            classification.is_button = True
            classification.type = 'functional'
            classification.sub_type = 'buttons'
            return classification

        # Example 1 & 2: Logo images
        logger.debug("Checking if image is a logo")
        if await self.is_logo(src, alt_text, img_element):
            logger.info("Image is a logo")
            classification.is_logo = True
            classification.is_text_image = True

            if context_info.get('inLink'):
                logger.info("Classified as functional logo (in link)")
                classification.is_functional = True
                classification.type = 'functional'
                classification.sub_type = 'logos'
            else:
                logger.info("Classified as informative logo (not in link)")
                classification.type = 'informative'
                classification.sub_type = 'logo'
            return classification

        # Example 3 & 4: Icon images
        logger.debug("Checking if image is an icon")
        if await self.is_icon(img_element, src, alt_text):
            logger.info("Image is an icon")
            classification.is_icon = True

            if context_info.get('inLink') or context_info.get('hasClickHandler'):
                logger.info("Classified as functional icon (clickable)")
                classification.is_functional = True
                classification.type = 'functional'
                classification.sub_type = 'icons'
            else:
                if alt_text.strip():
                    logger.info("Classified as informative icon (has alt text)")
                    classification.type = 'informative'
                    classification.sub_type = 'informative_icon'
                else:
                    logger.info("Classified as decorative icon (no alt text)")
                    classification.is_decorative = True
                    classification.type = 'decorative'
                    classification.sub_type = 'decorative_icon'
            return classification

        # Any clickable image is functional
        logger.debug("Checking if image is clickable")
        if context_info.get('inLink') or context_info.get('hasClickHandler'):
            logger.info("Classified as functional image (clickable)")
            classification.is_functional = True
            classification.type = 'functional'
            classification.sub_type = 'images'
            return classification

        # === DECORATIVE IMAGE CLASSIFICATION ===
        logger.debug("Checking if image is decorative")
        if (alt_text == '' or
                role in ['presentation', 'none'] or
                aria_hidden == 'true' or
                'decorat' in src.lower() or
                'background' in src.lower() or
                'spacer' in src.lower()):
            logger.info("Classified as decorative image")
            classification.is_decorative = True
            classification.type = 'decorative'
            classification.sub_type = 'decorative'
            return classification

        # === INFORMATIVE IMAGE CLASSIFICATION ===
        logger.debug("Checking if image is informative")
        if alt_text.strip() and len(alt_text) < 100:
            logger.info("Classified as informative (has succinct alt text)")
            classification.type = 'informative'
            classification.sub_type = 'succinct_information'

        # Check for supplementary images
        try:
            logger.debug("Checking for nearby text (supplementary image detection)")
            has_nearby_text = await img_element.evaluate('''el => {
                const parent = el.parentElement;
                if (!parent) return false;
                const siblings = Array.from(parent.children);
                const textSiblings = siblings.filter(s => 
                    s !== el && s.textContent && s.textContent.trim().length > 20
                );
                return textSiblings.length > 0;
            }''')
            logger.debug(f"Has nearby text: {has_nearby_text}")

            if has_nearby_text and alt_text.strip():
                logger.info("Classified as supplementary informative image")
                classification.type = 'informative'
                classification.sub_type = 'supplementary'
        except Exception as e:
            logger.debug(f"Error checking nearby text: {str(e)}")
            pass

        # Default: If has alt text, it's informative
        if alt_text.strip() and classification.type == 'informative' and not classification.sub_type:
            logger.info("Classified as general informative image")
            classification.sub_type = 'general_informative'

        logger.info(f"Final classification: type={classification.type}, sub_type={classification.sub_type}")
        return classification

    def get_image_hash(self, src: str) -> str:
        """Generate unique hash for image URL"""
        logger.debug(f"Generating hash for: {src[:50]}...")
        hash_value = hashlib.md5(src.encode()).hexdigest()[:12]
        logger.debug(f"Generated hash: {hash_value}")
        return hash_value

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
                # Wait for new images to load
                logger.debug("Waiting 2000ms for newly revealed images")
                await page.wait_for_timeout(2000)
                logger.debug("Final wait complete")

                # === GLOBAL ASSET EXTRACTION ===
                # Removed per user request to use only Playwright and skip global assets


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
                            logger.debug(f"Image {idx + 1} is a data URI - SKIPPING per user request")
                            skipped_data_uri += 1
                            continue

                        # Make absolute URL
                        logger.debug("Converting to absolute URL")
                        absolute_src = urljoin(url, src)
                        logger.debug(f"Absolute URL: {absolute_src[:50]}...")

                        # Get image info
                        logger.debug("Getting alt text and title")
                        alt_text = await img.get_attribute('alt') or ''
                        title = await img.get_attribute('title') or ''

                        # Classify image
                        print(f"\n  [{idx + 1}/{len(images)}] Analyzing image...")
                        logger.info(f"Classifying image {idx + 1}")
                        classification = await self.classify_image(img, page)

                        # Generate unique filename
                        logger.debug("Generating filename")
                        img_hash = self.get_image_hash(absolute_src)
                        filename = f"img_{img_hash}.png"
                        logger.debug(f"Filename: {filename}")

                        # Determine directory based on classification
                        logger.debug("Determining output directory based on classification")
                        if classification.type == 'complex':
                            logger.debug("Classification is 'complex' - SKIPPING per user request")
                            continue

                        if classification.type == 'functional':
                            img_dir = f"{self.output_dir}/functional/{classification.sub_type}"
                        elif classification.is_text_image and not classification.is_functional:
                            img_dir = f"{self.output_dir}/text_images"
                        else:
                            img_dir = f"{self.output_dir}/{classification.type}"
                        logger.debug(f"Output directory: {img_dir}")

                        screenshot_path = f"{img_dir}/{filename}"
                        logger.debug(f"Full screenshot path: {screenshot_path}")

                        # === CHANGED: DOWNLOAD ORIGINAL FILE INSTEAD OF SCREENSHOT ===
                        try:
                            # Use aiohttp to download the original source
                            async with aiohttp.ClientSession() as session:
                                logger.debug(f"Downloading original image to {screenshot_path}")
                                
                                # Fix extension if needed - we want the original file extension
                                # The filename was generated as .png by default above, let's look at the src extension
                                original_ext = os.path.splitext(urlparse(absolute_src).path)[1]
                                if original_ext:
                                    # Update filename (and screenshot_path) to match original extension
                                    if not filename.endswith(original_ext):
                                        filename = f"img_{img_hash}{original_ext}"
                                        screenshot_path = f"{img_dir}/{filename}"
                                
                                success = await self._download_file(session, absolute_src, screenshot_path)
                                
                                if success:
                                    logger.info(f"✓ Image downloaded: {screenshot_path}")
                                    print(f"  ✓ Downloaded: {filename}")
                                else:
                                    logger.warning(f"Failed to download {absolute_src}, fallback/skipping")
                                    print(f"  ✗ Failed to download: {filename}")
                                    continue # Skip if download failed

                                # Print classification details
                                print(f"    Type: {classification.type}")
                                if classification.sub_type:
                                    print(f"    Sub-type: {classification.sub_type}")
                                if classification.is_logo:
                                    print(f"    Logo: Yes")
                                if classification.is_icon:
                                    print(f"    Icon: Yes")
                                if classification.is_button:
                                    print(f"    Button Image: Yes")
                                if alt_text:
                                    print(f"    Alt text: {alt_text[:60]}{'...' if len(alt_text) > 60 else ''}")

                        except Exception as e:
                            logger.error(f"Failed to download image {idx + 1}: {str(e)}")
                            print(f"  ✗ Failed to download {filename}: {str(e)}")
                            continue

                        # Store image data using Pydantic model
                        logger.debug("Creating ImageData object")
                        image_data = ImageData(
                            url=url,
                            src=absolute_src,
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

                        logger.debug("Adding image to images_data list")
                        self.images_data.append(image_data)
                        logger.info(f"✓ Successfully processed image {idx + 1}")

                    except Exception as e:
                        logger.error(f"Error processing image {idx}: {str(e)}")
                        print(f"  ✗ Error processing image {idx}: {str(e)}")
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


async def main():
    logger.info("=" * 60)
    logger.info("ASYNC IMAGE CRAWLER - STARTING")
    logger.info("=" * 60)

    # Configuration
    website_url = "https://www.kao.com/global/en/"
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

    # Save results even if crawl was interrupted
    logger.info("Saving results")
    crawler.save_results()
    logger.info("Results saved. Program complete.")


if __name__ == "__main__":
    logger.info("Starting main program")
    asyncio.run(main())
    logger.info("Program finished")