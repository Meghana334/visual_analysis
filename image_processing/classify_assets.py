import os
import hashlib
import aiohttp
import logging
from urllib.parse import urljoin
from pydantic import BaseModel

from crawl.models import ImageData

logger = logging.getLogger(__name__)


class ImageClassification(BaseModel):
    classification: str = "informative"
    sub_type: str | None = None
    is_text_image: bool = False
    is_functional: bool = False
    is_decorative: bool = False
    is_complex: bool = False
    is_logo: bool = False
    is_icon: bool = False
    is_button: bool = False
    file_format: str | None = None


class ClassifyAssets:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.images_data: list[ImageData] = []


    def get_image_hash(self, src: str) -> str:
        """Generate unique hash for image URL"""
        return hashlib.md5(src.encode()).hexdigest()[:12]

    async def classify_image(self, img_element) -> dict:
        """Classify image based on its attributes and context"""
        logger.info("Starting image classification")

        alt_text = await img_element.get_attribute('alt') or ''
        src = await img_element.get_attribute('src') or ''
        role = await img_element.get_attribute('role') or ''
        aria_hidden = await img_element.get_attribute('aria-hidden') or ''

        classification = ImageClassification()

        try:
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
            context_info = {'width': 0, 'height': 0, 'inLink': False, 'inButton': False, 'hasClickHandler': False}

        #### buttons
        if await self.is_button_image(img_element) or context_info.get('inButton'):
            classification.is_functional = True
            classification.is_button = True
            classification.classification = 'functional'
            classification.sub_type = 'buttons'
            logger.info("Classified as functional button image")
            return classification.model_dump()

        #### logos
        if await self.is_logo(img_element, src, alt_text):
            classification.is_logo = True
            classification.is_text_image = True
            if context_info.get('inLink'):
                classification.is_functional = True
                classification.classification = 'functional'
                classification.sub_type = 'logos'
                logger.info("Classified as functional logo (in link)")
            else:
                classification.classification = 'informative'
                classification.sub_type = 'logo'
                logger.info("Classified as informative logo (not in link)")
            return classification.model_dump()

        #### icon
        if await self.is_icon(img_element, src, alt_text):
            classification.is_icon = True
            if context_info.get('inLink') or context_info.get('hasClickHandler'):
                classification.is_functional = True
                classification.classification = 'functional'
                classification.sub_type = 'icons'
                logger.info("Classified as functional icon (clickable)")
            else:
                if alt_text.strip():
                    classification.classification = 'informative'
                    classification.sub_type = 'general_informative'  # → informative/
                    logger.info("Classified as informative icon (has alt text)")
                else:
                    classification.is_decorative = True
                    classification.is_functional = False
                    classification.classification = 'decorative'
                    classification.sub_type = 'decorative'  # → decorative/
                    logger.info("Classified as decorative icon (no alt text)")
            return classification.model_dump()

        #### Any clickable image is functional
        if context_info.get('inLink') or context_info.get('hasClickHandler'):
            classification.is_functional = True
            classification.classification = 'functional'
            classification.sub_type = 'images'
            logger.info("Classified as functional image (clickable)")
            return classification.model_dump()


        #### decorative
        if (alt_text == '' or
                role in ['presentation', 'none'] or
                aria_hidden == 'true' or
                'decorat' in src.lower() or
                'background' in src.lower() or
                'spacer' in src.lower()):
            classification.is_decorative = True
            classification.is_functional = False
            classification.classification = 'decorative'
            classification.sub_type = 'decorative'
            logger.info("Classified as decorative image")
            return classification.model_dump()


        #### informative
        if alt_text.strip() and len(alt_text) < 100:
            classification.classification = 'informative'
            classification.sub_type = 'succinct_information'
            logger.info("Classified as informative (succinct alt text)")

        try:
            has_nearby_text = await img_element.evaluate('''el => {
                const parent = el.parentElement;
                if (!parent) return false;
                const siblings = Array.from(parent.children);
                const textSiblings = siblings.filter(s =>
                    s !== el && s.textContent && s.textContent.trim().length > 20
                );
                return textSiblings.length > 0;
            }''')
            if has_nearby_text and alt_text.strip():
                classification.classification = 'informative'
                classification.sub_type = 'supplementary'  # → informative/
                logger.info("Classified as supplementary informative image")
        except Exception as e:
            logger.debug(f"Error checking nearby text: {str(e)}")

        # Default informative fallback
        if alt_text.strip() and classification.classification == 'informative' and not classification.sub_type:
            classification.sub_type = 'general_informative'
            logger.info("Classified as general informative image")

        logger.info(
            f"Final classification: classification={classification.classification}, sub_type={classification.sub_type}")
        return classification.model_dump()

    async def is_logo(self, element, src: str, alt_text: str) -> bool:
        patterns = ["logo", "brand", "header-img", "site-logo", "company-logo"]
        src_lower = src.lower()
        alt_lower = alt_text.lower()
        if any(p in src_lower for p in patterns) or any(p in alt_lower for p in patterns):
            return True
        try:
            cls = await element.get_attribute("class") or ""
            if any(p in cls.lower() for p in patterns):
                return True
        except Exception:
            pass
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

    async def get_visual_container(self, img_element, page):
        return await img_element.evaluate_handle('''img => {
            function getVisibleText(element) {
                const text = element.innerText && element.innerText.trim();
                return text && text.length > 3 && text.length < 200;
            }

            let current = img.parentElement;
            const imgRect = img.getBoundingClientRect();

            // Skip overlay detection for small images (icons, social buttons)
            if (imgRect.width < 60 || imgRect.height < 60) {
                return img;
            }

            for (let i = 0; i < 3; i++) {
                if (!current || current.tagName === 'BODY' || current.tagName === 'HTML') break;

                const rect = current.getBoundingClientRect();
                const areaRatio = (rect.width * rect.height) / (imgRect.width * imgRect.height);

                // Container must be close in size to image
                if (areaRatio > 2) break;

                const children = Array.from(current.children);
                const hasOverlaySibling = children.some(child => {
                    if (child === img) return false;
                    const childStyle = window.getComputedStyle(child);
                    const childRect = child.getBoundingClientRect();

                    const intersect = !(childRect.right < imgRect.left ||
                                      childRect.left > imgRect.right ||
                                      childRect.bottom < imgRect.top ||
                                      childRect.top > imgRect.bottom);

                    const hasText = getVisibleText(child);
                    // Must be absolutely positioned AND overlapping AND meaningful text
                    return childStyle.position === 'absolute' && intersect && hasText;
                });

                if (hasOverlaySibling) return current;
                current = current.parentElement;
            }

            return img;
        }''')

    async def _download_file(self, session: aiohttp.ClientSession, url: str, path: str) -> bool:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "wb") as f:
                        f.write(await resp.read())
                    return True
                else:
                    logger.warning(f"Download failed {resp.status}: {url}")
                    return False
        except Exception as e:
            logger.error(f"Download error {url}: {e}")
            return False
