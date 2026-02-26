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
            context_info = {
                'width': 0, 'height': 0,
                'inLink': False, 'inButton': False, 'hasClickHandler': False
            }

        # ── BUTTONS (must come before logo/icon to catch button-wrapped logos/icons) ──
        # Note: is_button_image no longer returns True for plain <a> links
        is_button = await self.is_button_image(img_element)
        if is_button or context_info.get('inButton'):
            classification.is_functional = True
            classification.is_button = True
            classification.classification = 'functional'
            classification.sub_type = 'buttons'
            logger.info("Classified as functional button image")
            return classification.model_dump()

        # ── LOGOS ──
        if await self.is_logo(img_element, src, alt_text):
            classification.is_logo = True
            classification.is_text_image = True
            if context_info.get('inLink'):
                # Logo inside a link → functional
                classification.is_functional = True
                classification.classification = 'functional'
                classification.sub_type = 'logos'
                logger.info("Classified as functional logo (in link)")
            else:
                # Standalone logo → informative
                classification.classification = 'informative'
                classification.sub_type = 'logos'  # consistent plural
                logger.info("Classified as informative logo (not in link)")
            return classification.model_dump()

        # ── CHARTS / GRAPHS ──
        if await self.is_chart(img_element, src, alt_text):
            classification.is_complex = True
            classification.classification = 'complex'
            classification.sub_type = 'charts'
            logger.info("Classified as complex chart/graph")
            return classification.model_dump()

        # ── ICONS ──
        if await self.is_icon(img_element, src, alt_text):
            classification.is_icon = True
            # Clickable icon → functional (use context_info, NOT is_button_image)
            if context_info.get('inLink') or context_info.get('hasClickHandler'):
                classification.is_functional = True
                classification.classification = 'functional'
                classification.sub_type = 'icons'
                logger.info("Classified as functional icon (clickable)")
            elif alt_text.strip():
                # Non-clickable icon with alt → informative
                classification.classification = 'informative'
                classification.sub_type = 'general_informative'
                logger.info("Classified as informative icon (has alt text)")
            else:
                # Non-clickable icon without alt → decorative
                classification.is_decorative = True
                classification.is_functional = False
                classification.classification = 'decorative'
                classification.sub_type = 'decorative'
                logger.info("Classified as decorative icon (no alt text)")
            return classification.model_dump()

        # ── ANY OTHER CLICKABLE IMAGE → FUNCTIONAL ──
        if context_info.get('inLink') or context_info.get('hasClickHandler'):
            classification.is_functional = True
            classification.classification = 'functional'
            classification.sub_type = 'images'
            logger.info("Classified as functional image (clickable)")
            return classification.model_dump()

        # ── DECORATIVE ──
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

        # ── INFORMATIVE ──
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
                classification.sub_type = 'supplementary'
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
        """Detect if element is an icon — <img> icon, <i> font icon, or SVG icon"""
        logger.debug(f"Checking if element is icon: src={src[:50]}, alt={alt_text[:50]}")

        icon_patterns = ['icon', 'ico', 'symbol', 'glyph', 'sprite']
        FONT_ICON_PATTERNS = [
            'fa ', 'fa-', 'fas ', 'fas-', 'far ', 'far-', 'fab ', 'fab-',  # FontAwesome
            'material-icons', 'material-symbols',  # Material
            'bi bi-', 'bi-',  # Bootstrap Icons
            'glyphicon',  # Bootstrap 3
            'feather', 'lucide',  # Feather / Lucide
            'icon-', '-icon',  # Generic
        ]

        try:
            tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
            class_name = await element.get_attribute('class') or ''
            class_lower = class_name.lower()

            # ── FONT-BASED ICONS: <i> or <span> directly queried from extraction loop ──
            # These ARE icons by definition if they have font icon classes — return immediately
            if tag_name in ('i', 'span'):
                if any(p in class_lower for p in FONT_ICON_PATTERNS):
                    logger.info(f"Font icon confirmed (tag={tag_name}, class={class_name})")
                    return True
                # <i> without font icon class is still likely an icon element
                if tag_name == 'i':
                    logger.info(f"<i> element treated as icon (no matching class but is <i> tag)")
                    return True
                return False  # <span> without icon class → not an icon

            # ── SVG ICON ──
            if tag_name == 'svg':
                size = await element.evaluate('''el => {
                    const r = el.getBoundingClientRect();
                    return { width: r.width, height: r.height };
                }''')
                if size['width'] <= 64 and size['height'] <= 64:
                    logger.info(f"SVG icon detected by size: {size}")
                    return True
                if any(p in class_lower for p in icon_patterns):
                    logger.info(f"SVG icon detected by class: {class_name}")
                    return True
                return False  # Large SVG without icon class → not an icon

            # ── IMAGE-BASED ICONS (<img>) — only reaches here for <img> tags ──
            src_lower = src.lower()
            if any(p in src_lower for p in icon_patterns):
                logger.info(f"Icon detected in src: {src[:50]}")
                return True

            if any(p in class_lower for p in icon_patterns):
                logger.info(f"Icon detected in class: {class_name}")
                return True

            size = await element.evaluate('''el => {
                const r = el.getBoundingClientRect();
                return { width: r.width, height: r.height };
            }''')
            # Accept icons up to 96×96 px (covers 24/32/48/64/96 common sizes)
            # Widen aspect ratio to 0.5–2.0 to catch non-square icons
            # (hamburger menus, flag icons, social share buttons, etc.)
            if size['width'] <= 96 and size['height'] <= 96:
                if size['width'] > 0 and size['height'] > 0:
                    aspect_ratio = size['width'] / size['height']
                    if 0.5 <= aspect_ratio <= 2.0:
                        logger.info(f"Icon detected by size: {size}")
                        return True

        except Exception as e:
            logger.debug(f"Error in is_icon: {e}")

        logger.debug("Element is not an icon")
        return False

    async def is_chart(self, element, src: str, alt_text: str, page=None) -> bool:
        """
        Detect if element is a chart, diagram, or other complex visual.

        Signal priority (highest → lowest):
          1. Chart JS library detected on page
          2. SVG internal structure (many shapes + labels)
          3. Canvas element (large)
          4. Keyword match in alt / src / class / figcaption
          5. Parent context (<figure>, <canvas>, wrapper divs)
          6. Aspect ratio heuristic (wide landscape = chart, tall portrait = infographic)

        Returns True only when cumulative score >= threshold (avoids false positives).
        """

        logger.debug(f"is_chart check: src={src[:60]}, alt={alt_text[:60]}")

        score = 0  # Accumulate evidence; threshold = 3

        # ─────────────────────────────────────────────────────────────
        # 1. CHART JS LIBRARY DETECTION (page-level, strongest signal)
        #    If the page uses a charting library, nearby large images
        #    are very likely chart screenshots or exported chart images.
        # ─────────────────────────────────────────────────────────────
        CHART_LIBS = [
            "chart.js", "chartjs", "d3.js", "d3.min", "highcharts",
            "plotly", "echarts", "apexcharts", "recharts", "vega",
            "amcharts", "fusioncharts", "canvasjs", "chartist",
            "nivo", "victory", "c3.js", "c3.min"
        ]

        if page:
            try:
                page_scripts = await page.evaluate('''() => {
                    return Array.from(document.querySelectorAll("script[src]"))
                        .map(s => s.src.toLowerCase());
                }''')
                if any(lib in script for lib in CHART_LIBS for script in page_scripts):
                    logger.info("Chart JS library detected on page (+2)")
                    score += 2
            except Exception as e:
                logger.debug(f"Script scan error: {e}")

        # ─────────────────────────────────────────────────────────────
        # 2. KEYWORD DETECTION — alt text, src/filename, class, title
        # ─────────────────────────────────────────────────────────────
        CHART_KEYWORDS = [
            "chart", "graph", "plot", "bar-chart", "line-chart", "pie-chart",
            "scatter", "histogram", "heatmap", "treemap", "funnel", "waterfall",
            "trend", "analytics", "metric", "kpi", "dashboard", "infographic",
            "diagram", "flowchart", "flow-chart", "architecture", "schema",
            "network", "topology", "mindmap", "gantt", "timeline",
            "statistics", "report", "figure", "data-viz", "visualization"
        ]

        src_lower = (src or "").lower()
        alt_lower = (alt_text or "").lower()

        if any(kw in alt_lower for kw in CHART_KEYWORDS):
            logger.info(f"Chart keyword in alt text (+3)")
            score += 3  # alt text is intentional — highest weight

        if any(kw in src_lower for kw in CHART_KEYWORDS):
            logger.info(f"Chart keyword in src/filename (+2)")
            score += 2

        try:
            class_name = (await element.get_attribute("class") or "").lower()
            title_attr = (await element.get_attribute("title") or "").lower()

            if any(kw in class_name for kw in CHART_KEYWORDS):
                logger.info(f"Chart keyword in class name (+2)")
                score += 2

            if any(kw in title_attr for kw in CHART_KEYWORDS):
                logger.info(f"Chart keyword in title attr (+1)")
                score += 1
        except Exception as e:
            logger.debug(f"Attribute read error: {e}")

        # ─────────────────────────────────────────────────────────────
        # 3. PARENT CONTEXT — <figure>, <canvas>, wrapper div classes,
        #    and EXTENDED CAPTION DETECTION (figcaption + custom divs)
        #
        #    Real-world sites rarely use semantic <figcaption>.
        #    Common patterns seen in the wild:
        #      • <div class="g-Image__caption">Figure 1. Diagram of...</div>  ← Kao example
        #      • <p class="caption">Chart showing...</p>
        #      • <span class="image-caption">...</span>
        #      • <p style="text-align:center">Figure 1. ...</p> inside wrapper
        # ─────────────────────────────────────────────────────────────
        try:
            parent_info = await element.evaluate('''el => {
                const parent = el.parentElement;
                const grandparent = parent ? parent.parentElement : null;
                // Walk up further to catch deeply nested caption siblings
                const great = grandparent ? grandparent.parentElement : null;

                function getCaptionText(node) {
                    if (!node) return "";

                    // 1. Semantic <figcaption>
                    const fig = node.querySelector("figcaption");
                    if (fig) return fig.textContent.trim().toLowerCase();

                    // 2. Custom caption divs/spans/p by class or id keyword
                    //    Matches: g-Image__caption, l-Image__caption, image-caption,
                    //             caption-text, figure-caption, img-caption, etc.
                    const captionEl = node.querySelector(
                        '[class*="caption" i], [class*="Caption"], [id*="caption" i], ' +
                        '[class*="figure" i], [class*="Figure"], ' +
                        '[class*="img-desc" i], [class*="image-desc" i], ' +
                        '[class*="photo-desc" i], [class*="media-desc" i]'
                    );
                    if (captionEl) return captionEl.textContent.trim().toLowerCase();

                    // 3. Centered <p> siblings that look like captions
                    //    e.g. <p style="text-align: center;">Figure 1. Diagram of...</p>
                    const centeredP = node.querySelector('p[style*="text-align: center"], p[style*="text-align:center"]');
                    if (centeredP) {
                        const text = centeredP.textContent.trim().toLowerCase();
                        // Only treat as caption if it starts with figure/fig/table/chart/diagram
                        if (/^(figure|fig\.|table|chart|diagram|image|photo|graph|plot)/.test(text)) {
                            return text;
                        }
                    }

                    return "";
                }

                function collectInfo(node) {
                    if (!node) return null;
                    const captionText = getCaptionText(node);
                    return {
                        tag: node.tagName.toLowerCase(),
                        className: (node.className || "").toLowerCase(),
                        id: (node.id || "").toLowerCase(),
                        hasCaption: captionText.length > 0,
                        captionText: captionText
                    };
                }

                return {
                    parent: collectInfo(parent),
                    grandparent: collectInfo(grandparent),
                    great: collectInfo(great)
                };
            }''')

            CHART_PARENT_CLASSES = [
                "chart", "graph", "diagram", "viz", "visualization",
                "plot", "infographic", "canvas-wrap", "chart-container",
                "dashboard", "analytics", "figure", "image__caption",
                "img-caption", "media-caption", "caption"
            ]

            for ancestor in [parent_info.get("parent"), parent_info.get("grandparent"), parent_info.get("great")]:
                if not ancestor:
                    continue

                # <figure> is a strong semantic signal
                if ancestor["tag"] == "figure":
                    logger.info("Parent is <figure> (+2)")
                    score += 2

                # <canvas> sibling or wrapper
                if ancestor["tag"] == "canvas":
                    logger.info("Parent is <canvas> (+2)")
                    score += 2

                # Class / ID match on wrapper div
                combined = ancestor["className"] + " " + ancestor["id"]
                if any(kw in combined for kw in CHART_PARENT_CLASSES):
                    logger.info(f"Chart keyword in parent class/id: {combined[:60]} (+2)")
                    score += 2

                # Caption text (figcaption OR custom div) with chart keywords
                if ancestor["hasCaption"] and any(kw in ancestor["captionText"] for kw in CHART_KEYWORDS):
                    logger.info(f"Caption contains chart keyword: '{ancestor['captionText'][:80]}' (+3)")
                    score += 3  # Bumped from +2 — caption text is very reliable

        except Exception as e:
            logger.debug(f"Parent context error: {e}")

        # ─────────────────────────────────────────────────────────────
        # 4. SVG STRUCTURAL ANALYSIS
        #    Many <rect>/<path> elements + <text> labels → rendered chart
        # ─────────────────────────────────────────────────────────────
        try:
            tag_name = await element.evaluate("el => el.tagName.toLowerCase()")

            if tag_name == "svg":
                structure = await element.evaluate('''svg => ({
                    rects:   svg.querySelectorAll("rect").length,
                    paths:   svg.querySelectorAll("path").length,
                    lines:   svg.querySelectorAll("line").length,
                    circles: svg.querySelectorAll("circle").length,
                    texts:   svg.querySelectorAll("text").length,
                    groups:  svg.querySelectorAll("g").length,
                    hasAxis: svg.querySelector(".axis, .tick, [class*=axis], [class*=tick]") !== null,
                    hasLegend: svg.querySelector(".legend, [class*=legend]") !== null
                })''')

                logger.debug(f"SVG structure: {structure}")

                # Axis or legend is a near-certain chart signal
                if structure["hasAxis"] or structure["hasLegend"]:
                    logger.info("SVG has axis/legend (+4)")
                    score += 4

                # Many shapes + text labels = chart
                shape_count = structure["rects"] + structure["paths"] + structure["circles"]
                if structure["texts"] >= 3 and shape_count > 5:
                    logger.info(f"SVG: shapes={shape_count}, texts={structure['texts']} (+3)")
                    score += 3

                # Lots of groups can indicate layered chart structure
                if structure["groups"] > 10 and structure["texts"] >= 2:
                    logger.info(f"SVG: complex group structure (+1)")
                    score += 1

        except Exception as e:
            logger.debug(f"SVG analysis error: {e}")

        # ─────────────────────────────────────────────────────────────
        # 5. CANVAS ELEMENT (large rendered canvas = chart/viz)
        # ─────────────────────────────────────────────────────────────
        try:
            tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
            if tag_name == "canvas":
                size = await element.evaluate('''el => {
                    const r = el.getBoundingClientRect();
                    return {width: r.width, height: r.height};
                }''')
                if size["width"] > 200 and size["height"] > 150:
                    logger.info(f"Large <canvas> element: {size} (+3)")
                    score += 3
        except Exception as e:
            logger.debug(f"Canvas check error: {e}")

        # ─────────────────────────────────────────────────────────────
        # 6. ASPECT RATIO + SIZE HEURISTIC
        #    Charts: wide landscape (ratio > 1.3, width > 300)
        #    Infographics: tall portrait (ratio < 0.6, height > 400)
        #    Icons/thumbnails: small — skip
        # ─────────────────────────────────────────────────────────────
        try:
            size = await element.evaluate('''el => {
                const r = el.getBoundingClientRect();
                return {width: r.width, height: r.height};
            }''')

            w, h = size["width"], size["height"]

            if w > 0 and h > 0:
                ratio = w / h
                is_large = w > 300 or h > 300

                if is_large and (ratio > 1.3 or ratio < 0.6):
                    logger.info(f"Aspect ratio matches chart/infographic: {ratio:.2f}, size={w}x{h} (+1)")
                    score += 1

        except Exception as e:
            logger.debug(f"Size heuristic error: {e}")

        # ─────────────────────────────────────────────────────────────
        # 7. FILE FORMAT SIGNAL
        #    SVG file extension in src → likely diagram/icon (soft signal)
        # ─────────────────────────────────────────────────────────────
        if src_lower.endswith(".svg"):
            logger.info("SVG file format detected (+1)")
            score += 1

        # ─────────────────────────────────────────────────────────────
        # DECISION
        # ─────────────────────────────────────────────────────────────
        THRESHOLD = 3
        is_chart_result = score >= THRESHOLD

        logger.info(f"is_chart score={score}/{THRESHOLD} → {'CHART ✓' if is_chart_result else 'not chart'}")
        return is_chart_result

    async def is_button_image(self, element) -> bool:
        """Detect if element is a button or image inside a button"""
        logger.debug("Checking if element is a button")
        try:
            result = await element.evaluate('''el => {
                const tag = el.tagName.toLowerCase();
                const inputType = (el.getAttribute('type') || '').toLowerCase();

                // ── Element IS a button natively ──
                if (tag === 'button') return { isButton: true, reason: 'is <button> element' };

                if (tag === 'input' && ['button', 'submit', 'reset'].includes(inputType))
                    return { isButton: true, reason: 'is input[type=' + inputType + ']' };

                if (el.getAttribute('role') === 'button')
                    return { isButton: true, reason: 'has role=button' };

                // ── Element has an inline click handler on itself ──
                if (el.onclick !== null || el.getAttribute('onclick') !== null)
                    return { isButton: true, reason: 'has click handler' };

                // ── Element is INSIDE a button (for <img> elements) ──
                if (tag === 'img') {
                    // First: check for ANY button ancestor regardless of <a> in between
                    if (el.closest('button') !== null)
                        return { isButton: true, reason: 'img inside <button>' };

                    if (el.closest('[role="button"]') !== null)
                        return { isButton: true, reason: 'img inside role=button element' };

                    // Walk ancestors — max 4 levels
                    // Only bail on <a> if there is definitely NO button above it
                    let ancestor = el.parentElement;
                    let depth = 0;
                    let foundLink = false;
                    while (ancestor && depth < 4) {
                        const aTag = ancestor.tagName.toLowerCase();
                        if (aTag === 'body' || aTag === 'html') break;

                        const role = ancestor.getAttribute('role');
                        const type = (ancestor.getAttribute('type') || '').toLowerCase();
                        const className = (ancestor.className || '').toLowerCase();

                        if (role === 'button') return { isButton: true, reason: 'ancestor role=button' };
                        if (type === 'button') return { isButton: true, reason: 'ancestor type=button' };
                        // Only match btn as a whole word to avoid false positives (e.g. "submit-btn", "navbar")  
                        if (/\bbtn\b/.test(className)) return { isButton: true, reason: 'ancestor btn class' };

                        // Track <a> but don't bail immediately — button may appear above
                        if (aTag === 'a') foundLink = true;

                        ancestor = ancestor.parentElement;
                        depth++;
                    }

                    // If only a link was found and no button, it's not a button image
                    if (foundLink) return { isButton: false, reason: null };
                }

                return { isButton: false, reason: null };
            }''')

            logger.debug(f"Button detection result: {result}")
            if result.get('isButton'):
                logger.info(f"Button detected: {result.get('reason')}")
                return True

        except Exception as e:
            logger.debug(f"Error checking button: {str(e)}")

        logger.debug("Element is not a button")
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
