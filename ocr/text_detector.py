import os
import json
import shutil
import sys
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import cv2
from utils.color_picker import extract_color
from utils.contrast import contrast_analyzer
from utils.helper_modules.text_helper import TextDetectionResult, TextDetectionReport, DetailedDetection, _json_serializer
from ocr.easy_ocr import OCRReader
from config.logger import setup_logger
from utils.helper_modules.config_helper import load_config
config = load_config()

logger = setup_logger(name="KAC", tag="text_detector")
logger.info("Logger initialized")


class OCRPreprocessing:

    def __init__(self, source_directory: str, output_directory: Optional[str] = None):
        self.source_directory = source_directory

        if output_directory is None:
            self.base_output_dir = source_directory
        else:
            self.base_output_dir = output_directory

        ocr_config = config["ocr"]

        base_folder = config["ocr"]["base_folder"] or ""
        contrast_folder = config["ocr"]["contrast_folder"]
        category_config = ocr_config["categories"]

        self.text_detected_dir = os.path.join(
            self.base_output_dir,
            base_folder
        )

        self.contrast_dir = os.path.join(
            self.text_detected_dir,
            contrast_folder
        )
        logger.info(f"base output dir :{self.base_output_dir}")
        logger.info(f"text detected dir :{self.text_detected_dir}")

        self.categories = {
            key: os.path.join(self.text_detected_dir, value)
            for key, value in category_config.items()
        }
        # Create folders
        for path in self.categories.values():
            Path(path).mkdir(parents=True, exist_ok=True)

        Path(self.contrast_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Initializing EasyOCR Reader (loading models)...")
        self.reader = OCRReader(source_directory)
        logger.info("EasyOCR Reader initialized")

        self.results: List[TextDetectionResult] = []

    def _determine_category(self,original_path: str) -> str:
        """Heuristic to determine category based on source path."""
        path_str = str(original_path).lower()
        if "button" in path_str:
            return "button_text"
        elif "logo" in path_str:
            return "logo_text"
        elif "informative" in path_str or "informational" in path_str:
            return "informational_text"
        return "with_text"

    # def detect_text_in_image(self, image_path: str) -> TextDetectionResult:
    #     """Use EasyOCR to detect text and run contrast analysis"""
    #     logger.info(f"Processing: {image_path}")
    #
    #     filename = Path(image_path).name
    #     category = self._determine_category(image_path)
    #
    #     result = TextDetectionResult(
    #         filename=filename,
    #         original_path=image_path,
    #         category=category
    #     )
    #
    #     try:
    #         detections = self.reader.readtext(image_path)
    #
    #         if len(detections) > 0:
    #             result.has_text = True
    #             img = cv2.imread(image_path)
    #
    #             for bbox, text, conf in detections:
    #                 clean_bbox = [(int(p[0]), int(p[1])) for p in bbox]
    #
    #                 try:
    #                     contrast_info = contrast_analyzer.analyze_text_region(img, clean_bbox)
    #                 except Exception as e:
    #                     logger.warning(f"Contrast analysis failed: {e}")
    #                     contrast_info = None
    #
    #                 violations = []
    #
    #                 color_info = None
    #                 if 'utils.color_picker' in sys.modules:
    #                     try:
    #                         fg_color = extract_color.extract_text_color(img, clean_bbox)
    #                         bg_pixels = extract_color.extract_adjacent_text_pixels(img, clean_bbox)
    #                         bg_colors = extract_color.cluster_colors(bg_pixels, k=3)
    #
    #                         color_info = {
    #                             "foreground": fg_color,
    #                             "background_palette": bg_colors,
    #                             "contrast_checks": []
    #                         }
    #
    #                         fg_lum = fg_color['luminance']
    #                         for bg in bg_colors:
    #                             bg_lum = bg['luminance']
    #                             l1 = max(fg_lum, bg_lum)
    #                             l2 = min(fg_lum, bg_lum)
    #                             ratio = (l1 + 0.05) / (l2 + 0.05)
    #                             compliance = contrast_analyzer.check_wcag_compliance(ratio)
    #                             color_info["contrast_checks"].append({
    #                                 "bg_color": bg,
    #                                 "ratio": round(ratio, 2),
    #                                 "compliance": compliance
    #                             })
    #                             if not compliance['AA_normal']:
    #                                 violations.append(f"Fails AA Normal vs BG {bg['hex']}")
    #
    #                     except Exception as cp_err:
    #                         logger.warning(f"Color picker failed for region: {cp_err}")
    #
    #                 if contrast_info and not contrast_info.get('error'):
    #                     if 'compliance' in contrast_info:
    #                         compliance = contrast_info['compliance']
    #                         if not compliance.get('AA_normal', False):
    #                             violations.append("Fails AA Normal")
    #
    #                 if violations:
    #                     result.contrast_violations_count += 1
    #
    #                 result.detections.append(DetailedDetection(
    #                     text=text,
    #                     confidence=float(conf),
    #                     bbox=clean_bbox,
    #                     contrast_info=contrast_info,
    #                     color_info=color_info,
    #                     wcag_violations=violations
    #                 ))
    #
    #             # Copy image to appropriate text category folder
    #             dest_folder = self.categories.get(category, self.categories["with_text"])
    #             dest_path = os.path.join(dest_folder, filename)
    #             shutil.copy2(image_path, dest_path)
    #
    #             result.new_path = dest_path
    #
    #             logger.info(
    #                 f"✓ Detected {len(detections)} text regions. Category: {category}. Violations: {result.contrast_violations_count}")
    #         else:
    #             logger.debug(f"No text detected in {filename}")
    #
    #     except Exception as e:
    #         logger.error(f"Error processing {filename}: {str(e)}")
    #         import traceback
    #         traceback.print_exc()
    #
    #     return result
    def detect_text_in_image(self, image_path: str) -> TextDetectionResult:
        """Use EasyOCR to detect text and run contrast analysis"""
        logger.info(f"Processing: {image_path}")

        filename = Path(image_path).name
        category = self._determine_category(image_path)

        result = TextDetectionResult(
            filename=filename,
            original_path=image_path,
            category=category
        )

        try:
            detections = self.reader.readtext(image_path)

            if len(detections) > 0:
                result.has_text = True
                img = cv2.imread(image_path)

                for bbox, text, conf in detections:
                    clean_bbox = [(int(p[0]), int(p[1])) for p in bbox]

                    try:
                        contrast_info = contrast_analyzer.analyze_text_region(img, clean_bbox)
                    except Exception as e:
                        logger.warning(f"Contrast analysis failed: {e}")
                        contrast_info = None

                    violations = []

                    color_info = None
                    if contrast_info and not contrast_info.get('error'):
                        try:
                            extracted = extract_color.extract_colors_from_mask(
                                contrast_info['region'],
                                contrast_info['mask'],
                                k_bg=3
                            )

                            if "error" not in extracted:
                                fg_color = extracted["foreground"]
                                bg_colors = extracted["background_palette"]

                                color_info = {
                                    "foreground": fg_color,
                                    "background_palette": bg_colors,
                                    "contrast_checks": []
                                }

                                fg_lum = fg_color['luminance']
                                for bg in bg_colors:
                                    bg_lum = bg['luminance']
                                    l1 = max(fg_lum, bg_lum)
                                    l2 = min(fg_lum, bg_lum)
                                    ratio = (l1 + 0.05) / (l2 + 0.05)
                                    compliance = contrast_analyzer.check_wcag_compliance(ratio)
                                    color_info["contrast_checks"].append({
                                        "bg_color": bg,
                                        "ratio": round(ratio, 2),
                                        "compliance": compliance
                                    })
                                    if not compliance['AA_normal']:
                                        violations.append(f"Fails AA Normal vs BG {bg['hex']}")
                            else:
                                logger.warning(f"Color extraction failed: {extracted['error']}")

                        except Exception as cp_err:
                            logger.warning(f"Color picker failed for region: {cp_err}")

                    if contrast_info and not contrast_info.get('error'):
                        if 'compliance' in contrast_info:
                            compliance = contrast_info['compliance']
                            if not compliance.get('AA_normal', False):
                                violations.append("Fails AA Normal")

                    if violations:
                        result.contrast_violations_count += 1

                    result.detections.append(DetailedDetection(
                        text=text,
                        confidence=float(conf),
                        bbox=clean_bbox,
                        contrast_info=contrast_info,
                        color_info=color_info,
                        wcag_violations=violations
                    ))

                # Copy image to appropriate text category folder
                dest_folder = self.categories.get(category, self.categories["with_text"])
                dest_path = os.path.join(dest_folder, filename)
                shutil.copy2(image_path, dest_path)

                result.new_path = dest_path

                logger.info(
                    f"✓ Detected {len(detections)} text regions. Category: {category}. Violations: {result.contrast_violations_count}")
            else:
                logger.debug(f"No text detected in {filename}")

        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            import traceback
            traceback.print_exc()

        return result

    def scan_directory(self):
        """Scan source directory for images"""
        logger.info(f"Scanning directory: {self.source_directory}")

        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
        image_files = []
        for root, dirs, files in os.walk(self.source_directory):
            # Avoid scanning our own output directories if they are inside source
            if "text_detected" in root or "contrast" in root:
                continue

            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_files.append(os.path.join(root, file))

        for idx, image_path in enumerate(image_files, 1):
            result = self.detect_text_in_image(image_path)
            self.results.append(result)

            if result.has_text:
                print(f"  ✓ Found {len(result.detections)} text regions ({result.category})")
                if result.contrast_violations_count > 0:
                    print(f"  ⚠ {result.contrast_violations_count} contrast violations detected!")
            else:
                print(f"  . No text")


class TextClassification:

    def __init__(self, source_directory: str, output_directory: Optional[str] = None):
        self.source_directory = source_directory

        # Main output directory inside the crawl directory
        if output_directory is None:
            self.base_output_dir = source_directory
        else:
            self.base_output_dir = output_directory

        self.text_detected_dir = os.path.join(self.base_output_dir, "text_detected")
        self.contrast_dir = os.path.join(self.text_detected_dir, "contrast")

        self.results: List[TextDetectionResult] = []

        # Create output directory structure
        self._create_directories()

        # Initialize EasyOCR Reader
        logger.info("Initializing EasyOCR Reader (loading models)...")
        self.reader = OCRReader(source_directory)
        logger.info("EasyOCR Reader initialized")

    def _create_directories(self):
        """Create necessary output directories"""
        self.categories = {
            "button_text": os.path.join(self.text_detected_dir, "button_text"),
            "informational_text": os.path.join(self.text_detected_dir, "informational_text"),
            "logo_text": os.path.join(self.text_detected_dir, "logo_text"),
            "with_text": os.path.join(self.text_detected_dir, "with_text")
        }

        # Create text detection folders
        for path in self.categories.values():
            Path(path).mkdir(parents=True, exist_ok=True)

        # Create contrast folder
        Path(self.contrast_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directories in {self.base_output_dir}")

    def save_reports(self):
        """Save JSON report and Contrast Markdown report"""

        # 1. JSON Report
        json_file = os.path.join(self.text_detected_dir, "text_detection_report.json")

        images_with_text = sum(1 for r in self.results if r.has_text)
        images_with_violations = sum(1 for r in self.results if r.contrast_violations_count > 0)

        report = TextDetectionReport(
            scan_date=datetime.utcnow().isoformat(),
            source_directory=self.source_directory,
            total_images_scanned=len(self.results),
            images_with_text=images_with_text,
            images_with_contrast_violations=images_with_violations,
            results=self.results
        )

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(
                report.model_dump(),
                f,
                indent=2,
                ensure_ascii=False,
                default=_json_serializer
            )

        # 2. Contrast Markdown Report
        md_file = os.path.join(self.contrast_dir, "contrast_report.md")
        self._generate_contrast_markdown(md_file, images_with_violations)

        print(f"\n{'=' * 60}")
        print(f"SCAN COMPLETE")
        print(f"Total images: {len(self.results)}")
        print(f"With text: {images_with_text}")
        print(f"Contrast violations: {images_with_violations}")
        print(f"Reports saved to:")
        print(f"  - {json_file}")
        print(f"  - {md_file}")
        print(f"{'=' * 60}")

    def _generate_contrast_markdown(self, output_path: str, violation_count: int):
        """Generate a user-friendly Markdown report for contrast analysis"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# WCAG Contrast Analysis Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Violations Found:** {violation_count}\n\n")

            f.write("## Color Palette Analysis\n\n")
            for result in self.results:
                # Check if this image has any detections with color info
                if result.has_text and any(d.color_info for d in result.detections):
                    f.write(f"### Image: `{result.filename}`\n")
                    f.write(f"**Path**: `{result.original_path}`\n\n")

                    for i, det in enumerate(result.detections, 1):
                        if det.color_info:
                            text_snippet = det.text.replace("\n", " ")[:30]
                            # Clean up snippet
                            if len(det.text) > 30: text_snippet += "..."

                            f.write(f"#### {i}. Text: \"{text_snippet}\"\n")

                            fg = det.color_info.get("foreground", {})
                            f.write(
                                f"- **Detected Text Color**: {fg.get('hex', 'N/A')} (Lum: {fg.get('luminance', 'N/A')})\n\n")

                            f.write("| Background | Ratio | AA Normal | AA Large | AAA Normal | AAA Large |\n")
                            f.write("|---|---|---|---|---|---|\n")

                            for check in det.color_info.get("contrast_checks", []):
                                bg = check['bg_color']
                                ratio = check['ratio']
                                comp = check['compliance']

                                aa = "✅" if comp['AA_normal'] else "❌"
                                aa_lg = "✅" if comp['AA_large'] else "❌"
                                aaa = "✅" if comp['AAA_normal'] else "❌"
                                aaa_lg = "✅" if comp['AAA_large'] else "❌"

                                f.write(f"| {bg['hex']} | {ratio}:1 | {aa} | {aa_lg} | {aaa} | {aaa_lg} |\n")
                            f.write("\n")
                    f.write("---\n\n")


def main():
    if len(sys.argv) > 1:
        source_directory = sys.argv[1]
    else:
        # Default fallback logic
        source_directory = "crawled_images"
        if os.path.exists(source_directory):
            crawl_dirs = [d for d in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, d))]
            if crawl_dirs:
                crawl_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(source_directory, x)), reverse=True)
                source_directory = os.path.join(source_directory, crawl_dirs[0])

    if not os.path.exists(source_directory):
        print(f"Error: Directory {source_directory} not found.")
        return

    detector = OCRPreprocessing(source_directory)
    detector.scan_directory()
    save = TextClassification(source_directory)
    save.save_reports()


# if __name__ == "__main__":
#     main()



