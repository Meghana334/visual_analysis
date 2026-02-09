import os
import json
import shutil
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
import easyocr
import cv2
import numpy as np

# Import our custom contrast analyzer
try:
    import contrast_analyzer
except ImportError:
    # If running from a different directory, try to append the current directory
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import contrast_analyzer

# Import color_picker
try:
    import color_picker
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import color_picker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('text_detection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def detect_text_boxes(image_path: str):
    """
    Lightweight helper for other modules (contrast, color picker).
    Returns text bounding boxes as (x, y, w, h).
    """
    reader = easyocr.Reader(['en'], gpu=False)
    detections = reader.readtext(image_path)

    boxes = []
    for bbox, text, conf in detections:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]

        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))

        w = x_max - x_min
        h = y_max - y_min

        if w > 0 and h > 0:
            boxes.append((x_min, y_min, w, h))

    return boxes


class DetailedDetection(BaseModel):
    """Detailed information about a single detected text region"""
    text: str
    confidence: float
    bbox: List[List[int]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    contrast_info: Optional[Dict[str, Any]] = None
    color_info: Optional[Dict[str, Any]] = None  # New field for color palette
    wcag_violations: List[str] = Field(default_factory=list)

class TextDetectionResult(BaseModel):
    """Result of text detection in an image"""
    filename: str
    original_path: str
    has_text: bool = False
    detections: List[DetailedDetection] = Field(default_factory=list)
    contrast_violations_count: int = 0
    new_path: Optional[str] = None
    category: str = "other"  # button, logo, informational, other

class TextDetectionReport(BaseModel):
    """Complete text detection report"""
    scan_date: str
    source_directory: str
    total_images_scanned: int = 0
    images_with_text: int = 0
    images_with_contrast_violations: int = 0
    results: List[TextDetectionResult] = Field(default_factory=list)

class ImageTextDetector:
    """Detect text in images using EasyOCR and analyze contrast"""

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
        self.reader = easyocr.Reader(['en'], gpu=False) 
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
        # Optional: create a separate folder for images with violations inside contrast dir to link them
        # Path(os.path.join(self.contrast_dir, "violation_images")).mkdir(parents=True, exist_ok=True)

        logger.info(f"Created output directories in {self.base_output_dir}")

    def _determine_category(self, original_path: str) -> str:
        """Heuristic to determine category based on source path"""
        path_str = str(original_path).lower()
        
        if "button" in path_str:
            return "button_text"
        elif "logo" in path_str:
            return "logo_text"
        elif "informative" in path_str or "informational" in path_str:
            return "informational_text"
        else:
            return "with_text"

    def detect_text_in_image(self, image_path: str) -> TextDetectionResult:
        """Use EasyOCR to detect text and run contrast analysis"""
        logger.info(f"Processing: {image_path}")

        filename = Path(image_path).name
        # Determine category based on source path (folder structure from crawl.py)
        category = self._determine_category(image_path)
        
        result = TextDetectionResult(
            filename=filename,
            original_path=image_path,
            category=category
        )

        try:
            # Read image using EasyOCR
            detections = self.reader.readtext(image_path)
            
            if len(detections) > 0:
                result.has_text = True
                img = cv2.imread(image_path)
                
                for bbox, text, conf in detections:
                    clean_bbox = [[int(p[0]), int(p[1])] for p in bbox]
                    
                    # Contrast Analysis (existing)
                    contrast_info = contrast_analyzer.analyze_text_region(img, clean_bbox) if 'contrast_analyzer' in sys.modules else None

                    # Color Picker Integration (new)
                    color_info = None
                    if 'color_picker' in sys.modules:
                        try:
                            # Extract text color
                            fg_color = color_picker.extract_text_color(img, clean_bbox)
                            
                            # Extract background colors (palette)
                            bg_pixels = color_picker.extract_adjacent_text_pixels(img, clean_bbox)
                            bg_colors = color_picker.cluster_colors(bg_pixels, k=3)
                            
                            color_info = {
                                "foreground": fg_color,
                                "background_palette": bg_colors,
                                "contrast_checks": []
                            }
                            
                            # Perform contrast checks against palette
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

                        except Exception as cp_err:
                            logger.warning(f"Color picker failed for region: {cp_err}")
                    
                    violations = []
                    if contrast_info and not contrast_info.get('error'):
                         # Check compliance keys
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
                
                # If specifically "with_text" is meant to be a catch-all, we might want to copy ALL text images there too?
                # User request: "under text_detected/ button_text,informational_text, logo_text, with_text"
                # implying disjoint sets or at least categorized. 
                # If it didn't fit others, it goes to with_text (via _determine_category default).
                
                result.new_path = dest_path
                
                # We do NOT save contrast images separately unless requested, but user asked for "separate output for contrast check and report"
                # implies the report is the main thing. I won't duplicate images to contrast folder to save space unless strictly needed.
                
                logger.info(f"✓ Detected {len(detections)} text regions. Category: {category}. Violations: {result.contrast_violations_count}")
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

        print(f"\n{'=' * 60}")
        print(f"TEXT DETECTION STARTING (EasyOCR)")
        print(f"Images to process: {len(image_files)}")
        print(f"{'=' * 60}\n")

        for idx, image_path in enumerate(image_files, 1):
            print(f"[{idx}/{len(image_files)}] Processing {Path(image_path).name}...")
            result = self.detect_text_in_image(image_path)
            self.results.append(result)
            
            if result.has_text:
                print(f"  ✓ Found {len(result.detections)} text regions ({result.category})")
                if result.contrast_violations_count > 0:
                    print(f"  ⚠ {result.contrast_violations_count} contrast violations detected!")
            else:
                print(f"  . No text")

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
            json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)

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
            
            if violation_count == 0:
                f.write("✅ No contrast violations detected. All text meets WCAG AA standards.\n")
                return

            f.write("## Violations Detail\n\n")
            
            for result in self.results:
                if result.contrast_violations_count > 0:
                    f.write(f"### Image: `{result.filename}`\n")
                    f.write(f"- **Category**: {result.category}\n")
                    f.write(f"- **Violations**: {result.contrast_violations_count}\n")
                    f.write(f"- **Original Path**: `{result.original_path}`\n\n")
                    
                    f.write("| Detected Text | Contrast Ratio | FG Color | BG Color | WCAG Status |\n")
                    f.write("|---|---|---|---|---|\n")
                    
                    for det in result.detections:
                        if det.wcag_violations:
                            text_snippet = det.text.replace("\n", " ")[:30] + "..." if len(det.text) > 30 else det.text
                            
                            if det.contrast_info and "error" not in det.contrast_info:
                                ratio = det.contrast_info.get("contrast_ratio", "N/A")
                                fg = det.contrast_info.get("foreground_color", "N/A")
                                bg = det.contrast_info.get("background_color", "N/A")
                                status = "Wait" # Should be Fail
                                
                                # Format colors nicely
                                fg_str = f"rgb{fg}" if isinstance(fg, tuple) else str(fg)
                                bg_str = f"rgb{bg}" if isinstance(bg, tuple) else str(bg)
                                
                                status_emoji = "❌ FAIL"
                                f.write(f"| {text_snippet} | {ratio}:1 | {fg_str} | {bg_str} | {status_emoji} |\n")
                            else:
                                f.write(f"| {text_snippet} | Error | N/A | N/A | ❌ Error |\n")
                    f.write("\n---\n\n")

            # Add Color Analysis Section if available
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
                            f.write(f"- **Detected Text Color**: {fg.get('hex', 'N/A')} (Lum: {fg.get('luminance', 'N/A')})\n\n")
                            
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

    detector = ImageTextDetector(source_directory)
    detector.scan_directory()
    detector.save_reports()

if __name__ == "__main__":
    main()