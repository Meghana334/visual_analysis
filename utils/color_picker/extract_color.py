import cv2
import numpy as np
import easyocr
from collections import Counter

# -------------------------
# Utilities
# -------------------------

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def relative_luminance(rgb):
    rgb = np.array(rgb) / 255.0
    rgb = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


# -------------------------
# KMeans helper
# -------------------------

def cluster_colors(pixels, k):
    pixels = np.float32(pixels)
    _, labels, centers = cv2.kmeans(
        pixels, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2),
        10, cv2.KMEANS_RANDOM_CENTERS
    )

    counts = Counter(labels.flatten())
    total = sum(counts.values())

    colors = []
    for idx, count in counts.items():
        rgb = centers[idx].astype(int)
        colors.append({
            "rgb": tuple([int(c) for c in rgb]),  # Convert to standard int list then tuple
            "hex": rgb_to_hex(rgb),
            "percent": float(round(count / total * 100, 2)), # Cast to float
            "luminance": float(relative_luminance(rgb)) # Cast to float
        })

    return sorted(colors, key=lambda x: x["percent"], reverse=True)


# -------------------------
# Global image palette
# -------------------------

def extract_global_colors(img_rgb, k=8):
    pixels = img_rgb.reshape(-1, 3)
    return cluster_colors(pixels, k)


# -------------------------
# Text-adjacent background pixels
# -------------------------

def extract_adjacent_text_pixels(img_rgb, bbox, padding=6):
    h_img, w_img, _ = img_rgb.shape
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]

    x0 = max(min(xs) - padding, 0)
    y0 = max(min(ys) - padding, 0)
    x1 = min(max(xs) + padding, w_img)
    y1 = min(max(ys) + padding, h_img)

    region = img_rgb[y0:y1, x0:x1].copy()

    mask = np.ones(region.shape[:2], dtype=bool)

    tx0, ty0 = min(xs) - x0, min(ys) - y0
    tx1, ty1 = max(xs) - x0, max(ys) - y0

    mask[ty0:ty1, tx0:tx1] = False

    return region[mask]

def extract_colors_from_mask(region: np.ndarray,
                             mask: np.ndarray,
                             k_bg: int = 3) -> dict:
    """
    Extract foreground and background colors using segmentation mask.

    mask == 255 → text
    mask == 0   → background
    """

    if region is None or mask is None:
        return {"error": "Region or mask missing"}

    # Convert BGR → RGB
    region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)

    text_pixels = region_rgb[mask == 255]
    bg_pixels = region_rgb[mask == 0]

    if len(text_pixels) == 0 or len(bg_pixels) == 0:
        return {"error": "Invalid segmentation"}

    # ---------- FOREGROUND ----------
    fg_clusters = cluster_colors(text_pixels, k=2)

    # Choose dominant cluster (most %)
    fg_color = max(fg_clusters, key=lambda c: c["percent"])

    # ---------- BACKGROUND ----------
    bg_clusters = cluster_colors(bg_pixels, k=k_bg)

    return {
        "foreground": fg_color,
        "background_palette": bg_clusters
    }


# -------------------------
# Text color extraction
# -------------------------

def extract_text_color(img_rgb, bbox):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]

    crop = img_rgb[min(ys):max(ys), min(xs):max(xs)]
    pixels = crop.reshape(-1, 3)

    clusters = cluster_colors(pixels, k=2)

    # Pick the brightest cluster (white text)
    return max(clusters, key=lambda c: c["luminance"])



# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":
    image_path = "/home/meghana/Downloads/bc_asset1.jpg"

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    reader = easyocr.Reader(['en'], gpu=False)
    detections = reader.readtext(image_path)

    print("\n==============================")
    print("GLOBAL IMAGE COLORS")
    print("==============================")
    global_colors = extract_global_colors(img_rgb)
    for c in global_colors:
        print(c["hex"], c["percent"], "%")

    for bbox, text, conf in detections:
        bbox = [[int(p[0]), int(p[1])] for p in bbox]

        print("\n==============================")
        print(f'TEXT: "{text}"')
        print("==============================")

        # TEXT COLOR
        fg = extract_text_color(img_rgb, bbox)
        print("Text color (FG):")
        print(f"  {fg['hex']}  luminance={round(fg['luminance'],3)}")

        # BACKGROUND COLORS
        bg_pixels = extract_adjacent_text_pixels(img_rgb, bbox)
        bg_colors = cluster_colors(bg_pixels, k=3)

        print("Text-adjacent background colors:")
        for bg in bg_colors:
            print(f"  {bg['hex']}  luminance={round(bg['luminance'],3)}")