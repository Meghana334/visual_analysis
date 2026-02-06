import cv2
import numpy as np
from typing import Tuple, Dict, Any, List, Union

def srgb_to_linear(channel: np.ndarray) -> np.ndarray:
    """
    Convert sRGB color values to linear RGB (remove gamma correction).
    Vectorized implementation for performance.
    
    Args:
        channel: Normalized RGB channel values (0-1 range)
        
    Returns:
        Linear RGB values (0-1 range)
    """
    return np.where(
        channel <= 0.04045,
        channel / 12.92,
        np.power((channel + 0.055) / 1.055, 2.4)
    )

def calculate_luminance_contrast(region: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate WCAG 2.1 compliant relative luminance and contrast ratio.
    
    Args:
        region: BGR image region containing text
        mask: Binary mask (255=text, 0=background)
        
    Returns:
        Tuple of (text_luminance, bg_luminance, contrast_ratio)
        Returns (None, None, None) if segmentation failed
    """
    # Step 1: Convert BGR to RGB
    rgb_region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
    
    # Step 2: Normalize pixel values to 0-1 range
    rgb_normalized = rgb_region / 255.0
    
    # Step 3: Convert sRGB to linear RGB
    linear_r = srgb_to_linear(rgb_normalized[:, :, 0])
    linear_g = srgb_to_linear(rgb_normalized[:, :, 1])
    linear_b = srgb_to_linear(rgb_normalized[:, :, 2])
    
    # Step 4: Calculate relative luminance using WCAG formula
    relative_luminance = (0.2126 * linear_r + 
                         0.7152 * linear_g + 
                         0.0722 * linear_b)
    
    # Step 5: Extract text and background pixels using the mask
    text_luminance_pixels = relative_luminance[mask == 255]
    bg_luminance_pixels = relative_luminance[mask == 0]
    
    if len(text_luminance_pixels) == 0 or len(bg_luminance_pixels) == 0:
        return None, None, None
    
    # Step 6: Calculate average luminance
    text_L = float(np.mean(text_luminance_pixels))
    bg_L = float(np.mean(bg_luminance_pixels))
    
    # Step 7: Calculate contrast ratio
    lighter = max(text_L, bg_L)
    darker = min(text_L, bg_L)
    contrast_ratio = (lighter + 0.05) / (darker + 0.05)
    
    return text_L, bg_L, contrast_ratio

def segment_text_region(region: np.ndarray) -> np.ndarray:
    """
    Separate text pixels from background pixels using Otsu's thresholding.
    
    Args:
        region: BGR image region containing text
        
    Returns:
        Binary mask where white (255) = text, black (0) = background
    """
    # Convert to grayscale
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply Otsu's thresholding
    # Note: We assume text is darker than background usually, OR we rely on the fact 
    # that usually text is the "foreground". 
    # cv2.THRESH_BINARY_INV assumes foreground is darker (if using simple threshold).
    # With Otsu, it finds the split. We need to know which is text.
    # Heuristic: Text usually occupies less area than background in a bounding box.
    
    # Get standard Otsu threshold
    d, thresh_simple = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Check pixel counts to decide if we need to invert
    white_pixels = np.sum(thresh_simple == 255)
    total_pixels = thresh_simple.size
    
    if white_pixels > total_pixels / 2:
        # If "white" is majority, assume it's background, so text is black.
        # We want text to be white in our mask (255).
        # So we invert.
        mask = cv2.bitwise_not(thresh_simple)
    else:
        # "White" is minority, likely text.
        mask = thresh_simple
        
    # Apply morphological closing to fill small gaps in text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def get_average_rgb(region: np.ndarray, mask: np.ndarray) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Calculate average RGB for foreground (mask=255) and background (mask=0).
    """
    # Create masked arrays
    # region is BGR
    b_channel = region[:,:,0]
    g_channel = region[:,:,1]
    r_channel = region[:,:,2]
    
    # Foreground
    fg_mask_bool = (mask == 255)
    if np.any(fg_mask_bool):
        fg_r = int(np.mean(r_channel[fg_mask_bool]))
        fg_g = int(np.mean(g_channel[fg_mask_bool]))
        fg_b = int(np.mean(b_channel[fg_mask_bool]))
        fg_rgb = (fg_r, fg_g, fg_b)
    else:
        fg_rgb = (0, 0, 0) # Fallback
        
    # Background
    bg_mask_bool = (mask == 0)
    if np.any(bg_mask_bool):
        bg_r = int(np.mean(r_channel[bg_mask_bool]))
        bg_g = int(np.mean(g_channel[bg_mask_bool]))
        bg_b = int(np.mean(b_channel[bg_mask_bool]))
        bg_rgb = (bg_r, bg_g, bg_b)
    else:
        bg_rgb = (255, 255, 255) # Fallback
        
    return fg_rgb, bg_rgb

def check_wcag_compliance(ratio: float) -> Dict[str, Any]:
    """
    Check WCAG 2.1 complaince levels.
    """
    return {
        "contrast_ratio": round(ratio, 2),
        "AA_normal": ratio >= 4.5,
        "AA_large": ratio >= 3.0,
        "AAA_normal": ratio >= 7.0,
        "AAA_large": ratio >= 4.5
    }

def analyze_text_region(image: Union[str, np.ndarray], bbox: List[Tuple[int, int]]) -> Dict[str, Any]:
    """
    Main entry point for analyzing contrast of a specific text region.
    """
    try:
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                return {"error": "Could not load image"}
        else:
            img = image
            
        # Crop region
        pts = np.array(bbox, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        
        # Add small padding to ensure we capture some background if bbox is tight
        pad = 2
        h_img, w_img = img.shape[:2]
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(w_img - x, w + 2*pad)
        h = min(h_img - y, h + 2*pad)
        
        region = img[y:y+h, x:x+w]
        
        if region.size == 0:
            return {"error": "Empty region"}

        # Segmentation
        mask = segment_text_region(region)
        
        # Calculate Stats
        text_L, bg_L, ratio = calculate_luminance_contrast(region, mask)
        
        if text_L is None:
            return {"error": "Segmentation failed"}
            
        # Get Average Colors for Report
        fg_rgb, bg_rgb = get_average_rgb(region, mask)
        
        # Compliance
        compliance = check_wcag_compliance(ratio)
        
        return {
            "foreground_color": fg_rgb,
            "background_color": bg_rgb,
            "luminance_fg": round(text_L, 4),
            "luminance_bg": round(bg_L, 4),
            "contrast_ratio": round(ratio, 2),
            "compliance": compliance
        }
        
    except Exception as e:
        return {"error": str(e)}

# Backward compatibility aliases if needed
calculate_contrast_ratio = None # Deprecated in favor of calculating inside
calculate_luminance = None # Deprecated
