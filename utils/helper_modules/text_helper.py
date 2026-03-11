import os
import json
import shutil
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
from pydantic import BaseModel, Field




class DetailedDetection(BaseModel):

    """Detailed information about a single detected text region"""
    text: str
    confidence: float
    bbox: List[tuple[int, int]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
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



def _json_serializer(obj):
    """Convert numpy types to JSON-serializable Python types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")