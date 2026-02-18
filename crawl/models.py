# crawl/models.py
from typing import Optional
from pydantic import BaseModel

class ImageData(BaseModel):
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