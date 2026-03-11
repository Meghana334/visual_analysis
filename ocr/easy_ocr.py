import easyocr
from typing import Optional


class OCRReader:

    def __init__(self, source_directory: str, output_directory: Optional[str] = None):
        self.source_directory = source_directory
        self.output_directory = output_directory
        self.reader = self._create_reader()

    def _create_reader(self):
        return easyocr.Reader(['en'], gpu=False)

    def readtext(self, image_path: str):
        return self.reader.readtext(image_path)