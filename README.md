# Steps

1) Classification of Image type ( Functional, Informative, Decorative ) **Rule no: EN 301 549 **9.1.1.1****
2) Whether the text is present in the image ( Text on the image ,  Text as logo, Text within the image) Rule No: **EN 301 549 – Clause 9.1.4.5**
   2.1 Check For the contrast and luminance Foreground Contrast Text Ratio must be higher than the background (Minimum 4.5:1)
3) Whether the image has alt text or not ? **EN 301 549 **9.1.1.1****
   3.1) if image has the alt text whether the alt text is accurate or not ? **EN 301 549 – Clause 9.5.1.3**

# Modules

- Playwright (Crawler)
- onclick event checker (Interactive or Functional ) (Buttons , Clickable images, Logo's --> Logos sometimes might take us to home page))
- ALT text checker function
- Text Detection OCR
- Contrast checker ( using opencv image processing , Luminance)
- Segmentation using Bilatreral filtering (sperating foreground and backgorund)
- otsu's threshold (to binarise)
- Calculating Luminance
- Violation Checker module (calculated results are compared with wcag standard values)
- Hexa code colour picker

# Limitation

- Text detection OCR has some False positive text's for eg Text on product objects in the images are detected and considered for checking contrast and luminance
- Overlay text over images are sperated in crawler module so text over image module fails here so we have question how can we approach this issue

# Questions

1) Do we need test accesibility for product based images which might include faded text whch is a manufacturing design ?
