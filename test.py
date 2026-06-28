import pytesseract
from PIL import Image

# If Windows needs path (very common)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

print(extract_text_from_image("test.jpeg"))