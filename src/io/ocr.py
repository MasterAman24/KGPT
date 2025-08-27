from PIL import Image
import pytesseract

def ocr_image_to_text(img: Image.Image) -> str:
    img = img.convert("L")
    w, h = img.size
    scale = 1.25 if max(w, h) < 1500 else 1.0
    if scale != 1.0:
        img = img.resize((int(w*scale), int(h*scale)))
    text = pytesseract.image_to_string(img)
    return text.strip()
