import os
from datetime import datetime
from langdetect import detect, DetectorFactory

# make langdetect consistent
DetectorFactory.seed = 0


# -----------------------------
# LANGUAGE DETECTION
# -----------------------------
def detect_language(text: str) -> str:
    try:
        lang = detect(text)

        if lang == "bn":
            return "Bangla"
        elif lang == "en":
            return "English"
        else:
            return "Mixed"

    except:
        return "Unknown"


# -----------------------------
# DOCUMENT TYPE DETECTION
# -----------------------------
def get_document_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        return "PDF"
    elif ext in [".png", ".jpg", ".jpeg"]:
        return "Image"
    else:
        return "Unknown"


# -----------------------------
# MAIN METADATA CREATION
# -----------------------------
def create_metadata(filename: str, chunks: list):
    """
    chunks: list of text chunks
    return: list of metadata dict per chunk
    """

    metadata_list = []

    doc_type = get_document_type(filename)
    date = datetime.now().strftime("%Y-%m-%d")

    for i, chunk in enumerate(chunks):

        # detect language per chunk (important for Bangla/English mix)
        language = detect_language(chunk)

        metadata = {
            "filename": filename,
            "chunk_id": i,
            "page": i + 1,
            "language": language,
            "document_type": doc_type,
            "date": date,
            "length": len(chunk)
        }

        metadata_list.append(metadata)

    return metadata_list