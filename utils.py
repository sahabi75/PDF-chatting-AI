from PyPDF2 import PdfReader

from pdf2image import convert_from_path

import os

from ocr import extract_text_from_image

from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_pdf_text(pdf_path):

    text = ""

    reader = PdfReader(pdf_path)

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text

    if text.strip():

        return text

    images = convert_from_path(pdf_path)

    for i, image in enumerate(images):

        image_path = f"temp/page_{i}.png"

        image.save(image_path)

        text += extract_text_from_image(image_path)

        os.remove(image_path)

    return text


def get_text_chunks(text):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=200

    )

    return splitter.split_text(text)