from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"page": page_number, "text": text.strip()})
    return pages

def create_chunks(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = []
    for page in pages:
        page_chunks = text_splitter.split_text(page["text"])
        for chunk in page_chunks:
            chunks.append({"text": chunk, "page": page["page"]})
    return chunks