# Handles document loading and text chunking

import pdfplumber

def extract_text(uploaded_file):
    """Extract text from PDF or TXT file."""
    if uploaded_file.name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start += chunk_size - overlap

    return chunks