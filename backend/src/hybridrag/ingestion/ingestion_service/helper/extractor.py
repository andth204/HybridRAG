import io
import json
from pathlib import Path
from io import BytesIO
from typing import Optional
import pdfplumber
from docx import Document
import csv
import docx2txt

def _txt(d: BytesIO) -> str:
    return d.read().decode("utf-8", errors="replace")

def _pdf(d: BytesIO) -> str:
    parts = []
    with pdfplumber.open(d) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    return "\n\n".join(parts)

def _docx(d: BytesIO) -> str:
    return "\n\n".join(p.text for p in Document(d).paragraphs if p.text.strip())

def _csv(d: BytesIO) -> str:
    return "\n".join(
        ", ".join(r)
        for r in csv.reader(io.StringIO(d.read().decode("utf-8", errors="replace")))
    )

def _json_file(d: BytesIO) -> str:
    return json.dumps(
        json.loads(d.read().decode("utf-8", errors="replace")),
        ensure_ascii=False,
        indent=2,
    )

def _doc(d: BytesIO) -> str:
    d.seek(0)
    temp_path = "/tmp/temp_doc_file.doc"
    with open(temp_path, "wb") as f:
        f.write(d.read())
    text = docx2txt.process(temp_path)
    import os
    os.remove(temp_path)
    return text
    
_EXT = {
    ".txt": _txt,
    ".md": _txt,
    ".pdf": _pdf,
    ".docx": _docx,
    ".doc": _doc,
    ".csv": _csv,
    ".json": _json_file,
}

def extract_text(key: str, data: BytesIO) -> Optional[str]:
    fn = _EXT.get(Path(key).suffix.lower())
    if not fn:
        return None
    try:
        data.seek(0)
        text = fn(data).strip()
        return text or None
    except Exception:
        return None

async def fetch_and_extract(minio_client, bucket: str, key: str) -> Optional[str]:
    bio = await minio_client.get(bucket, key)
    return extract_text(key, bio)