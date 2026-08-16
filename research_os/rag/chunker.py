import os
import re
from typing import List, Union

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()


SECTION_PATTERN = re.compile(
    r'^(?:(?:[IVXLCDM]+\.|[0-9]+(?:\.[0-9]+)*)\s+)?'
    r'(abstract|introduction|background|related\s+work|methodology|methods|'
    r'proposed\s+approach|architecture|system\s+design|experiment(?:s)?|'
    r'evaluation|result(?:s)?|discussion|conclusion(?:s)?|'
    r'limitation(?:s)?|future\s+work|reference(?:s)?)\b',
    re.IGNORECASE
)


def extract_section(text: str, default: str = "General") -> str:
    """Detect an academic section heading near the beginning of a text block."""

    lines = text.split("\n")

    for line in lines[:5]:
        line = line.strip()

        if not line:
            continue

        match = SECTION_PATTERN.search(line)

        if match:
            return match.group(1).title()

    return default


def _to_document(item: Union[Document, dict]) -> Document:
    """
    Convert loader output into a LangChain Document.

    The paper loader currently returns dictionaries:
        {
            "page_content": "...",
            "metadata": {...}
        }

    The chunker works internally with Document objects.
    """

    if isinstance(item, Document):
        return item

    if isinstance(item, dict):
        return Document(
            page_content=item.get("page_content", ""),
            metadata=item.get("metadata", {})
        )

    raise TypeError(
        f"Unsupported document type: {type(item)}"
    )


def split_documents(
    documents: List[Union[Document, dict]],
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[Document]:

    if chunk_size is None:
        chunk_size = int(os.getenv("CHUNK_SIZE", 1200))

    if chunk_overlap is None:
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = []

    paper_chunk_counters = {}

    for item in documents:

        # Convert loader dictionary → LangChain Document
        doc = _to_document(item)

        if not doc.page_content.strip():
            continue

        metadata = doc.metadata or {}

        paper_id = metadata.get("paper_id")

        if not paper_id:
            raise ValueError(
                "Document is missing paper_id. "
                "Every paper document must contain a Semantic Scholar paper_id."
            )

        title = metadata.get("title", "Unknown Title")

        authors = metadata.get("authors", [])

        if isinstance(authors, str):
            authors = [
                a.strip()
                for a in authors.split(",")
                if a.strip()
            ]

        year = metadata.get("year")

        source = metadata.get(
            "source",
            metadata.get("source_url", "Unknown Source")
        )

        page_val = metadata.get("page")

        if page_val is not None:
            try:
                page = int(page_val) + 1
            except (ValueError, TypeError):
                page = str(page_val)
        else:
            page = "N/A"

        section = extract_section(
            doc.page_content,
            metadata.get("section", "General")
        )

        split_texts = text_splitter.split_text(
            doc.page_content
        )

        for text in split_texts:

            if not text.strip():
                continue

            if paper_id not in paper_chunk_counters:
                paper_chunk_counters[paper_id] = 0

            paper_chunk_counters[paper_id] += 1

            chunk_idx = paper_chunk_counters[paper_id]

            chunk_metadata = {
                "paper_id": paper_id,
                "title": title,
                "authors": authors,
                "year": year,
                "section": extract_section(text, section),
                "page": page,
                "chunk_id": f"{paper_id}_chunk_{chunk_idx}",
                "source": source
            }

            chunks.append(
                Document(
                    page_content=text,
                    metadata=chunk_metadata
                )
            )

    return chunks