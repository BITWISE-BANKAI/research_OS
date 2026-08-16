"""tools/paper_indexer.py

Indexes a research paper into ChromaDB and MongoDB.
"""

import uuid as _uuid
from typing import List, Optional

from langchain_core.tools import tool
from langchain_core.documents import Document

from rag.chroma_store import ChromaStore
from rag.mongo_store import MongoStore
from rag.chunker import split_documents
from tools.paper_loader import load_paper, fetch_paper_details
from utils.deduplication import is_duplicate
from utils.logging import setup_logger


logger = setup_logger("PaperIndexer")


@tool
def index_paper(
    paper_id: str,
    paper_data: Optional[dict] = None
) -> str:
    """
    Download, load, chunk and index a research paper.

    Pipeline:

        paper_id
            ↓
        Semantic Scholar metadata
            ↓
        PDF / abstract loading
            ↓
        LangChain Documents
            ↓
        chunking
            ↓
        MongoDB chunks
            ↓
        ChromaDB embeddings
    """

    try:

        # ---------------------------------------------------------
        # 0. Validate paper_id
        # ---------------------------------------------------------

        paper_id = str(paper_id).strip()

        if not paper_id:
            return "Error: paper_id is required."

        logger.info("=" * 70)
        logger.info(f"STARTING INDEXING: {paper_id}")
        logger.info("=" * 70)


        # ---------------------------------------------------------
        # 1. Initialize stores
        # ---------------------------------------------------------

        chroma_store = ChromaStore()
        mongo_store = MongoStore()


        # ---------------------------------------------------------
        # 2. Check current indexing state
        # ---------------------------------------------------------

        chroma_has_chunks = chroma_store.paper_exists(paper_id)

        mongo_has_metadata = mongo_store.paper_exists(paper_id)

        existing_mongo_chunks = mongo_store.get_chunks(paper_id)

        mongo_chunk_count = len(existing_mongo_chunks)

        mongo_has_chunks = mongo_chunk_count > 0

        logger.info(
            f"[STATE] paper_id={paper_id} | "
            f"Mongo metadata={mongo_has_metadata} | "
            f"Mongo chunks={mongo_chunk_count} | "
            f"Chroma chunks={chroma_has_chunks}"
        )


        # ---------------------------------------------------------
        # 3. Already completely indexed
        # ---------------------------------------------------------

        if chroma_has_chunks and mongo_has_chunks:

            logger.info(
                f"[SKIP] Paper {paper_id} already exists "
                f"in MongoDB and ChromaDB."
            )

            return (
                f"Paper already indexed. "
                f"Mongo chunks={mongo_chunk_count}, "
                f"ChromaDB chunks=present."
            )


        is_known_id = (
            chroma_has_chunks
            or mongo_has_metadata
        )


        # ---------------------------------------------------------
        # 4. Get paper metadata
        # ---------------------------------------------------------

        details = fetch_paper_details(paper_id)

        if not details and isinstance(paper_data, dict):
            details = dict(paper_data)

        if not details:
            details = mongo_store.get_paper(paper_id)

        if not details:
            details = {
                "paper_id": paper_id,
                "title": f"Paper {paper_id}",
                "abstract": "Metadata unavailable.",
                "authors": ["Unknown"],
                "year": None,
                "url": f"https://www.semanticscholar.org/paper/{paper_id}",
            }

        # ---------------------------------------------------------
        # IMPORTANT:
        # The requested paper_id is ALWAYS authoritative.
        # ---------------------------------------------------------

        details["paper_id"] = paper_id

        logger.info(
            f"[METADATA] title={details.get('title')} | "
            f"paper_id={details.get('paper_id')}"
        )


        # ---------------------------------------------------------
        # 5. Duplicate check
        # ---------------------------------------------------------

        if not is_known_id:

            existing_papers = (
                mongo_store.get_all_papers()
                or chroma_store.get_all_papers()
            )

            if is_duplicate(details, existing_papers):

                logger.info(
                    f"[DUPLICATE] "
                    f"{details.get('title')}"
                )

                return "Paper already indexed."


        # ---------------------------------------------------------
        # 6. Save paper metadata
        # ---------------------------------------------------------

        mongo_store.save_paper(details)

        logger.info(
            f"[MONGO] Paper metadata saved: {paper_id}"
        )


        # ---------------------------------------------------------
        # 7. Load paper
        # ---------------------------------------------------------

        logger.info(
            f"[LOAD] Loading paper: {paper_id}"
        )

        pages_raw = load_paper.invoke({
            "paper_id": paper_id
        })

        logger.info(
            f"[LOAD] Loader returned type={type(pages_raw).__name__}"
        )

        logger.info(
            f"[LOAD] Number of returned items="
            f"{len(pages_raw) if isinstance(pages_raw, list) else 'N/A'}"
        )


        # ---------------------------------------------------------
        # 8. Detect loader failure
        # ---------------------------------------------------------

        has_pages = (
            isinstance(pages_raw, list)
            and len(pages_raw) > 0
            and isinstance(pages_raw[0], dict)
            and "error" not in pages_raw[0]
        )


        # ---------------------------------------------------------
        # 9. Convert loader output → LangChain Documents
        # ---------------------------------------------------------

        documents: List[Document] = []

        if has_pages:

            for page in pages_raw:

                content = page.get("page_content", "")

                metadata = dict(
                    page.get("metadata", {})
                )

                # Force authoritative ID
                metadata["paper_id"] = paper_id

                if content and content.strip():

                    documents.append(
                        Document(
                            page_content=content,
                            metadata=metadata
                        )
                    )

        else:

            logger.warning(
                f"[LOAD] Full PDF unavailable for {paper_id}. "
                f"Using abstract/content fallback."
            )

            if details.get("content"):

                content_text = str(
                    details["content"]
                )

            elif (
                details.get("abstract")
                and details.get("abstract") != "N/A"
            ):

                content_text = (
                    f"Title: {details.get('title')}\n\n"
                    f"Abstract: {details.get('abstract')}"
                )

            else:

                content_text = (
                    f"Title: {details.get('title')}\n"
                    f"Authors: {', '.join(details.get('authors', []))}\n"
                    f"Year: {details.get('year', 'N/A')}"
                )


            fallback_meta = {
                "paper_id": paper_id,
                "title": details.get(
                    "title",
                    "Untitled Paper"
                ),
                "authors": details.get(
                    "authors",
                    []
                ),
                "year": details.get("year"),
                "url": details.get("url", ""),
                "doi": details.get("doi"),
                "arxiv_id": details.get("arxiv_id"),
                "open_access_url": details.get(
                    "open_access_url"
                ),
                "source": (
                    details.get("url")
                    or details.get("open_access_url")
                    or "Manual/Abstract"
                ),
                "page": 0,
            }

            documents.append(
                Document(
                    page_content=content_text,
                    metadata=fallback_meta
                )
            )


        logger.info(
            f"[DOCUMENTS] Created {len(documents)} Documents"
        )


        if not documents:

            return (
                f"Error: No documents were created "
                f"for paper {paper_id}."
            )


        # ---------------------------------------------------------
        # 10. Chunk documents
        # ---------------------------------------------------------

        logger.info(
            f"[CHUNK] Starting chunking for {paper_id}"
        )

        chunks = split_documents(documents)

        logger.info(
            f"[CHUNK] Created {len(chunks)} chunks"
        )


        # ---------------------------------------------------------
        # 11. Safety fallback if chunker returns nothing
        # ---------------------------------------------------------

        if not chunks:

            logger.warning(
                f"[CHUNK] split_documents returned 0 chunks."
            )

            raw_text = " ".join(
                doc.page_content
                for doc in documents
            ).strip()

            if not raw_text:

                return (
                    f"Error: Documents contain no text "
                    f"for paper {paper_id}."
                )

            fallback_meta = dict(
                documents[0].metadata
            )

            fallback_meta["paper_id"] = paper_id

            fallback_meta["chunk_id"] = (
                f"{paper_id}_chunk_fallback"
            )

            chunks = [
                Document(
                    page_content=raw_text[:4000],
                    metadata=fallback_meta
                )
            ]

            logger.warning(
                "[CHUNK] Created fallback chunk."
            )


        # ---------------------------------------------------------
        # 12. Ensure every chunk has correct paper_id
        # ---------------------------------------------------------

        for index, chunk in enumerate(chunks):

            chunk.metadata["paper_id"] = paper_id

            if not chunk.metadata.get("chunk_id"):

                chunk.metadata["chunk_id"] = (
                    f"{paper_id}_chunk_{index + 1}"
                )


        logger.info(
            f"[CHUNK] Final chunk count = {len(chunks)}"
        )


        # ---------------------------------------------------------
        # 13. Save chunks to MongoDB
        # ---------------------------------------------------------

        if not mongo_has_chunks:

            chunks_json = []

            for chunk in chunks:

                chunks_json.append({
                    "chunk_id": (
                        chunk.metadata.get("chunk_id")
                        or str(_uuid.uuid4())
                    ),

                    "paper_id": paper_id,

                    "title": chunk.metadata.get(
                        "title"
                    ),

                    "authors": chunk.metadata.get(
                        "authors"
                    ),

                    "year": chunk.metadata.get(
                        "year"
                    ),

                    "section": chunk.metadata.get(
                        "section"
                    ),

                    "page": chunk.metadata.get(
                        "page"
                    ),

                    "source": chunk.metadata.get(
                        "source"
                    ),

                    "content": chunk.page_content,
                })


            logger.info(
                f"[MONGO] Saving {len(chunks_json)} chunks "
                f"for paper {paper_id}"
            )

            mongo_success = mongo_store.save_chunks(
                paper_id,
                chunks_json
            )

            if not mongo_success:

                return (
                    "Error: Failed to save chunks "
                    "to MongoDB."
                )


            # Verify immediately
            saved_chunks = mongo_store.get_chunks(
                paper_id
            )

            logger.info(
                f"[MONGO] Verification: "
                f"{len(saved_chunks)} chunks found "
                f"for {paper_id}"
            )

            if len(saved_chunks) == 0:

                return (
                    f"Error: save_chunks() reported success "
                    f"but MongoDB contains 0 chunks for "
                    f"paper_id={paper_id}."
                )

        else:

            logger.info(
                f"[MONGO] Already has "
                f"{mongo_chunk_count} chunks. "
                f"Skipping MongoDB write."
            )


        # ---------------------------------------------------------
        # 14. Add chunks to ChromaDB
        # ---------------------------------------------------------

        if not chroma_has_chunks:

            logger.info(
                f"[CHROMA] Adding {len(chunks)} chunks..."
            )

            success = chroma_store.add_documents(
                chunks
            )

            if not success:

                return (
                    "Error: Failed to store embeddings "
                    "in ChromaDB."
                )

            logger.info(
                f"[CHROMA] Successfully stored chunks."
            )

        else:

            logger.info(
                "[CHROMA] Chunks already exist. "
                "Skipping vector insertion."
            )


        # ---------------------------------------------------------
        # 15. Final verification
        # ---------------------------------------------------------

        final_mongo_chunks = mongo_store.get_chunks(
            paper_id
        )

        final_mongo_count = len(
            final_mongo_chunks
        )

        logger.info(
            f"[VERIFY] paper_id={paper_id} | "
            f"chunks_created={len(chunks)} | "
            f"mongo_chunks={final_mongo_count}"
        )


        if final_mongo_count == 0:

            return (
                f"Error: Indexing completed but MongoDB "
                f"contains 0 chunks for {paper_id}."
            )


        source_type = (
            "Full PDF text"
            if has_pages
            else "Abstract & metadata"
        )

        success_msg = (
            f"Paper '{details.get('title', paper_id)}' "
            f"indexed successfully "
            f"({source_type}). "
            f"Ingested {len(chunks)} chunks."
        )

        logger.info(
            f"[SUCCESS] {success_msg}"
        )

        return success_msg


    except Exception as e:

        logger.exception(
            f"[ERROR] Error during indexing "
            f"of paper {paper_id}: {e}"
        )

        return (
            f"Error occurred during indexing: {str(e)}"
        )