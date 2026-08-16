import os
from typing import List, Dict, Any, Optional
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from rag.chroma_store import ChromaStore
from rag.mongo_store import MongoStore
from rag.reranker import Reranker
from prompts.rag_prompts import RAG_PROMPT
from utils.logging import setup_logger

logger = setup_logger("Retriever")

class RAGRetriever:
    def __init__(self):
        self.store = ChromaStore()
        self.mongo_store = MongoStore()
        self.reranker = Reranker()
        
        # Load environment configuration
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.chat_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:latest")
        self.top_k = int(os.getenv("TOP_K", 5))
        
        # Initialize generation model
        self.llm = ChatOllama(
            base_url=self.base_url,
            model=self.chat_model,
            temperature=0
        )
        self.qa_chain = RAG_PROMPT | self.llm | StrOutputParser()

    def generate_answer(
        self,
        query: str,
        mode: str = "library",
        paper_ids: Optional[List[str]] = None,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieves evidence, reranks if necessary, constructs context, 
        and generates an evidence-grounded answer using the local Qwen3 model.
        Logs interactions to MongoDB.
        
        Returns:
            Dict containing: 'answer', 'sources', 'retrieved_chunks', 'status'
        """
        try:
            logger.info(f"RAG query received (mode: {mode}, paper_ids: {paper_ids}): '{query}'")
            
            # 1. Check if Ollama embeddings are healthy before querying ChromaDB
            embedding_healthy = self.store.is_embedding_healthy()
            
            raw_chunks = []
            if embedding_healthy:
                # Similarity Search in ChromaDB (only when real embeddings are available)
                raw_chunks = self.store.similarity_search(query, paper_ids=paper_ids, k=k)
            else:
                logger.warning("Ollama embeddings unavailable — skipping ChromaDB to avoid mismatched vectors.")
            
            # Hybrid / Fallback: Check MongoDB JSON store if ChromaDB returned empty or was skipped
            if not raw_chunks:
                if not embedding_healthy:
                    logger.info("Using MongoDB keyword fallback (Ollama offline).")
                else:
                    logger.info("ChromaDB returned 0 chunks. Checking MongoDB JSON store for fallback chunks...")
                    
                mongo_chunks = []
                if paper_ids and len(paper_ids) > 0:
                    for pid in paper_ids:
                        if pid:
                            mongo_chunks.extend(self.mongo_store.get_chunks(pid))
                else:
                    all_papers = self.mongo_store.get_all_papers()
                    for p in all_papers:
                        pid = p.get("paper_id")
                        if pid:
                            mongo_chunks.extend(self.mongo_store.get_chunks(pid))

                # Second fallback: pull raw documents from ChromaDB collection directly
                # (no embedding needed — just reads stored text and metadata)
                if not mongo_chunks:
                    logger.info("MongoDB chunks empty. Pulling raw documents from ChromaDB collection...")
                    try:
                        collection = self.store.collection
                        if paper_ids and len(paper_ids) > 0:
                            where_filter = {"paper_id": paper_ids[0]} if len(paper_ids) == 1 else {"paper_id": {"$in": paper_ids}}
                            chroma_raw = collection.get(where=where_filter)
                        else:
                            chroma_raw = collection.get()

                        docs = chroma_raw.get("documents", [])
                        metas = chroma_raw.get("metadatas", [])
                        ids = chroma_raw.get("ids", [])
                        for i, doc_text in enumerate(docs):
                            meta = metas[i] if i < len(metas) else {}
                            authors_raw = meta.get("authors", "")
                            if isinstance(authors_raw, str):
                                authors_val = [a.strip() for a in authors_raw.split(",") if a.strip()]
                            else:
                                authors_val = authors_raw or []
                            mongo_chunks.append({
                                "chunk_id": ids[i] if i < len(ids) else str(i),
                                "paper_id": meta.get("paper_id", ""),
                                "title": meta.get("title", ""),
                                "authors": authors_val,
                                "year": meta.get("year"),
                                "section": meta.get("section", "General"),
                                "page": meta.get("page", 0),
                                "source": meta.get("source", "ChromaDB-raw"),
                                "content": doc_text or "",
                            })
                        if mongo_chunks:
                            logger.info(f"Retrieved {len(mongo_chunks)} raw chunks from ChromaDB collection.")
                    except Exception as chroma_err:
                        logger.warning(f"ChromaDB raw fetch failed: {chroma_err}")

                # Third fallback: use paper metadata (abstract) as a minimal chunk
                if not mongo_chunks:
                    logger.info("No chunks anywhere. Using paper abstracts as last-resort chunks...")
                    all_papers = self.mongo_store.get_all_papers()
                    if paper_ids:
                        all_papers = [p for p in all_papers if p.get("paper_id") in paper_ids]
                    for p in all_papers:
                        abstract = p.get("abstract", "")
                        if abstract and abstract != "N/A":
                            mongo_chunks.append({
                                "chunk_id": f"abstract-{p.get('paper_id', '')}",
                                "paper_id": p.get("paper_id", ""),
                                "title": p.get("title", ""),
                                "authors": p.get("authors", []),
                                "year": p.get("year"),
                                "section": "Abstract",
                                "page": 0,
                                "source": "Paper-Metadata",
                                "content": f"Title: {p.get('title', '')}\n\nAbstract: {abstract}",
                            })

                if mongo_chunks:
                    query_terms = [t.lower() for t in query.split() if len(t) > 2]
                    def score_chunk(c):
                        text = (c.get("content", "") + " " + c.get("title", "")).lower()
                        return sum(1 for term in query_terms if term in text)

                    mongo_chunks.sort(key=score_chunk, reverse=True)
                    top_mongo = mongo_chunks[:k]

                    from langchain_core.documents import Document
                    for c in top_mongo:
                        raw_chunks.append(Document(
                            page_content=c.get("content", ""),
                            metadata={
                                "chunk_id": c.get("chunk_id"),
                                "paper_id": c.get("paper_id"),
                                "title": c.get("title"),
                                "authors": c.get("authors", []),
                                "year": c.get("year"),
                                "section": c.get("section", "General"),
                                "page": c.get("page", 0),
                                "source": c.get("source", "MongoDB"),
                            }
                        ))

            if not raw_chunks:
                res = {
                    "answer": "The database contains no indexed papers matching the scope of your query. Please search & index papers or click '➕ Add Manual Paper' first.",
                    "sources": [],
                    "retrieved_chunks": [],
                    "status": "no_indexed_papers"
                }
                return res

                
            # 2. Rerank Chunks (stub pass-through)
            ranked_chunks = self.reranker.rerank(query, raw_chunks, top_n=k)
            
            # 3. Build Context String
            context_blocks = []
            sources_set = set()
            sources_list = []
            retrieved_chunks_info = []
            
            for doc in ranked_chunks:
                # Add to context block
                title = doc.metadata.get("title", "Unknown Title")
                authors = doc.metadata.get("authors", [])
                year = doc.metadata.get("year", "N/A")
                sec = doc.metadata.get("section", "General")
                pg = doc.metadata.get("page", "N/A")
                src = doc.metadata.get("source", "N/A")
                pid = doc.metadata.get("paper_id", "N/A")
                
                authors_str = ", ".join(authors) if isinstance(authors, list) else str(authors)
                
                block = (
                    f"Paper: {title} | Authors: {authors_str} | Year: {year} | Section: {sec} | Page: {pg}\n"
                    f"Snippet: {doc.page_content}\n"
                )
                context_blocks.append(block)
                
                # Format chunk details for API
                retrieved_chunks_info.append({
                    "paper_id": pid,
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "section": sec,
                    "page": pg,
                    "source": src,
                    "content": doc.page_content
                })
                
                # Format unique source details
                if pid not in sources_set:
                    sources_set.add(pid)
                    sources_list.append({
                        "paper_id": pid,
                        "title": title,
                        "authors": authors,
                        "year": year,
                        "url": doc.metadata.get("url"),
                        "doi": doc.metadata.get("doi"),
                        "arxiv_id": doc.metadata.get("arxiv_id"),
                        "open_access_url": doc.metadata.get("open_access_url")
                    })
                    
            context_str = "\n---\n".join(context_blocks)
            
            # 4. Invoke Grounded Chain with resilient fallback
            logger.info("Invoking grounded QA model...")
            try:
                answer = self.qa_chain.invoke({
                    "context": context_str,
                    "question": query
                })
            except Exception as llm_err:
                logger.warning(f"Ollama generation failed ({llm_err}). Generating grounded extractive summary.")
                # Extractive grounded answer directly from matching chunks
                extracted_lines = []
                for chunk in retrieved_chunks_info[:3]:
                    c_title = chunk.get("title", "Indexed Paper")
                    c_text = chunk.get("content", "").strip()
                    first_sent = c_text.split("\n")[0][:400]
                    extracted_lines.append(f"**From *{c_title}*:**\n> {first_sent}...")
                
                answer = (
                    f"Based on the indexed evidence in your research library:\n\n"
                    + "\n\n".join(extracted_lines)
                    + f"\n\n*(Evidence retrieved from {len(sources_list)} indexed paper(s))*"
                )
            
            # Check for standard fallback phrases to set status
            insufficient = False
            fallback_phrases = [
                "insufficient to answer",
                "insufficient evidence",
                "does not contain sufficient",
                "not mentioned in the provided",
                "not found in the retrieved"
            ]
            for phrase in fallback_phrases:
                if phrase in answer.lower():
                    insufficient = True
                    break
                    
            status = "insufficient_evidence" if insufficient else "success"

            
            response_payload = {
                "answer": answer,
                "sources": sources_list,
                "retrieved_chunks": retrieved_chunks_info,
                "status": status
            }

            # Save interaction to MongoDB
            try:
                self.mongo_store.save_chat_interaction({
                    "query": query,
                    "mode": mode,
                    "paper_ids": paper_ids,
                    "answer": answer,
                    "sources": sources_list,
                    "retrieved_chunks_count": len(retrieved_chunks_info),
                    "status": status
                })
            except Exception as e:
                logger.warning(f"Failed to log chat interaction to MongoDB: {e}")
            
            return response_payload
            
        except Exception as e:
            logger.error(f"Error in RAG retrieval: {e}")
            return {
                "answer": f"An error occurred while generating the answer: {str(e)}",
                "sources": [],
                "retrieved_chunks": [],
                "status": "error"
            }

