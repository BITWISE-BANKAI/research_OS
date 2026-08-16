import os
from typing import List, Optional
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag.chroma_store import ChromaStore
from rag.mongo_store import MongoStore
from utils.logging import setup_logger

logger = setup_logger("ComparisonTool")

COMPARISON_SYSTEM_PROMPT = """You are RESEARCHOS, a senior research scientist.
Your task is to compare the selected research papers based on the retrieved evidence below.

You must build a detailed, side-by-side comparison. Structure your response under these sections:
1. **Summary Table**: Present a markdown table comparing: Paper (Title/Year), Contribution, Methodology, Dataset/Baselines, Key Findings, and Core Limitations.
2. **Methodological Divergence**: Discuss how their approaches differ, their structural architectures, and why they chose these paths.
3. **Empirical Comparison & Trade-offs**: Contrast their performance, efficiency, datasets, and trade-offs.
4. **Research Gaps & Contradictions**: Identify if there are any conflicting findings, disagreements, or gaps in their collective findings.

RETRIEVED EVIDENCE PER PAPER:
{evidence_context}

Provide a rigorous, academic-grade comparative analysis. Ground everything in the evidence.
"""

COMPARISON_PROMPT = ChatPromptTemplate.from_template(COMPARISON_SYSTEM_PROMPT)

@tool
def compare_papers(paper_ids: List[str]) -> str:
    """
    Compares multiple research papers across methodology, datasets, findings, and limitations.
    Retrieves evidence from the database for each paper and synthesizes a structured comparison.
    Persists comparison reports to MongoDB.
    """
    try:
        if not paper_ids or len(paper_ids) < 2:
            return "Error: At least two paper IDs are required to perform a comparison."
            
        store = ChromaStore()
        mongo_store = MongoStore()
        evidence_context = []
        
        # 1. Fetch evidence for each paper separately to keep context clean and avoid cross-contamination
        for pid in paper_ids:
            meta = mongo_store.get_paper(pid) or store.get_paper_metadata(pid)
            if not meta:
                continue
                
            title = meta.get("title", "Unknown Title")
            year = meta.get("year", "N/A")
            
            # Fetch chunks on methodology/contributions
            meth_chunks = store.similarity_search("methodology method architecture contribution algorithm", paper_ids=[pid], k=3)
            # Fetch chunks on findings/results/limitations
            res_chunks = store.similarity_search("results findings metrics dataset limitations weaknesses", paper_ids=[pid], k=3)
            
            combined_chunks = meth_chunks + res_chunks
            # Deduplicate chunks in case some matched both queries
            seen_content = set()
            unique_chunks = []
            for doc in combined_chunks:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    unique_chunks.append(doc)
            
            paper_evidence = f"### PAPER: {title} ({year}) (ID: {pid})\n"
            for doc in unique_chunks:
                sec = doc.metadata.get("section", "General")
                pg = doc.metadata.get("page", "N/A")
                paper_evidence += f"- [Section: {sec} | Page: {pg}]: {doc.page_content}\n"
                
            evidence_context.append(paper_evidence)
            
        if not evidence_context:
            return "Error: Could not retrieve any evidence from ChromaDB for the requested paper IDs."
            
        # 2. Invoke local LLM to synthesize the comparison
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        chat_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:latest")
        
        llm = ChatOllama(
            base_url=base_url,
            model=chat_model,
            temperature=0
        )
        
        chain = COMPARISON_PROMPT | llm | StrOutputParser()
        
        logger.info(f"Synthesizing comparison for papers: {paper_ids}")
        response = chain.invoke({
            "evidence_context": "\n\n".join(evidence_context)
        })
        
        # Save comparison into MongoDB
        mongo_store.save_comparison({
            "paper_ids": paper_ids,
            "comparison": response,
            "evidence_context": "\n\n".join(evidence_context)
        })
        
        return response
        
    except Exception as e:
        logger.error(f"Error comparing papers {paper_ids}: {e}")
        return f"Error comparing papers: {str(e)}"

