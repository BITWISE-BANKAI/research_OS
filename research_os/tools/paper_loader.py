import os
import time
import requests
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from utils.logging import setup_logger

load_dotenv()
logger = setup_logger("PaperLoader")

# Simple in-memory metadata cache to reduce repeated API calls during a session
_METADATA_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}

def fetch_paper_details(paper_id: str) -> Optional[Dict[str, Any]]:
    """Fetch metadata details for a specific Semantic Scholar paper ID.

    Tries a few common identifier formats: as-provided, and if the value looks like an arXiv id
    (e.g. 2103.00020) it will also try the "ARXIV:<id>" form. Implements simple retry/backoff on 429
    responses and caches successful results in memory for the lifetime of the process.
    """
    # Return cached value if available
    if paper_id in _METADATA_CACHE:
        logger.info(f"Using cached metadata for {paper_id}")
        return _METADATA_CACHE[paper_id]

    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    headers = {}
    if api_key and api_key != "your_semantic_scholar_api_key_here":
        headers["x-api-key"] = api_key

    params = {
        "fields": "paperId,title,abstract,authors,year,citationCount,influentialCitationCount,fieldsOfStudy,url,externalIds,openAccessPdf"
    }

    # Prepare candidate identifier forms to try
    candidates = [paper_id]
    pid_up = paper_id.upper()
    if not pid_up.startswith("ARXIV:"):
        if ("." in paper_id and paper_id.replace('.', '').isdigit()) or paper_id.lower().startswith("arxiv"):
            candidates.append(f"ARXIV:{paper_id}")

    # Retry/backoff parameters
    max_retries = 3
    base_delay = 2  # seconds (initial backoff)

    for candidate in candidates:
        url = f"https://api.semanticscholar.org/graph/v1/paper/{candidate}"

        attempt = 0
        delay = base_delay
        while attempt <= max_retries:
            attempt += 1
            try:
                ts = datetime.utcnow().isoformat() + 'Z'
                logger.info(f"Attempting to fetch metadata for {candidate} at {ts} (attempt {attempt})")
                # Small spacing before the request to avoid bursts
                time.sleep(1)
                response = requests.get(url, headers=headers, params=params, timeout=15)

                if response.status_code == 200:
                    p = response.json()
                    authors_list = [author.get("name") for author in p.get("authors", []) if author and isinstance(author, dict) and author.get("name")]
                    ext_ids = p.get("externalIds") or {}
                    doi = ext_ids.get("DOI")
                    arxiv_id = ext_ids.get("ArXiv")

                    oa_pdf = p.get("openAccessPdf") or {}
                    open_access_url = oa_pdf.get("url")

                    result = {
                        "paper_id": p.get("paperId"),
                        "title": p.get("title", "N/A"),
                        "abstract": p.get("abstract", "N/A"),
                        "authors": authors_list,
                        "year": p.get("year"),
                        "citation_count": p.get("citationCount", 0),
                        "influential_citation_count": p.get("influentialCitationCount", 0),
                        "fields_of_study": p.get("fieldsOfStudy") or [],
                        "url": p.get("url"),
                        "doi": doi,
                        "arxiv_id": arxiv_id,
                        "open_access_url": open_access_url
                    }
                    # Cache and return
                    _METADATA_CACHE[paper_id] = result
                    return result

                elif response.status_code == 429:
                    # Rate limited — inspect Retry-After header if present
                    ra = response.headers.get("Retry-After")
                    if ra:
                        try:
                            ra_delay = int(ra)
                            logger.warning(f"Rate limited for {candidate}, server requested Retry-After={ra_delay}s")
                            time.sleep(ra_delay)
                        except ValueError:
                            logger.warning(f"Rate limited for {candidate}, Retry-After header present but not integer: {ra}")
                            time.sleep(delay)
                    else:
                        logger.warning(f"Rate limited for {candidate}, backing off for {delay}s")
                        time.sleep(delay)
                        delay = min(60, delay * 2)

                    # Increment attempt and retry
                    attempt += 1
                    continue

                else:
                    logger.warning(f"Semantic Scholar returned status {response.status_code} for identifier {candidate}. Response body: {response.text[:500]}")
                    break

            except Exception as e:
                logger.error(f"Error fetching paper details for {candidate} on attempt {attempt}: {e}")
                # small backoff before retrying
                time.sleep(delay)
                delay = min(60, delay * 2)
                continue

    # Cache negative result to avoid repeated failing calls during session
    _METADATA_CACHE[paper_id] = None
    return None

def download_file(url: str, dest_path: str) -> bool:
    """Download a file from a URL to a local destination."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"Failed to download PDF from {url}: {e}")
        return False

@tool
def load_paper(paper_id: str) -> List[dict]:
    """
    Downloads and loads the full text of a paper given its Semantic Scholar paper_id.
    Returns the loaded text pages as a list of dictionaries containing page_content and metadata.
    """
    # 1. Fetch metadata to get download URL
    paper = fetch_paper_details(paper_id)
    if not paper:
        return [{"error": f"Failed to retrieve metadata for paper ID: {paper_id}"}]
        
    # 2. Determine download URL (arXiv has highest priority, then open_access_url)
    pdf_url = None
    if paper.get("arxiv_id"):
        pdf_url = f"https://arxiv.org/pdf/{paper['arxiv_id']}.pdf"
        logger.info(f"Using arXiv PDF link: {pdf_url}")
    elif paper.get("open_access_url"):
        pdf_url = paper["open_access_url"]
        logger.info(f"Using Open Access PDF link: {pdf_url}")
        
    if not pdf_url:
        return [{"error": f"No full-text PDF URL available for paper: {paper['title']}"}]
        
    # 3. Create temp file and download
    temp_dir = tempfile.gettempdir()
    temp_pdf_path = os.path.join(temp_dir, f"{paper_id}.pdf")
    
    logger.info(f"Downloading PDF for '{paper['title']}'...")
    success = download_file(pdf_url, temp_pdf_path)
    
    if not success:
        # Retry with open_access_url if arXiv failed
        if paper.get("arxiv_id") and paper.get("open_access_url"):
            logger.info("arXiv download failed, retrying with Open Access URL...")
            success = download_file(paper["open_access_url"], temp_pdf_path)
            
    if not success:
        return [{"error": f"Failed to download full text PDF for '{paper['title']}' from {pdf_url}"}]
        
    # 4. Load PDF with PyPDFLoader
    try:
        loader = PyPDFLoader(temp_pdf_path)
        pages = loader.load()
        
        # 5. Enrich page document metadata with paper metadata
        results = []
        for i, page in enumerate(pages):
            enriched_metadata = {
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "authors": paper["authors"],
                "year": paper["year"],
                "url": paper["url"],
                "doi": paper["doi"],
                "arxiv_id": paper["arxiv_id"],
                "open_access_url": paper["open_access_url"],
                "source": pdf_url,
                "page": i
            }
            results.append({
                "page_content": page.page_content,
                "metadata": enriched_metadata
            })
            
        logger.info(f"Successfully loaded {len(results)} pages for '{paper['title']}'")
        
        # Clean up temp file
        try:
            os.remove(temp_pdf_path)
        except Exception:
            pass
            
        return results
    except Exception as e:
        logger.error(f"Error reading PDF for '{paper['title']}': {e}")
        return [{"error": f"Failed to parse PDF file contents: {str(e)}"}]
