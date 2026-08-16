import os
import time
import requests
from typing import List, Dict, Any
from langchain_core.tools import tool
from dotenv import load_dotenv
from utils.logging import setup_logger

load_dotenv()
logger = setup_logger("SemanticScholarTool")

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

@tool
def search_papers(query: str) -> List[Dict[str, Any]]:
    """
    Search Semantic Scholar for research papers matching a query.
    Returns a list of normalized paper metadata dictionaries.
    """
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    
    headers = {}
    if api_key and api_key != "your_semantic_scholar_api_key_here":
        headers["x-api-key"] = api_key
    
    params = {
        "query": query,
        "limit": 10,  # Grab top 10 relevant papers
        "fields": "paperId,title,abstract,authors,year,citationCount,influentialCitationCount,fieldsOfStudy,url,externalIds,openAccessPdf"
    }
    
    max_retries = 3
    backoff = 2.0
    response = None
    
    # If no api key, introduce a delay to avoid rate limiting
    if not api_key:
        time.sleep(2.0)
        
    for attempt in range(max_retries):
        try:
            response = requests.get(API_URL, params=params, headers=headers, timeout=10)
            if response.status_code == 429:
                logger.warning(f"429 Rate Limit hit. Retrying in {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2
                continue
            response.raise_for_status()
            break
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Semantic Scholar API request failed: {e}")
                return []
            time.sleep(backoff)
            backoff *= 2

    if response is None or response.status_code != 200:
        return []

    try:
        data = response.json()
        raw_papers = data.get("data", [])
        normalized_papers = []
        
        for p in raw_papers:
            if not p.get("paperId"):
                continue
                
            # Extract authors list
            authors_list = []
            for author in p.get("authors", []) or []:
                if author and isinstance(author, dict) and author.get("name"):
                    authors_list.append(author["name"])
            
            # Extract external IDs (arxiv, doi)
            ext_ids = p.get("externalIds") or {}
            doi = ext_ids.get("DOI")
            arxiv_id = ext_ids.get("ArXiv")
            
            # Extract open access URL
            oa_pdf = p.get("openAccessPdf") or {}
            open_access_url = oa_pdf.get("url")
            
            normalized_papers.append({
                "paper_id": p.get("paperId", "N/A"),
                "title": p.get("title", "N/A"),
                "abstract": p.get("abstract", "N/A"),
                "authors": authors_list,
                "year": p.get("year"),
                "citation_count": p.get("citationCount", 0),
                "influential_citation_count": p.get("influentialCitationCount", 0),
                "fields_of_study": p.get("fieldsOfStudy") or [],
                "url": p.get("url", f"https://www.semanticscholar.org/paper/{p.get('paperId')}"),
                "doi": doi,
                "arxiv_id": arxiv_id,
                "open_access_url": open_access_url
            })
            
        logger.info(f"Successfully found and normalized {len(normalized_papers)} papers for query: '{query}'")
        return normalized_papers
    except Exception as e:
        logger.error(f"Error parsing search results: {e}")
        return []
