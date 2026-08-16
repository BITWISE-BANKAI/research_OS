import re

def normalize_title(title: str) -> str:
    """Normalize paper title for fuzzy comparison."""
    if not title:
        return ""
    # Lowercase, strip punctuation and extra spaces
    title = title.lower().strip()
    title = re.sub(r'[^\w\s]', '', title)
    return " ".join(title.split())

def is_duplicate(new_paper: dict, existing_papers: list) -> bool:
    """
    Check if a paper is already present in the existing indexed papers list.
    Checks paper_id, then DOI, then arXiv ID, then normalized title.
    """
    new_pid = new_paper.get("paper_id")
    new_doi = new_paper.get("doi")
    new_arxiv = new_paper.get("arxiv_id")
    new_title_norm = normalize_title(new_paper.get("title", ""))

    for paper in existing_papers:
        # 1. Check paper_id
        if new_pid and paper.get("paper_id") == new_pid:
            return True
            
        # 2. Check DOI
        p_doi = paper.get("doi")
        if new_doi and p_doi and str(new_doi).lower().strip() == str(p_doi).lower().strip():
            return True
            
        # 3. Check arXiv ID
        p_arxiv = paper.get("arxiv_id")
        if new_arxiv and p_arxiv and str(new_arxiv).lower().strip() == str(p_arxiv).lower().strip():
            return True
            
        # 4. Check normalized title
        if new_title_norm and normalize_title(paper.get("title", "")) == new_title_norm:
            return True
            
    return False
