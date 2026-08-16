def format_citation(metadata: dict) -> str:
    """
    Format citation strings from metadata details.
    Must not fabricate page numbers.
    """
    title = metadata.get("title", "Unknown Title")
    page = metadata.get("page")
    section = metadata.get("section", "General")
    source = metadata.get("source")
    
    citation_parts = [f"[{title}]"]
    
    if section and section != "General":
        citation_parts.append(f"Section: {section}")
        
    if page is not None and str(page).strip() != "" and str(page).lower() != "none":
        citation_parts.append(f"Page: {page}")
    elif source:
        citation_parts.append(f"Source: {source}")
    else:
        paper_id = metadata.get("paper_id")
        if paper_id:
            citation_parts.append(f"Paper ID: {paper_id}")
            
    return " | ".join(citation_parts)
