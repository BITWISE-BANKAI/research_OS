from dotenv import find_dotenv, load_dotenv
import os
import sys
from langchain_core.output_parsers import StrOutputParser
import requests
import time
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from rich import print
from langchain_groq import ChatGroq

if sys.platform.startswith('win') and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
env_path = find_dotenv()
load_dotenv(env_path)

api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

@tool
def search_papers(query: str) -> list:
    """
    Search Semantic Scholar for relevant research papers.
    """
    try:
        headers = {}
        if api_key and api_key != "SEMANTIC_SCHOLAR_API_KEY":
            headers["x-api-key"] = api_key
        else:
            # Add delay for free tier to avoid rate limiting
            time.sleep(2)
        
        params = {
            "query": query,
            "limit": 50,
            "fields": "paperId,title,abstract,authors,year,citationCount,influentialCitationCount,fieldsOfStudy,url"
        }
        
        response = requests.get(API_URL, params=params, headers=headers, timeout=10)
        
        if response.status_code == 429:
            time.sleep(5)
            response = requests.get(API_URL, params=params, headers=headers, timeout=10)
        
        response.raise_for_status()
        
        data = response.json()
        papers = data.get("data", [])
        
        results = []
        for p in papers:
            results.append({
                "title": p.get("title", "N/A"),
                "abstract": p.get("abstract", "N/A"),
                "authors": [a.get("name", "Unknown") for a in (p.get("authors") or []) if a and isinstance(a, dict)],
                "year": p.get("year"),
                "citations": p.get("citationCount", 0),
                "influential_citations": p.get("influentialCitationCount", 0),
                "fields_of_study": p.get("fieldsOfStudy") or [],
                "url": p.get("url", f"https://www.semanticscholar.org/paper/{p.get('paperId')}"),
            })
        
        return results
    except Exception as e:
        print(f"❌ Error searching papers: {e}")
        return []


summary_prompt = ChatPromptTemplate.from_template(
"""You are ResearchX, a Lead Academic Analyst and Domain Intelligence Specialist. Your objective is to produce an in-depth, rigorously structured research briefing on the topic: **{query}**.

You have been given a payload of research papers relevant to this topic ({papers}), including Semantic Scholar metadata. Use this payload as your evidence base to build a comprehensive understanding of {query} as a whole — not just a list of individual paper summaries, but a synthesized picture of where the field stands, what's contested, and where it's heading.

---

### Phase 0: Topic Primer

Before analyzing individual papers, ground the reader in {query}:
* **What is {query}?** [A paragraph of plain-language explanation, assuming an informed but non-specialist reader]
* **Why does it matter?** [A paragraph on scientific, technical, or real-world significance]
* **Field snapshot:** [1 short paragraph summarizing the general shape of research on {query}, based on what's represented in {papers} — is this a mature field, an emerging one, dominated by a few approaches, fragmented across many?]

---

### Phase 1: Individual Paper Breakdown

For EACH paper in {papers}, provide the following structured analysis:

#### 1. Metadata & Citation Signal
* **Title & Venue:** [Paper Title] — *Published in [Journal/Conference, Year]*
* **Semantic Scholar Signal:**
  * Total Citations: [Citation Count] | Influential Citations: [Influential Citation Count]
  * Core Fields: [Fields of Study]
  * **Impact Assessment:** [1 paragraph evaluating its reach—e.g., "Foundational Benchmark," "Rapidly Growing Method," or "Specialized Domain Study" based on citation velocity and influential citations].

#### 2. Strategic Overview
* **TLDR / Core Premise:** [4-5 sentences capturing the high-level elevator pitch].
* **Research Problem & Gap:** What specific bottleneck, theoretical flaw, or unaddressed challenge does this work target within {query}? Why did previous approaches fail?

#### 3. Technical Blueprint
* **Main Contribution:** The primary novel artifact, framework, algorithm, or theoretical proof introduced.
* **Methodology & Setup:** How was the hypothesis tested? Highlight key datasets, model architectures, baseline comparisons, or mathematical frameworks.

#### 4. Empirical Evidence & Vulnerabilities
* **Key Findings:** 2–3 key quantitative metrics, benchmarks, or qualitative outcomes validating the core claims.
* **Limitations & Blind Spots:** Structural weaknesses, dataset constraints, unstated assumptions, or high computational costs (noting both author-acknowledged limitations and methodological trade-offs).

---

### Phase 2: Cross-Paper Synthesis

1. **The Common Thread:** What shared objective, methodology, or foundational theory unites these papers on {query}?
2. **Methodological Divergence:** Where do these approaches disagree or take fundamentally different architectural/theoretical paths?
3. **The Next Frontier:** Based on the collective limitations identified, what is the single most critical open research question that future work on {query} should solve?

---

### Phase 3: Topic-Level Deep Dive

1. **Historical Trajectory:** Based on the publication years and approaches represented here, how has thinking on {query} evolved?
2. **Consensus vs. Controversy:** What do these papers broadly agree on regarding {query}? Where is there active disagreement or unresolved tension?
3. **Practical Implications:** What would a practitioner, engineer, or policymaker need to understand about {query} based on this evidence base?
4. **Closing Synthesis:** In 6-7 sentences, characterize the overall state of {query} as a field right now.
5. **Citation Landscape:** Provide a brief overview of the citation landscape for {query}, highlighting any seminal works, highly cited papers, or influential authors that have shaped the discourse(atleast 1 citation count from each).
---

**Topic:** {query}
**Input Payload:** {papers}
"""
)


llm = ChatOllama(model="qwen3", temperature=0)
'''class PaperSummary(BaseModel):
    title: str
    research_problem: str
    contribution: str
    methodology: str
    findings: str
    limitations: str
    citation_count: int

structured_llm = llm.with_structured_output(PaperSummary)'''

summary_chain = (summary_prompt | llm | StrOutputParser())

from pydantic import BaseModel, Field, field_validator
from typing import List

evaluator_prompt = ChatPromptTemplate.from_template("""
You are a strict research-summary evaluator.

Evaluate the summary against the research papers.

You MUST follow this EXACT output format.

Do not use Markdown tables.
Do not add explanations before or after the format.

SCORE: <integer from 0 to 10>

UNSUPPORTED_CLAIMS:
- claim 1
- claim 2

MISSING_LIMITATIONS:
- limitation 1
- limitation 2

MISSING_QUANTITATIVE_EVIDENCE:
- evidence 1
- evidence 2

REDUNDANCY_ISSUES:
- issue 1
- issue 2

REVISION_INSTRUCTIONS:
- instruction 1
- instruction 2

If there are no issues in a category, write:

NONE

SUMMARY:
{summary}
""")

evaluator_llm = ChatGroq(
    model="openai/gpt-oss-20b",  
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


evaluator_chain = evaluator_prompt | evaluator_llm | StrOutputParser()
import re
def parse_evaluation(text: str):

    score_match = re.search(
        r"SCORE\s*:\s*(\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE
    )

    if score_match:
        score = float(score_match.group(1))
    else:
        score = None

    def get_section(section_name, next_sections):

        pattern = rf"{section_name}\s*:\s*(.*?)(?=\n(?:{'|'.join(next_sections)})\s*:|\Z)"

        match = re.search(
            pattern,
            text,
            re.IGNORECASE | re.DOTALL
        )

        if not match:
            return []

        content = match.group(1).strip()

        if content.upper() == "NONE":
            return []

        return [
            line.strip()
            .lstrip("-•")
            .strip()
            for line in content.splitlines()
            if line.strip()
        ]


    sections = [
        "UNSUPPORTED_CLAIMS",
        "MISSING_LIMITATIONS",
        "MISSING_QUANTITATIVE_EVIDENCE",
        "REDUNDANCY_ISSUES",
        "REVISION_INSTRUCTIONS"
    ]


    return {
        "score": score,

        "unsupported_claims": get_section(
            "UNSUPPORTED_CLAIMS",
            sections[1:]
        ),

        "missing_limitations": get_section(
            "MISSING_LIMITATIONS",
            sections[2:]
        ),

        "missing_quantitative_evidence": get_section(
            "MISSING_QUANTITATIVE_EVIDENCE",
            sections[3:]
        ),

        "redundancy_issues": get_section(
            "REDUNDANCY_ISSUES",
            sections[4:]
        ),

        "revision_instructions": get_section(
            "REVISION_INSTRUCTIONS",
            []
        )
    }


regeneration_prompt = ChatPromptTemplate.from_template("""
You are ResearchX, a Lead Academic Analyst and Domain Intelligence Specialist. You previously
wrote a research briefing on {query}. A peer-review pass has identified specific issues with it.

Revise your summary to fix EVERY issue in the reviewer feedback below, while preserving the parts
that were already accurate and well-supported. Do not just patch — produce a complete, improved
version of the full briefing.

--- YOUR PREVIOUS SUMMARY ---
{previous_summary}

--- REVIEWER FEEDBACK (address ALL of this) ---
{revision_instructions}

Rules:
1. Every claim must be traceable to something in the source payload above.
2. Add specific numbers/metrics/citations wherever the reviewer flagged missing evidence.
3. Where papers disagree or take different approaches, say so explicitly — don't smooth it into
   false consensus.
4. Do not introduce new unsupported claims while fixing the flagged ones.

**Topic:** {query}
""")

regen_llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)
regeneration_chain = regeneration_prompt | regen_llm | StrOutputParser()


def iterative_summarize(query: str, papers: list, max_iterations: int = 3, score_threshold: int = 9):
    summary_text = summary_chain.invoke({"papers": papers, "query": query})  # initial draft
    summary_text = str(summary_text)

    for i in range(max_iterations):
        print(f"\n🔍 Evaluating draft {i + 1}...")
        evaluation_text = evaluator_chain.invoke({
            "query": query,
            "summary": summary_text,
        })
        evaluation = parse_evaluation(
            evaluation_text
        )

        score = evaluation["score"]
        if score==0.0 or score is None:
            print("⚠️  Evaluation failed to produce a score. Stopping loop.")
            break

        print(f"   Score: {score}/10")
        if score >= score_threshold:
            print("✅ Quality threshold met. Stopping loop.")
            break

        n_issues = sum(
            len(evaluation[key])
            for key in [
                "unsupported_claims",
                "missing_limitations",
                "missing_quantitative_evidence",
                "redundancy_issues"
            ]
        )
        print(f"   Issues found: {n_issues} — regenerating with feedback...")

        summary_text = regeneration_chain.invoke({
            "query": query,
            "papers": papers,
            "previous_summary": summary_text,
            "evaluation": evaluation_text,
            "revision_instructions":evaluation["revision_instructions"]
        })
    else:
        print(f"⚠️  Reached max iterations ({max_iterations}) without hitting score threshold.")

    return summary_text


if __name__ == "__main__":
    print("=" * 70)
    print("RESEARCH PAPER ANALYSIS PIPELINE")
    print("=" * 70)
    
    query = "why are LLMs hallucinating?"
    papers = search_papers.invoke({"query": query})
    
    if not papers:
        print("\n❌ Error: No research papers were found for the query. Exiting pipeline.")
        sys.exit(1)
    
    summary = iterative_summarize(query, papers, max_iterations=3, score_threshold=8)
    
    if summary:
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(summary)
    else:
        print("\n⚠️ No summary generated")