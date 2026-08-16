"""agents/research_agent.py - LangChain agent that orchestrates research tools."""
import os
from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools.semantic_scholar import search_papers
from tools.paper_loader import load_paper
from tools.paper_indexer import index_paper
from tools.retrieval_tools import retrieve_evidence
from tools.comparison_tools import compare_papers
from utils.logging import setup_logger

logger = setup_logger("ResearchAgent")

AGENT_SYSTEM_PROMPT = """You are RESEARCHOS, an expert AI research assistant.

You have access to the following tools:
- search_papers: Search Semantic Scholar for relevant research papers by query.
- load_paper: Download and retrieve the full text of a paper by its Semantic Scholar paper_id.
- index_paper: Chunk, embed, and store a paper into the local knowledge database by its paper_id.
- retrieve_evidence: Retrieve relevant text evidence from indexed papers to answer a question.
- compare_papers: Compare multiple indexed papers across methodology, datasets, findings, and limitations.

AGENT RULES:
1. Never hallucinate paper IDs. Only use paper IDs that you received from search_papers or that the user explicitly provides.
2. Before retrieve_evidence or compare_papers, ensure papers are indexed using index_paper.
3. For user questions about specific papers, first search, then index the most relevant one, then retrieve evidence.
4. For comparison tasks, index all requested papers first, then call compare_papers.
5. For general research library questions, use retrieve_evidence without paper_id filtering.
6. Always provide a helpful, grounded, and concise response to the user.
"""


def get_research_agent() -> AgentExecutor:
    """
    Build and return a configured LangChain tool-calling agent backed by local Qwen3.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    chat_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:latest")

    llm = ChatOllama(base_url=base_url, model=chat_model, temperature=0)

    tools = [search_papers, load_paper, index_paper, retrieve_evidence, compare_papers]

    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENT_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=6,
        handle_parsing_errors=True,
    )
    logger.info("Research agent initialized successfully.")
    return executor
