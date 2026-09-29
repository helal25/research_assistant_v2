import operator
import arxiv
from typing import Annotated, List, Optional,TypedDict, Literal,Any, Union
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph,END, START
from pydantic import BaseModel, Field
from langgraph.types import Send
from pprint import pprint
import bs4
import requests
import io
import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from functools import lru_cache
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
import threading
import time
import logging
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

_arxiv_lock = threading.Lock()
_last_arxiv_request_time = 0.0
ARXIV_DELAY_SECONDS = 3.5



load_dotenv()
system_msg_query="""You are an expert Research Planning Agent responsible for breaking down complex research topics into targeted, high-yield retrieval tasks.

### Objective
Analyze the user's input and produce a structured execution plan. Deconstruct the inquiry into independent, non-overlapping sub-queries optimized for academic literature (arXiv) or general web search (Tavily/Google).

### Guidelines
1. **Deconstruct, Don't Duplicate:**
   - Split broad or comparative inquiries into distinct analytical dimensions (e.g., mechanisms, benchmarks, trade-offs, recent developments).
   - Generate between 2 to 4 focused sub-queries. Avoid redundancy across queries.

2. **Optimize for the Target Engine:**
   - **arXiv (`arxiv`):** Use for peer-reviewed research, foundational mathematics, theoretical frameworks, physics/engineering principles, and formal algorithms. Strip conversational filler and use precise scientific terminology or keyword combinations.
   - **Web Search (`web_search`):** Use for real-time news, commercial applications, company announcements, open-source repository updates, or industry benchmark reports.

3. **Keyword Precision:**
   - Formulate search strings using high-information density terms.
   - Omit punctuation, conversational framing ("tell me about", "what is"), and stop words.

4. **Output Integrity:**
   - Output must strictly conform to the provided structured schema. Do not generate explanations, preambles, or conversational commentary.
"""

llm_subquery=ChatOpenAI(model='gpt-4o-mini')
llm_drafting=ChatOpenAI(model='gpt-4o')


class SubInput(BaseModel):
    query: Union[HumanMessage, str] = Field(description="User prompt/inquiry as a message or string")


class SubState(BaseModel):
    query: Union[HumanMessage, str] = Field(description="User prompt/inquiry as a message or string")    
    arxiv_queries: Annotated[list[str], operator.add] = Field(default_factory=list,description="Subqueries destined for the arxiv search node.")
    tavily_results: Annotated[list[str], operator.add] = Field(default_factory=list)
    tavily_queries: Annotated[list[str], operator.add] = Field(default_factory=list, description="Subqueries destined for the tavily search node.")
    arxiv_results: Annotated[list[str], operator.add] = Field(default_factory=list, description="Retrieved arXiv RAG results.")
    url_list: Annotated[list[str], operator.add] = Field(default_factory=list, description="List of retrieved paper URLs.")
    draft: str = Field(default="", description="Initial synthesized research draft")
    final_report: str = Field(default="", description="Polished final synthesis")
class QueryPlan(BaseModel):
    arxiv_queries: list[str] = Field(description="List of academic subqueries")
    tavily_queries: list[str] = Field(description="List of web search subqueries")
class SearchState(BaseModel):
    subquery: str                              
    target: Literal["search_item"]
class RAGState(BaseModel):
    subquery: str
    target: Literal["arxiv_item"] = "arxiv_item"
    url: str = ""
    url_list: Annotated[list[str],operator.add] = Field(default_factory=list)
    doc_splits: Any = None
    docs_list: list = Field(default_factory=list)
    docs: list = Field(default_factory=list)
    results: Annotated[list, operator.add] = Field(default_factory=list)
    arxiv_results: Annotated[list, operator.add] = Field(default_factory=list)
class RAGOutput(BaseModel):
    results: Annotated[list[str], operator.add] = Field(default_factory=list)
    arxiv_results: Annotated[list[str], operator.add] = Field(default_factory=list)
    url_list: Annotated[list[str], operator.add] = Field(default_factory=list)

def get_query_text(query: Union[BaseMessage, str]) -> str:
    """Safely extracts a string whether input is a LangChain message or raw str."""
    if isinstance(query, BaseMessage):
        return str(query.content)
    return str(query)

def sub_query(state: SubState) -> dict:   
    structured_llm = llm_subquery.with_structured_output(QueryPlan)
    human_msg = state.query if isinstance(state.query, HumanMessage) else HumanMessage(content=get_query_text(state.query))
    plan: QueryPlan = structured_llm.invoke([SystemMessage(content=system_msg_query),human_msg])     
    return {
        "arxiv_queries": plan.arxiv_queries,
        "tavily_queries": plan.tavily_queries}

def route_to_search(state:SubState):
    tasks = []
    for q in state.arxiv_queries:
        tasks.append(Send("arxiv", RAGState(subquery=q, target="arxiv_item")))
    for q in state.tavily_queries:
        tasks.append(Send("web_search", SearchState(subquery=q, target="search_item")))
    return tasks

def web_search(state: SearchState)-> dict:
    """Execute live web search for real-time information, commercial updates, and industry applications.

    Use this tool when the query requires:
    - Real-time or recent real-world events, announcements, and news.
    - Commercial technology updates, production benchmarks, and deployment case studies.
    - Open-source software documentation, GitHub repository statuses, and release notes.
    - Industry reports, startup initiatives, or multi-source web syntheses that are 
      not covered by academic literature or arXiv preprints.

    Do NOT use this tool for formal mathematical proofs, academic physics formulas, 
    or peer-reviewed scientific literature where arXiv is the authoritative source.

    Args:
        query: Concise, keyword-optimized search string without conversational filler.
        max_results: Number of search results to return (default: 5).

    Returns:
        Structured string containing page titles, source URLs, and relevant content snippets.
    """
    tavily = TavilySearch(max_results=3)
    results_from_search = tavily.invoke(state.subquery) 
    return {"tavily_results": [str(results_from_search)]}

def call_arxiv_subgraph(state: RAGState) -> dict:
    output = arxiv_subgraph.invoke(state)
    return {
        "results": output.get("results", []),
        "arxiv_results": output.get("arxiv_results", []),
        "url_list": output.get("url_list", []),
    }

def arxiv_search(state: RAGState) -> dict:
    """Queries arXiv, saves direct PDF links, and converts abstracts to Documents.
    
    Includes global rate-limiting lock across threads to respect arXiv's policy
    (<= 1 request per 3 seconds) and automatic fallback to Tavily academic search
    if arXiv returns HTTP 406 / 429 or is temporarily unreachable.
    """
    global _last_arxiv_request_time
    clean_query = state.subquery.replace('"', '').strip()
    urls = []
    docs = []
    with _arxiv_lock:
        now = time.time()
        elapsed = now - _last_arxiv_request_time
        if elapsed < ARXIV_DELAY_SECONDS:
            time.sleep(ARXIV_DELAY_SECONDS - elapsed)
        _last_arxiv_request_time = time.time()

        try:
            client = arxiv.Client(
                page_size=3,
                delay_seconds=3.5,
                num_retries=2
            )
            search = arxiv.Search(
                query=clean_query,
                max_results=3,
                sort_by=arxiv.SortCriterion.Relevance
            )
            for paper in client.results(search):
                urls.append(paper.pdf_url)
                content = f"Title: {paper.title}\nPDF Link: {paper.pdf_url}\nSummary: {paper.summary}"
                docs.append(Document(
                    page_content=content,
                    metadata={"source": paper.pdf_url, "title": paper.title}
                ))
            _last_arxiv_request_time = time.time()
        except Exception as e:
            logger.warning(f"Direct arXiv search failed for '{clean_query}': {e}. Falling back to Tavily for arXiv papers.")
            try:
                tavily = TavilySearch(max_results=3)
                tav_res = tavily.invoke(f"{clean_query} site:arxiv.org")
                items = tav_res.get("results", []) if isinstance(tav_res, dict) else []
                for item in items:
                    raw_url = item.get("url", "")
                    title = item.get("title", "Unknown Title")
                    summary = item.get("content", "")
                    pdf_link = raw_url.replace("/abs/", "/pdf/") if "/abs/" in raw_url else raw_url
                    urls.append(pdf_link)
                    content = f"Title: {title}\nPDF Link: {pdf_link}\nSummary: {summary}"
                    docs.append(Document(
                        page_content=content,
                        metadata={"source": pdf_link, "title": title}
                    ))
            except Exception as fb_err:
                logger.error(f"Fallback search also failed for '{clean_query}': {fb_err}")

    return {"url_list": urls, "docs": docs}

def load_pdfs(state: RAGState) -> dict:
    """Downloads and extracts full text from arXiv PDFs or HTML abstracts in url_list."""
    loaded_docs = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    target_urls = state.url_list[:2] if state.url_list else []
    
    for url in target_urls:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "").lower()

            if "pdf" in content_type or url.endswith(".pdf") or "/pdf/" in url:
                reader = pypdf.PdfReader(io.BytesIO(resp.content))
                extracted_pages = []
                for i, page in enumerate(reader.pages[:5]):
                    page_text = page.extract_text()
                    if page_text:
                        extracted_pages.append(page_text.strip())
                
                text_content = "\n\n".join(extracted_pages)
                if text_content.strip():
                    loaded_docs.append(Document(
                        page_content=text_content,
                        metadata={"source": url, "type": "pdf"}
                    ))
            else:
                soup = bs4.BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                clean_text = soup.get_text(separator=" ", strip=True)
                if clean_text:
                    loaded_docs.append(Document(
                        page_content=clean_text,
                        metadata={"source": url, "type": "html"}
                    ))
        except Exception as e:
            logger.warning(f"Failed to download/parse PDF from {url}: {e}")
    if not loaded_docs and state.docs:
        loaded_docs = state.docs

    return {"docs": loaded_docs}

def splitters(state: RAGState) -> dict:
    """Splits documents into smaller chunks for dense vector embedding."""
    if not state.docs:
        return {"doc_splits": []}
        
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200,
        chunk_overlap=50,
    )
    splits = text_splitter.split_documents(state.docs)
    return {"doc_splits": splits}

def run_retriever(state: RAGState) -> dict:
    """Indexes chunks, retrieves relevant sections, and returns matches to 'results'."""
    if not state.doc_splits:
        return {"arxiv_results": [], "url_list": state.url_list}

    vectorstore = InMemoryVectorStore.from_documents(
        documents=state.doc_splits,
        embedding=OpenAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    matched_docs = retriever.invoke(state.subquery)
    
    extracted_snippets = [
        f"ArXiv Match [{doc.metadata.get('source', 'Unknown')}]:\n{doc.page_content}"
        for doc in matched_docs
    ]
    return {"arxiv_results": extracted_snippets, "url_list": state.url_list}

def collector(state: SubState) -> dict:
    """Barrier synchronization node that waits for all parallel Send tasks to complete."""
    return {}
def drafter(state: SubState) -> dict:
    DRAFTER_SYSTEM_PROMPT = """You are a Principal Scientific Research Synthesizer.
Your objective is to produce a detailed, technically rigorous preliminary draft addressing the user's research topic.

### Inputs Provided:
1. **Academic Findings (arXiv):** Mathematical proofs, algorithms, core theoretical mechanisms, and paper abstracts.
2. **Web & Industry Findings (Tavily):** Benchmark results, commercial implementations, practical deployments, and recent ecosystem updates.

### Directives:
- Synthesize both sources into a unified, coherent technical exposition.
- Clearly differentiate foundational/theoretical mechanisms from real-world applications.
- Cite specific arXiv IDs or paper URLs inline when presenting theoretical results (e.g., `[arXiv:...]`).
- Highlight open trade-offs, limitations, and future directions.
- Produce structured Markdown with clear section headings.
"""

    """Synthesizes academic findings and web search results into a structured initial draft."""
    unique_urls = list(dict.fromkeys(state.url_list))
    arxiv_context = (
        "\n\n---\n\n".join(state.arxiv_results)
        if state.arxiv_results
        else "No academic papers retrieved."
    )
    web_context = (
        "\n\n---\n\n".join(state.tavily_results)
        if state.tavily_results
        else "No external web sources retrieved."
    )
    query_str = get_query_text(state.query)
    user_prompt = f"""### Target Research Topic
    {query_str}

### Academic Literature Findings (arXiv)
{arxiv_context}

    ### Real-World & Industry Findings (Web Search)
    {web_context}

    ### Reference Paper Links
    {chr(10).join(f"- {u}" for u in unique_urls) if unique_urls else "None"}

    Please produce a comprehensive, structured research synthesis based on the findings above."""
    messages = [
        SystemMessage(content=DRAFTER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm_drafting.invoke(messages)
    return {"draft": response.content}

FINALIZER_SYSTEM_PROMPT = """You are an Executive Academic Editor.
Review and refine the provided research draft into a final publication-grade brief:
- Polish technical clarity, grammar, and flow.
- Ensure all claims are properly qualified and structured with logical headings.
- Verify that a clean 'References / Sources' list is appended at the bottom.
"""



def finalizer(state: SubState) -> dict:
    """Refines the preliminary research draft into a publication-grade final report."""
    query_str = get_query_text(state.query)
    user_prompt = f"""### Original Query
{query_str}

### Initial Draft
{state.draft}

Please produce the final publication-grade report based on the findings above."""

    messages = [
        SystemMessage(content=FINALIZER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm_drafting.invoke(messages)
    return {
        "final_report": response.content,
    }

rag_builder = StateGraph(RAGState, output_schema=RAGOutput)
rag_builder.add_node("arxiv_search", arxiv_search)
rag_builder.add_node("load_pdfs", load_pdfs)
rag_builder.add_node("splitters", splitters)
rag_builder.add_node("retriever", run_retriever)

rag_builder.add_edge(START, "arxiv_search")
rag_builder.add_edge("arxiv_search", "load_pdfs")
rag_builder.add_edge("load_pdfs", "splitters")
rag_builder.add_edge("splitters", "retriever")
rag_builder.add_edge("retriever", END)

arxiv_subgraph = rag_builder.compile()


graph = StateGraph(SubState, input_schema=SubInput)
graph.add_node("subquery", sub_query)
graph.add_node("web_search", web_search)
graph.add_node("arxiv", call_arxiv_subgraph)
graph.add_node("collector", collector)
graph.add_node("drafter", drafter)
graph.add_node("finalizer", finalizer)
graph.add_edge(START, "subquery")

graph.add_conditional_edges(
    "subquery",
    route_to_search,
    ["web_search", "arxiv"]
)

graph.add_edge("web_search", 'drafter')
graph.add_edge("arxiv", 'collector')
graph.add_edge("collector", "drafter")
graph.add_edge('drafter', 'finalizer')
graph.add_edge('finalizer', END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    test_query = "neutron moderation"
    print(f"Executing research graph for query: {test_query}")
    config = {"configurable": {"thread_id": "research-session-1"}}
    result = app.invoke({"query": test_query}, config=config)
    print("\n=== FINAL REPORT ===")
    print(result.get("final_report"))