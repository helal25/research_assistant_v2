# Autonomous Multi-Agent Scientific Research Engine

An end-to-end multi-agent literature research engine built with **LangGraph**, **LangChain**, and **OpenAI**. The system decomposes high-level technical inquiries into targeted academic and web queries, executes parallel retrieval across **arXiv** and **Tavily**, runs full PDF text extraction and semantic vector retrieval inside an isolated subgraph, synchronizes parallel streams through a barrier **Collector** node, and synthesizes publication-grade research reports.

---

## Architecture Flow

The engine applies a structured **Map-Reduce** architectural pattern:

```text
                      [ User Research Query ]
                                 │
                                 ▼
                         [ subquery Node ]
                    (Structured Decomposition)
                                 │
            ┌────────────────────┴────────────────────┐
            │ Dynamic Send() Tasks                    │ Dynamic Send() Tasks
            ▼                                         ▼
   [ call_arxiv_subgraph ]                     [ web_search ]
   (Isolated RAG Subgraph)                 (Live Web Search via Tavily)
            │                                         │
            └────────────────────┬────────────────────┘
                                 ▼
                         [ collector Node ]  <-- (Barrier Synchronization)
                                 │
                                 ▼
                          [ drafter Node ]
                     (Initial Research Draft)
                                 │
                                 ▼
                        [ finalizer Node ]
                 (Publication-Grade Polish & Bib)
                                 │
                                 ▼
                              [ END ]
```

### Isolated arXiv RAG Subgraph

Each academic query dispatched via `Send` executes inside an independent child graph:

```text
[ START ] 
    │
    ▼
[ arxiv_search ]   --> Rate-limited query (3.5s lock) + Tavily fallback
    │
    ▼
[ load_pdfs ]      --> Fetches full PDF text (pypdf) or HTML abstracts (bs4)
    │
    ▼
[ splitters ]      --> RecursiveCharacterTextSplitter (chunk size: 200, overlap: 50)
    │
    ▼
[ retriever ]      --> InMemoryVectorStore (OpenAIEmbeddings), k=2 top matches
    │
    ▼
 [ END ]
```

---

## Core Components

| Component | Responsibility |
| :--- | :--- |
| **`subquery`** | Deconstructs user prompts into targeted `arxiv_queries` and `tavily_queries` using `ChatOpenAI.with_structured_output(QueryPlan)`. |
| **`route_to_search`** | Uses LangGraph's dynamic `Send` API to map subqueries concurrently to independent search targets. |
| **`call_arxiv_subgraph`** | Executes the RAG child graph, handling arXiv rate-limiting locks (`threading.Lock`), Tavily fallback on HTTP 406/429 errors, PDF parsing, chunking, and vector retrieval. |
| **`web_search`** | Retrieves real-world benchmarks, ecosystem news, and commercial tech updates via Tavily. |
| **`collector`** | Acts as a **barrier synchronization node** that gathers all parallel search returns into `SubState` reducers (`operator.add`) before synthesis begins. |
| **`drafter`** | Combines retrieved paper excerpts, abstracts, and web snippets into an initial structured technical draft. |
| **`finalizer`** | Refines prose clarity, standardizes headings, and compiles clean reference links into a final publication-grade brief. |

---

## State Architecture

The system uses typed schemas to enforce data isolation between global execution and worker nodes:

- **`SubInput`:** Minimal entry schema (`query: Union[HumanMessage, str]`).
- **`SubState`:** Root state with `Annotated[list[str], operator.add]` reducers for non-destructive parallel result aggregation across `arxiv_results`, `tavily_results`, and `url_list`.
- **`RAGState` / `RAGOutput`:** Local state schema dedicated to PDF extraction, document splitting, and vector retrieval.
- **`SearchState`:** Target schema for web search workers.

---

## Setup & Installation

### 1. Prerequisites
- Python 3.10+
- OpenAI API Key
- Tavily API Key

### 2. Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/research-assistant.git
cd research-assistant

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Recommended `requirements.txt`:
```text
langgraph
langchain
langchain-core
langchain-openai
langchain-tavily
arxiv
pypdf
beautifulsoup4
requests
python-dotenv
tiktoken
pydantic
```

### 3. Environment Variables
Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

---

## Usage

Run the graph directly from the command line:

```bash
python research_assistant.py
```

To invoke programmatically with thread-based persistence:

```python
from research_assistant import app

config = {"configurable": {"thread_id": "research-session-1"}}
result = app.invoke({"query": "neutron moderation"}, config=config)

print("=== FINAL REPORT ===")
print(result.get("final_report"))
```

## License

This project is licensed under the MIT License.