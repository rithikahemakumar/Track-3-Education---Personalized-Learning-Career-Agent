import os
import requests
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain.agents import create_agent
import gradio as gr

# STEP 1 — LLM setup
 
llm = ChatOllama(model="qwen3:0.6b", temperature=0.2)
 
_test = llm.invoke("Reply with the single word: ready")
print("[Step 1] LLM check:", _test.content.strip())
 

# STEP 2 — Document loading
 
PDF_PATH = "course_notes_dsa.pdf"
loader = PyPDFLoader(PDF_PATH)
raw_docs = loader.load()
print(f"[Step 2] Loaded {len(raw_docs)} page(s) from {PDF_PATH}")
 

# STEP 3 — Text splitting

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(raw_docs)
print(f"[Step 3] Split into {len(chunks)} chunks")
 
 
# STEP 4 — Embeddings + vector store

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
 
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_course_notes",
    collection_name="course_notes",
)
print("[Step 4] Embedded and persisted to Chroma at ./chroma_course_notes")
 

# STEP 5 — Retriever

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
 
for test_query in ["time complexity of binary search", "explain how dynamic programming works"]:
    hits = retriever.invoke(test_query)
    print(f"[Step 5] '{test_query}' -> top hit starts with: {hits[0].page_content[:80]!r}")
 
 

# STEP 6 — Custom tools
 
# (a) Course-material retrieval tool, wrapping the retriever above
@tool
def retrieve_course_material(query: str) -> str:
    """Retrieve relevant course notes/content for a student's question about
    a topic covered in class (concepts, definitions, complexity, algorithms).
    Use this for 'explain X' or 'what is X' style questions about course
    material. Do NOT use this for job or internship searches."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant course material found for this query."
    return "\n\n---\n\n".join(d.page_content for d in docs)
 
 
# (b) Job/internship search tool, matched to a topic keyword.
# Uses the Adzuna API (https://developer.adzuna.com/) — sign up for a free
# app_id / app_key and set them as environment variables below.
ADZUNA_APP_ID = os.environ.get("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY = os.environ.get("ADZUNA_APP_KEY", "")
 
@tool
def search_internships(topic: str) -> str:
    """Search for live internship/entry-level job listings matched to a
    given topic or skill keyword (e.g. 'data structures', 'python',
    'machine learning'). Use this for 'find me an internship/job about X'
    style questions. Do NOT use this for explaining course concepts."""
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        return ("Job search is not configured: set ADZUNA_APP_ID and "
                "ADZUNA_APP_KEY environment variables to enable live search.")
 
    url = "https://api.adzuna.com/v1/api/jobs/in/search/1"
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "what": topic,
        "results_per_page": 5,
        "content-type": "application/json",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"Job search failed: {e}"
 
    results = data.get("results", [])
    if not results:
        return f"No internship/job listings found for '{topic}'."
 
    lines = []
    for job in results:
        title = job.get("title", "Untitled")
        company = job.get("company", {}).get("display_name", "Unknown company")
        location = job.get("location", {}).get("display_name", "Unknown location")
        url_ = job.get("redirect_url", "")
        lines.append(f"- {title} at {company} ({location}) — {url_}")
    return "\n".join(lines)
 
 
tools = [retrieve_course_material, search_internships]
print("[Step 6] Built tools:", [t.name for t in tools])
 
 
# STEP 7 — Tool calling (each tool already decorated + documented above;
# quick standalone test before wiring into the agent)

print("[Step 7] Standalone tool test (retrieval):")
print(retrieve_course_material.invoke("what is a hash table"))
 
 
# STEP 8 — Agent
 
SYSTEM_PROMPT = """You are a helpful study companion for a Data Structures &
Algorithms course.
 
For every user message, decide:
1. Is this a course-content question (explain a concept, definition,
   complexity, algorithm)? -> call retrieve_course_material.
2. Is this an opportunity-search question (internship, job, "find me a
   role about X")? -> call search_internships.
3. If it is both, do both, in that order.
 
When you return job/internship results, briefly explain *why* each result
matches the topic the student has been studying, using the retrieved course
material as context. Never fabricate a job listing or a course fact that
wasn't returned by a tool. Keep answers concise and student-friendly.
"""
 
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)
 
print("[Step 8] Agent built.")
 
 
def ask_agent(user_message: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})
    return result["messages"][-1].content
 
 
# STEP 9 — Deployment (Gradio chat UI)
 
def chat_fn(message, history):
    return ask_agent(message)
 
 
demo = gr.ChatInterface(
    fn=chat_fn,
    title="Study Companion — DSA Course Agent",
    description=(
        "Ask about any Data Structures & Algorithms topic, or ask for "
        "internships/jobs related to what you're studying. "
        "This is a hackathon demo, not official career advice."
    ),
    examples=[
        "Explain how dynamic programming works",
        "What is the time complexity of binary search?",
        "Find me internships related to data structures",
    ],
)
 
if __name__ == "__main__":
    demo.launch()
 