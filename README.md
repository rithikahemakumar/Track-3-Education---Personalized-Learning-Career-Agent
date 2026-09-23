# Track-3-Education---Personalized-Learning-Career-Agent

A study companion agent that:
  1) answers questions grounded in course notes (RAG), and
  2) searches for internship/job listings matched to a topic keyword.
 
Place course_notes_dsa.pdf in the same folder before running.
 
Install:

  pip install langchain langchain-community langchain-ollama langchain-chroma
  
  pip install chromadb sentence-transformers pypdf gradio requests langgraph
 
Run Ollama first:

  ollama pull qwen3:0.6b
  
  ollama serve
 
Then:

  python education_agent.py
