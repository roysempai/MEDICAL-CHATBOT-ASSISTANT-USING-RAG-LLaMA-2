**MEDICAL CHATBOT ASSISTANT USING RAG & LLaMA 2**

Developed an AI-powered medical assistant chatbot leveraging Retrieval-Augmented Generation (RAG) architecture and LLaMA 2 language model to provide accurate, context-aware health information. Built a comprehensive system integrating natural language processing, vector databases, and web frameworks for intelligent document retrieval and response generation.

**Key Technical Components:**
- Implemented document ingestion pipeline using LangChain to process medical PDF documents, splitting content into optimized chunks (500 characters with 50-character overlap) for efficient retrieval
- Designed FAISS vector database using HuggingFace sentence-transformers (all-MiniLM-L6-v2) for semantic search and similarity matching across medical knowledge base
- Integrated LLaMA 2 7B quantized model (GGML Q4_0) with custom prompt engineering to ensure factual, hallucination-free responses
- Developed dual interfaces: Flask REST API for backend integration and Chainlit conversational UI for interactive user experience
- Engineered RetrievalQA chain with k=2 retrieval strategy, optimizing context relevance while maintaining response accuracy
- Implemented asynchronous callback handlers for real-time streaming responses and source attribution

**Technologies:** Python, LangChain, LLaMA 2, FAISS, HuggingFace Transformers, Flask, Chainlit, PyPDF, Vector Embeddings, NLP

**Impact:** Created an intelligent healthcare information system capable of answering medical queries with cited sources, reducing information retrieval time and improving accessibility to medical knowledge.
