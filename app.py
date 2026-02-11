from flask import Flask, request, jsonify
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

# Paths
DB_FAISS_PATH = "vectorstores/db_faiss"

# Custom Prompt Template
custom_prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't have an answer. Don't try to make up an answer.

Context:
{context}

Question:
{question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Set up custom prompt
def set_custom_prompt():
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

# Load model
def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        max_new_tokens=256,
        temperature=0.5
    )
    return llm

# Create retrieval QA chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# Load QA bot components
def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

# Initialize Flask
app = Flask(__name__)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json.get('message')
    if user_message:
        chain = qa_bot()
        response = chain.invoke(user_message)
        return jsonify({'response': response['result']})
        
    return jsonify({'error': 'No message provided'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
