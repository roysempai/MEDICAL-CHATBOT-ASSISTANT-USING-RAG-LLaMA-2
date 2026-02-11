from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

# Path to your FAISS database
DB_FAISS_PATH = "vectorstores/db_faiss"

# Custom prompt template
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

# Set up the custom prompt
def set_custom_prompt():
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

# Load LLaMA model via CTransformers
def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        max_new_tokens=256,  
        temperature=0.5
    )
    return llm

# Create the RetrievalQA chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
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

# Simple query function (optional utility)
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({"query": query})
    return response

# --- Chainlit Event Handlers ---

@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()

    msg.content = "Hi, welcome to the Medical Assistant Bot! What would you like to know?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += "\n\nSources:\n"
        for doc in sources:
            answer += f"- {doc.metadata.get('source', 'Unknown Source')}\n"
    else:
        answer += "\n\nNo sources found."

    await cl.Message(content=answer).send()
