from langchain_groq import ChatGroq
from dotenv import load_dotenv

def get_llm():
    load_dotenv()
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0)

def answer_question(question, vector_store):
    # Retrieve relevant document chunks
    documents = vector_store.similarity_search(question, k=10)
    # Get source pages
    sources = sorted(set(doc.metadata["page"] for doc in documents))
    # Build context
    context = "\n\n".join(f"Page {doc.metadata['page']}:\n{doc.page_content}" for doc in documents)
    # Grounded prompt
    prompt = f"""
        You are a document question-answering assistant.
        Answer the user's question using ONLY the information contained in the document context.
        If the answer is present in the context, answer it directly.
        If the answer is not present in the context, say: "I couldn't find the answer in the document."
        Do not use outside knowledge.
        DOCUMENT CONTEXT:
        {context}
        USER QUESTION:
        {question}
        ANSWER:
        """
    # Generate answer
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content, sources