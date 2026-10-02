from langchain_chroma import Chroma

def create_vector_store(chunks, embeddings):
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{"page": chunk["page"]} for chunk in chunks]
    vector_store = Chroma.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    return vector_store