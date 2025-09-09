from langchain_community.embeddings import HuggingFaceBgeEmbeddings, OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.prompts import PromptTemplate

import gradio as gr
import os, hashlib, requests

# suppress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')


## LLM
def get_llm(llm_name: str):
    return Ollama(model=llm_name, base_url="http://localhost:11434")



## Document loader
def document_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document

## Text splitter
def text_splitter(data, chunk_size=1000, chunk_overlap=150):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks

## Embedding model
def embedding(embed_choice: str):
    if embed_choice.startswith("HF:" ):
        model_name = embed_choice.replace("HF: ","")
        return HuggingFaceBgeEmbeddings(model_name=model_name, encode_kwargs = {"normalize_embeddings": True})
    else:
        #Ollama Embedding
        model_name = embed_choice.replace("Ollama: ", "")
        return OllamaEmbeddings(model=model_name, base_url="http://localhost:11434")

cross = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

def build_retriever_with_rerank(vectordb, k_candidates=7, k_final=4):
    base = vectordb.as_retriever(search_kwargs={"k": k_candidates})
    reranker = CrossEncoderReranker(model=cross, top_n=k_final)
    return ContextualCompressionRetriever(base_retriever=base, base_compressor=reranker)
 


#cache helper 
index_cache = {}
def cache_key(path, embed_choice, chunk_size, chunk_overlap):
    mtime = os.path.getmtime(path)
    return f"{path}|{mtime}|{embed_choice}|cs{chunk_size}|co{chunk_overlap}"

def persist_dir_for_key(key: str):
    return os.path.join("chroma_store", hashlib.md5(key.encode()).hexdigest())

def get_db_cached(file_path: str, chunk_size: int, chunk_overlap: int, embed_choice: str):  #retrieves and loads Chroma DB at once
    key = cache_key(file_path, embed_choice, chunk_size, chunk_overlap)
    if key in index_cache:
        return index_cache[key]

    #build/load persisted chroma index
    persist_dir = persist_dir_for_key(key)
    embeddings = embedding(embed_choice)
    if os.path.isdir(persist_dir) and os.listdir(persist_dir):
        db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        docs = document_loader(file_path)
        chunks = text_splitter(docs, chunk_size, chunk_overlap)
        db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
        db.persist()
    index_cache[key] = db
    return db

def make_retriever(db, use_reranker: bool, top_k: int, k_final: int):
    if use_reranker:
        return build_retriever_with_rerank(db, k_candidates=top_k, k_final=k_final)
    else:
        return db.as_retriever(search_kwargs={"k": top_k})
    

#Prompt Template
prompt = PromptTemplate.from_template(
    "You are a helpful assistant. Use ONLY the context to answer.\n"
    "If the answer is not in the context, say 'I don't know.'\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)



## QA Chain
def retriever_qa(file_path: str, query: str, top_k: int, chunk_size: int, chunk_overlap: int, use_reranker: bool, k_final: int, llm_name: str, embed_choice: str):
    if (llm_name or embed_choice.startswith("Ollama:")):
        try:
            requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        except Exception:
            raise gr.Error("Ollama server not reachable at http://localhost:11434. Please install/start Ollama.")
    db = get_db_cached(file_path, chunk_size, chunk_overlap, embed_choice)
    retr = make_retriever(db, use_reranker, top_k, k_final)
    qa = RetrievalQA.from_chain_type(llm=get_llm(llm_name), 
                                    chain_type="stuff", 
                                    retriever=retr, 
                                    return_source_documents=True,
                                    chain_type_kwargs={"prompt": prompt},
                                    )
    try:
        response = qa.invoke({"query": query})
    except Exception as e:
        raise gr.Error(f"Model couldn't load (low RAM). Try a smaller model. Details:{e}")
    #Citations:
    cites = []
    for i, d in enumerate(response["source_documents"], 1):
        page = d.metadata.get("page")
        if isinstance(page, int):
            page += 1 #display 1-based
            cites.append(f"{i}.page{page}")
    return response["result"] + ("\nSources:\n" + "\n".join(cites) if cites else "")
    

# Create Gradio interface - added UI - tunable parameters follow sequence of parameters in retriever function
rag_application = gr.Interface(
    fn=retriever_qa,
    flagging_mode="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),  # Drag and drop file upload
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here..."),
        gr.Slider(3, 12, value=8, step=1, label="Top-K"),
        gr.Slider(300, 1500, value=1000, step=50, label="Chunk size"),
        gr.Slider(0, 300, value=120, step=10, label="Chunk overlap"),
        gr.Checkbox(value=True, label="Use reranker"),
        gr.Slider(2, 6, value=4, step=1, label="K_final (after rerank)"),
        gr.Dropdown(choices=["mixtral", "mistral", "llama3:8b", "llama3:8b-instruct-q4_0", "phi3:mini"],
                    value="mistral",
                    label="LLM (Ollama)"
                    ),
        gr.Dropdown(choices=["HF: sentence-transformers/all-MiniLM-L6-v2",
                     "Ollama: nomic-embed-text",
                     "Ollama: mxbai-embed-large"
                     ],
                    value="HF: sentence-transformers/all-MiniLM-L6-v2",
                    label = "Embedding Models"
        )
    ],
    outputs=gr.Textbox(label="Answer"),
    title="RAG-based PDF QA Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document. Retrieval parameters are tunable with sliders, and specific LLM and embedding model can be chosen. Scores are shown when reranker is off."
)

# Launch the app
rag_application.launch(server_name="localhost", server_port=7860)