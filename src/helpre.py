from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

# Extraxt text from PDFs
def load_pdf_files(data):
    loader = directoryLoader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    contining only 'source' in metadata and yje original page_content."""
    minimal_docs:List[Document] = []
    for doc in docs:
       src = doc.metadata.get("source")
       minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src} 
            )
       )
    return minimal_docs

#split text into chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len
    )
    text_chunk = text_splitter.split_documents(minimal_docs)
    return text_chunk

def download_hugging_face_embeddings():
    
    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings