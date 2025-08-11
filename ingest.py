from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
import os


DATA_PATH = "knowledge_base/"
DB_CHROMA_PATH = "db_chroma/"



def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob='*.txt', show_progress=True)
    documents = loader.load()
    print(f"{len(documents)} document(s) loaded.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"{len(texts)} text chunks created.")

    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                          model_kwargs={'device': 'cpu'})
    
    print("Vector database creating. it takes little time.")
    db = Chroma.from_documents(texts, embeddings, persist_directory=DB_CHROMA_PATH)

    print("Vector database successfully created and db_chroma saved in folder.")


if __name__ == "__main__":
    create_vector_db()