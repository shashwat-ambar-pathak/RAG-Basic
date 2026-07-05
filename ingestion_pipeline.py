import os
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# -----------------------------
# Configuration
# -----------------------------
DOCS_PATH = "docs"
PERSIST_DIRECTORY = "db/chroma_db"
EMBEDDING_MODEL = "mxbai-embed-large"


def load_documents(docs_path=DOCS_PATH):
    """
    Load all .txt files from the docs directory.
    """

    print(f"\nLoading documents from '{docs_path}'...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"Directory '{docs_path}' does not exist."
        )

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    documents = loader.load()

    if not documents:
        raise FileNotFoundError(
            f"No .txt files found inside '{docs_path}'."
        )

    print(f"\nLoaded {len(documents)} document(s).\n")

    for i, doc in enumerate(documents[:2], start=1):
        print(f"Document {i}")
        print(f"Source : {doc.metadata['source']}")
        print(f"Length : {len(doc.page_content)} characters")
        print(f"Preview: {doc.page_content[:100]}...")
        print("-" * 60)

    return documents


def split_documents(documents):
    """
    Split documents into chunks.
    """

    print("\nSplitting documents into chunks...\n")

    splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.\n")

    return chunks


def create_vector_store(chunks):
    """
    Create and save the Chroma vector database.
    """

    print("Loading Ollama embedding model...")

    embedding_model = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    print("Creating Chroma vector store...\n")
    """
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIRECTORY,
        collection_metadata={
            "hnsw:space": "cosine"
        },
    )
    """
    
    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )

    batch_size = 50

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Adding batch {i//batch_size + 1}: {len(batch)} chunks")
        vectorstore.add_documents(batch)

    print("Vector database created successfully!")
    
    print(
        f"Successfully stored {vectorstore._collection.count()} chunks."
    )

    return vectorstore

#print(f"Largest chunk: {max_len} characters")
def load_existing_vector_store():
    """
    Load an existing Chroma vector database.
    """

    embedding_model = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model,
        collection_metadata={
            "hnsw:space": "cosine"
        },
    )

    print(
        f"Loaded existing vector database containing {vectorstore._collection.count()} chunks."
    )

    return vectorstore


def main():

    print("=" * 50)
    print("RAG DOCUMENT INGESTION PIPELINE")
    print("=" * 50)

    if os.path.exists(PERSIST_DIRECTORY):

        print("\nVector database already exists.")

        load_existing_vector_store()

        print("\nNothing to do.")

        return

    print("\nNo vector database found.")

    documents = load_documents()

    chunks = split_documents(documents)

    create_vector_store(chunks)

    print("\nIngestion completed successfully!")


if __name__ == "__main__":
    main()



# documents = [
#    Document(
#        page_content="Google LLC is an American multinational corporation and technology company focusing on online advertising, search engine technology, cloud computing, computer software, quantum computing, e-commerce, consumer electronics, and artificial intelligence (AI).",
#        metadata={'source': 'docs/google.txt'}
#    ),
#    Document(
#        page_content="Microsoft Corporation is an American multinational corporation and technology conglomerate headquartered in Redmond, Washington.",
#        metadata={'source': 'docs/microsoft.txt'}
#    ),
#    Document(
#        page_content="Nvidia Corporation is an American technology company headquartered in Santa Clara, California.",
#        metadata={'source': 'docs/nvidia.txt'}
#    ),
#    Document(
#        page_content="Space Exploration Technologies Corp., commonly referred to as SpaceX, is an American space technology company headquartered at the Starbase development site in Starbase, Texas.",
#        metadata={'source': 'docs/spacex.txt'}
#    ),
#    Document(
#        page_content="Tesla, Inc. is an American multinational automotive and clean energy company headquartered in Austin, Texas.",
#        metadata={'source': 'docs/tesla.txt'}
#    )
# ]