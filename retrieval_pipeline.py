from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# Configuration
PERSIST_DIRECTORY = "db/chroma_db"
EMBEDDING_MODEL = "mxbai-embed-large"

# Load Embedding Model
embedding_model = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

# Load Chroma Database
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)
print(f"Loaded {db._collection.count()} document chunks.")

# User Query
user_message = input("What's your query?: ")
query = user_message

# Retriever
retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

# Alternative:
#
# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3
#     }
# )

# Retrieve Documents

relevant_docs = retriever.invoke(query)


print("User Query")
print(query)
print("Retrieved Context")

if not relevant_docs:
    print("No relevant documents found.")
else:
    for i, doc in enumerate(relevant_docs, start=1):
        print(f"\nDocument {i}")
        print("-" * 60)
        print(doc.page_content)
        print("-" * 60)
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")

print("\nRetrieved", len(relevant_docs), "documents.")

# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"
