import os
from importlib import import_module
llama_index_core = import_module("llama_index.core")
SimpleDirectoryReader = llama_index_core.SimpleDirectoryReader
VectorStoreIndex = llama_index_core.VectorStoreIndex
StorageContext = llama_index_core.StorageContext
try:
    ChromaVectorStore = import_module("llama_index.vector_stores.chroma").ChromaVectorStore
except ModuleNotFoundError as exc:
    if exc.name == "llama_index.vector_stores.chroma":
        raise RuntimeError(
            "Chroma integration is required. Install it with: "
            "pip install llama-index-vector-stores-chroma"
        ) from exc
    raise

def run_ingestion():
    pdf_dir = "data/manuals"
    db_path = "data/chroma_db"
    
    if not os.path.exists(pdf_dir) or not os.listdir(pdf_dir):
        print(f"No PDF files found in {pdf_dir}")
        return

    print("Initializing LlamaParse parser...")
    # LlamaParse converts complex multi-column PDFs & tables into clean Markdown
    try:
        LlamaParse = import_module("llama_parse").LlamaParse
    except ModuleNotFoundError as exc:
        if exc.name == "llama_parse":
            raise RuntimeError("LlamaParse is required. Install it with: pip install llama-parse") from exc
        raise

    parser = LlamaParse(
        result_type="markdown",
        verbose=True,
        # api_key=os.getenv("LLAMA_CLOUD_API_KEY") # Required for LlamaParse
    )
    
    file_extractor = {".pdf": parser}
    
    print("Extracting PDF contents...")
    reader = SimpleDirectoryReader(input_dir=pdf_dir, file_extractor=file_extractor)
    documents = reader.load_data()

    # Automatically attach series metadata based on filename
    for doc in documents:
        file_name = doc.metadata.get("file_name", "").lower()
        if "wec" in file_name:
            doc.metadata["series"] = "WEC"
        elif "imsa" in file_name:
            doc.metadata["series"] = "IMSA"
        else:
            doc.metadata["series"] = "General"

    print("Storing embeddings in ChromaDB...")
    chromadb = import_module("chromadb")
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection("racing_regulations")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    print("Ingestion complete! Index saved to data/chroma_db")

if __name__ == "__main__":
    run_ingestion()