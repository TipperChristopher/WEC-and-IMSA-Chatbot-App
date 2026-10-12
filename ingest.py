import os
from importlib import import_module

_llama_index_core = import_module("llama_index.core")
SimpleDirectoryReader = _llama_index_core.SimpleDirectoryReader
VectorStoreIndex = _llama_index_core.VectorStoreIndex
StorageContext = _llama_index_core.StorageContext
try:
    ChromaVectorStore = import_module(
        "llama_index.vector_stores.chroma"
    ).ChromaVectorStore
except ImportError as exc:
    raise ImportError(
        "Chroma integration is unavailable. Install it with: "
        "pip install llama-index-vector-stores-chroma"
    ) from exc

def run_ingestion():
    try:
        LlamaParse = import_module("llama_parse").LlamaParse
    except ImportError as exc:
        raise ImportError(
            "LlamaParse is not installed. Install it with: pip install llama-parse"
        ) from exc

    # Update the path to match your folder structure
    pdf_dir = "PDF Files for the Research"
    db_path = "db/chroma_db" # Let's save the vector DB in your existing 'db' folder
    
    if not os.path.exists(pdf_dir) or not os.listdir(pdf_dir):
        print(f"No PDF files found in {pdf_dir}")
        return

    print("Initializing LlamaParse parser...")
    parser = LlamaParse(
        result_type="markdown",
        verbose=True
    )
    
    file_extractor = {".pdf": parser}
    
    # We define a function to filter ONLY the rulebooks and ignore university docs
    def file_filter(file_path: str) -> bool:
        file_name = os.path.basename(file_path).lower()
        # Ignore your COMPSCI and ENGGEN research project files
        if "compsci" in file_name or "enggen" in file_name or "research" in file_name:
            return False
        return True

    print(f"Extracting specific PDF contents from {pdf_dir}...")
    reader = SimpleDirectoryReader(
        input_dir=pdf_dir, 
        file_extractor=file_extractor,
        file_metadata=lambda filename: {"file_name": os.path.basename(filename)}
    )
    
    # Manually filter the files before loading
    reader.input_files = [f for f in reader.input_files if file_filter(str(f))]
    documents = reader.load_data()

    # Attach series metadata so the chatbot knows which rules apply to WEC vs IMSA
    for doc in documents:
        file_name = doc.metadata.get("file_name", "").lower()
        if "wec" in file_name:
            doc.metadata["series"] = "WEC"
        elif "imsa" in file_name:
            doc.metadata["series"] = "IMSA"
        else:
            doc.metadata["series"] = "General"

    print("Storing embeddings in ChromaDB...")
    vector_store = ChromaVectorStore.from_params(
        collection_name="racing_regulations",
        persist_dir=db_path,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    print(f"Ingestion complete! Index saved to {db_path}")

if __name__ == "__main__":
    run_ingestion()