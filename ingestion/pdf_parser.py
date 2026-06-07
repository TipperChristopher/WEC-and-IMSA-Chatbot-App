# ingestion/pdf_parser.py
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

def parse_alkamel_tables(pdf_path):
    """
    Parses structured timing, starting grids, and entry list PDFs
    using LlamaParse's multimodal layout engine.
    """
    # Configure local LlamaParse engine
    parser = LlamaParse(
        result_type="markdown",
        verbose=True,
        language="en",
        instruction="Extract all timing columns, car numbers, driver ranks, and team names exactly as tabular markdown."
    )
    
    # Parse PDF file
    documents = parser.load_data(pdf_path)
    # The output will be clean Markdown containing structured tables (e.g., S1, S2, lap times)
    return documents.text