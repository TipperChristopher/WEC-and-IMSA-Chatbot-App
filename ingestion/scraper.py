# ingestion/scraper.py
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote

def download_alkamel_pdfs(base_url, target_dir="data/timing_pdfs"):
    """
    Scrapes the Al Kamel timing results webpage and downloads all PDF assets
    preserving event names and session states.
    """
    os.makedirs(target_dir, exist_ok=True)
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Locate all timing, entry list, and classification PDF links
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.lower().endswith('.pdf'):
            full_url = urljoin(base_url, href)
            file_name = unquote(full_url.split('/')[-1])
            
            # Local write path
            file_path = os.path.join(target_dir, file_name)
            print(f"Downloading: {file_name}")
            
            pdf_data = requests.get(full_url).content
            with open(file_path, "wb") as f:
                with open(file_path, "wb") as f:
                    f.write(pdf_data)