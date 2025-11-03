# ---------- crawl_and_store.py ----------
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client, Client

# ------------------ CONFIG ------------------
BASE_URL = "https://docs.n8n.io/"
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # use service role key
SUPABASE_TABLE = "documents"
# --------------------------------------------

def crawl_website(base_url, max_pages=50):
    """Recursively crawl and extract text from a website."""
    visited = set()
    to_visit = [base_url]
    docs = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited or not url.startswith(base_url):
            continue
        visited.add(url)

        try:
            response = requests.get(url, timeout=10)
            if "text/html" not in response.headers.get("content-type", ""):
                continue
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract text content
            text = ' '.join(p.get_text() for p in soup.find_all(['p', 'li', 'h1', 'h2', 'h3']))
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": url}))

            # Extract links
            for link_tag in soup.find_all("a", href=True):
                link = urljoin(url, link_tag['href'])
                if link.startswith(base_url) and link not in visited:
                    to_visit.append(link)
        except Exception as e:
            print(f"Error fetching {url}: {e}")

    return docs


def main():
    print(f"Crawling {BASE_URL} ...")
    all_docs = crawl_website(BASE_URL, max_pages=2)
    print(f"Extracted {len(all_docs)} pages from {BASE_URL}")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = text_splitter.split_documents(all_docs)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Connect to Supabase
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Store in Supabase vector table
    print("Uploading embeddings to Supabase...")
    SupabaseVectorStore.from_documents(
        chunks,
        embeddings,
        client=supabase,
        table_name=SUPABASE_TABLE,
    )

    print("✅ Data uploaded successfully to Supabase!")


if __name__ == "__main__":
    main()