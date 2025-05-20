import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv

# Load your .env environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Define your full list of important URLs
urls = [
    "https://americanhairline.com/non-surgical-hair-replacement-systems-in-delhi/",
    "https://americanhairline.com/about-us/",
    "https://americanhairline.com/australian-mirage-hair-patch/",
    "https://americanhairline.com/support/",
    "https://americanhairline.com/scalp-micropigmentation/#smp-for-alopecia",
    "https://americanhairline.com/blogs/everything-you-need-to-know-about-hair-systems-for-men/",
    "https://americanhairline.com/swiss-lace-hair-patch/",
    "https://americanhairline.com/blogs/hair-patch-for-men-or-hair-transplant-in-india-whats-right-for-you/",
    "https://americanhairline.com/scalp-micropigmentation/#smp-for-crown-area",
    "https://americanhairline.com/scalp-micropigmentation/",
    "https://americanhairline.com/front-hairline-transplant-2/",
    "https://americanhairline.com/blogs/selecting-the-correct-hair-density-for-your-hair-system/",
    "https://americanhairline.com/privacy-policy/",
    "https://americanhairline.com/shop/",
    "https://americanhairline.com/hair-transplant/",
    "https://americanhairline.com/tape-glue-hair-system/",
    "https://americanhairline.com/u-n-l-framework/",
    "https://americanhairline.com/customized-hair-systems/",
    "https://americanhairline.com/front-hairline-patch/",
    "https://americanhairline.com/blogs/myths-about-non-surgical-hair-replacement/",
    "https://americanhairline.com/hair-replacement-services/",
    "https://americanhairline.com/my-account/",
    "https://americanhairline.com/non-surgical-hair-replacement-in-mumbai/",
    "https://americanhairline.com/blogs/are-hair-patches-safe/",
    "https://americanhairline.com/hair-wigs-for-men/",
    "https://americanhairline.com/crown-area-patch/",
    "https://americanhairline.com/hair-replacement-training/",
    "https://americanhairline.com/scalp-micropigmentation/#smp-for-full-head",
    "https://americanhairline.com/clipon-tape-hair-system/",
    "https://americanhairline.com/common-questions/",
    "https://americanhairline.com/clip-on-hair-system/",
    "https://americanhairline.com/blogs/should-i-consider-getting-a-hair-system/",
    "https://americanhairline.com/contact-us",
    "https://americanhairline.com/scalp-micropigmentation/#smp-for-front-hairline",
    "https://americanhairline.com/terms-of-service/",
    "https://americanhairline.com/blogs/what-causes-itchiness-or-irritation-in-hair-systems/",
    "https://americanhairline.com/disclaimer/",
    "https://americanhairline.com/hair-replacement-for-men/",
    "https://americanhairline.com/full-lace-french-lace-hair-systems/",
    "https://americanhairline.com/blogs/travel-tips-for-hair-system-wearers/",
    "https://americanhairline.com/scalp-micropigmentation/#smp-for-hair-transplant",
    "https://americanhairline.com/blogs/the-benefits-of-scalp-micropigmentation-for-thinning-hair-in-india/",
    "https://americanhairline.com/blogs/everything-you-need-to-know-about-maintaining-your-hair-system-in-india/",
    "https://americanhairline.com/",
    "https://americanhairline.com/blogs/",
    "https://americanhairline.com/cart/",
    "https://americanhairline.com/consultation-form/",
    "https://americanhairline.com/blogs/most-natural-looking-hairline-patches-in-india/",
    "https://americanhairline.com/hair-patch-for-men/",
    "https://americanhairline.com/non-surgical-hair-replacement-in-bangalore/",
    "https://americanhairline.com/skin-base-hair-systems/",
]

# Step 1: Scrape content from each URL
def scrape_page(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract paragraphs, headings, list items
        texts = [p.get_text(separator=" ", strip=True) for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li'])]

        page_text = "\n".join(texts)
        return page_text.strip()
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

# Step 2: Load and chunk content
all_documents = []
for url in urls:
    content = scrape_page(url)
    if content:
        all_documents.append({"text": content, "source": url})

print(f"Total documents scraped: {len(all_documents)}")

# Step 3: Chunk text into small blocks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

final_chunks = []
for doc in all_documents:
    chunks = splitter.split_text(doc["text"])
    for chunk in chunks:
        final_chunks.append({"text": chunk, "source": doc["source"]})

print(f"Total chunks prepared: {len(final_chunks)}")

# Step 4: Generate embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

texts = [chunk["text"] for chunk in final_chunks]
sources = [chunk["source"] for chunk in final_chunks]

print("Generating embeddings...")
embeddings = embedding_model.embed_documents(texts)
print("Embeddings generated.")

# Step 5: Upload to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pc.Index(PINECONE_INDEX_NAME)

print("Uploading to Pinecone...")
records = []
for i, (embedding, text, source) in enumerate(zip(embeddings, texts, sources)):
    records.append({
        "id": f"website-{i}",
        "values": embedding,
        "metadata": {
            "text": text,
            "source": source
        }
    })

# Optional: Upload in batches if very large
# Upload in batches of 100
batch_size = 100

for i in range(0, len(records), batch_size):
    batch = records[i:i+batch_size]
    index.upsert(batch)
    print(f"Uploaded batch {i // batch_size + 1} ({len(batch)} records)")

print("Upload complete!")
