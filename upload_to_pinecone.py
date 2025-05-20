import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === Step 1: Load API Keys from .env ===
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
pinecone_region = "us-east-1"  # Update if your Pinecone region is different

# === Step 2: Initialize Clients ===
client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)

# === Step 3: Create Index if it doesn't exist ===
embedding_model = "text-embedding-3-small"
embedding_dim = 1536

if pinecone_index_name not in pc.list_indexes().names():
    pc.create_index(
        name=pinecone_index_name,
        dimension=embedding_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=pinecone_region)
    )
    print(f"✅ Created index: {pinecone_index_name}")

index = pc.Index(pinecone_index_name)

# === Step 4: Text Splitter ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# === Step 5: Upload Function ===
def upload_to_pinecone(file_path, video_id):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    chunks = text_splitter.split_text(text)
    print(f"📄 Text split into {len(chunks)} chunks.")

    response = client.embeddings.create(
        input=chunks,
        model=embedding_model
    )
    embeddings = [item.embedding for item in response.data]

    vectors = []
    for i, embedding in enumerate(embeddings):
        vectors.append({
            "id": f"{video_id}_chunk_{i}",
            "values": embedding,
            "metadata": {
                "video_id": video_id,
                "chunk_index": i,
                "text": chunks[i]
            }
        })

    index.upsert(vectors=vectors)
    print(f"✅ Uploaded {len(vectors)} vectors for video_id '{video_id}'")

# === Step 6: Run the Script ===
if __name__ == "__main__":
    file_path = "MERGED_DATA.txt"  # Your transcript file
    video_id = "MERGED_DATA"       # Unique ID per transcript
    if not os.path.exists(file_path):
        print("❌ File not found:", file_path)
    else:
        upload_to_pinecone(file_path, video_id)
