import os
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ==== API Keys ====
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_region = "us-east-1"  # Update based on your Pinecone deployment

# ==== Initialize Clients ====
client = OpenAI(api_key=openai_api_key)

pc = Pinecone(api_key=pinecone_api_key)
index_name = "ahl-video-data"
embedding_model = "text-embedding-3-small"
embedding_dim = 1536

# ==== Create Pinecone Index if Needed ====
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embedding_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=pinecone_region)
    )
    print(f"✅ Created index: {index_name}")

index = pc.Index(index_name)

# ==== Initialize Text Splitter ====
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# ==== Upload to Pinecone ====
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
                "text": chunks[i]  # ✅ Add this line
            }
        })


    index.upsert(vectors=vectors)
    print(f"✅ Uploaded {len(vectors)} vectors for {video_id}")

# ==== Main Execution ====
if __name__ == "__main__":
    file_path = "your_file.txt"  # Replace with your actual transcript file
    video_id = "ahl_video_001"   # Update this identifier as needed
    if not os.path.exists(file_path):
        print("❌ File not found:", file_path)
    else:
        upload_to_pinecone(file_path, video_id)
