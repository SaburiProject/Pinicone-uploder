import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Step 1: Load environment variables from .env file
load_dotenv()

# Step 2: Initialize Pinecone client
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Step 3: Connect to your index
index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

# Step 4: Delete all vectors
index.delete(delete_all=True)

print("✅ All vectors inside the index have been deleted.")
