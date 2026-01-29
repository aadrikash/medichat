from dotenv import load_dotenv
import os
# Importing the logic we built in our previous steps
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# We only need the Pinecone key for indexing
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# 1. Load and Split Data
# Note: Using 'research/data/' because that's where we found your PDFs
print("🔄 Loading PDFs from research/data/...")
extracted_data = load_pdf_file(data='research/data/') 
text_chunks = text_split(extracted_data)
print(f"✅ Created {len(text_chunks)} text chunks.")

# 2. Setup Embeddings (HuggingFace 384-dim)
print("🔄 Downloading HuggingFace Embeddings...")
embeddings = download_hugging_face_embeddings()

# 3. Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Using 'medichat' as we established in our previous chat
index_name = "medichat - test" 

# 4. Create Index if it doesn't exist
if index_name not in pc.list_indexes().names():
    print(f"🔄 Creating new index: {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=384, # Must be 384 for MiniLM-L6
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("✅ Index created successfully.")
else:
    print(f"ℹ️ Index '{index_name}' already exists. Adding data...")

# 5. Push to Pinecone Vector Store
print("🚀 Uploading documents to Pinecone... This may take a minute.")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=medichat,
    embedding=embeddings, 
)

print("✨ Done! Your medical knowledge base is now live in Pinecone.")