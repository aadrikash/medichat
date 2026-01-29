from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq  # Changed to Groq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Create Flask App
app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')  # Use Groq instead of OpenAI

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# 1. Load Embeddings (384-dim for medichat)
embeddings = download_hugging_face_embeddings()

# 2. Connect to existing Pinecone index
index_name = "medichat" 
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# 3. Setup Retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# 4. Initialize Groq LLM (The "Brain")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4
)

# 5. Setup Prompt Template
system_prompt = (
    "You are a helpful medical assistant. Use the following pieces of "
    "retrieved context to answer the user's question. If you don't know "
    "the answer, just say you don't know based on the documents. \n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 6. Create RAG Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- ROUTES ---

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"User Query: {msg}")
    
    # Generate response
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    
    return str(response["answer"])

if __name__ == '__main__':
    # Default Flask port is 5000, video suggested 8080
    app.run(host="0.0.0.0", port=9000, debug=True)





    print("1. Starting App...")
load_dotenv()
print("2. Env loaded.")

print("3. Loading Embeddings (This can take 2-5 mins the first time)...")
embeddings = download_hugging_face_embeddings()
print("4. Embeddings Ready.")

print("5. Connecting to Pinecone...")
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medichat",
    embedding=embeddings
)
print("6. Pinecone Connected.")