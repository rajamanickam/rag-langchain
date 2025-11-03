# ---------- chatbot_rag.py ----------
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from supabase import create_client, Client

# ------------------ CONFIG ------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_TABLE = "documents"
# --------------------------------------------

def main():
    print("🔗 Connecting to Supabase vector store...")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Embeddings must match the ones used during upload
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load existing vector database
    vector_db = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name=SUPABASE_TABLE,
    )

    retriever = vector_db.as_retriever()

    # Set up LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    # Create RAG chain
    prompt = PromptTemplate.from_template(
        "You are a helpful AI assistant. Use the following context to answer accurately.\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}\nAnswer:"
    )

    stuff_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, stuff_chain)

    print("🤖 Chatbot is ready! Type 'exit' to quit.\n")

    while True:
        query = input("Enter your question: ")
        if query.strip().lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        response = rag_chain.invoke({"input": query})
        print("\nAnswer:", response["answer"])
        print("-" * 80)


if __name__ == "__main__":
    main()