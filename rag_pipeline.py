# rag_pipeline.py — The Complete RAG Pipeline
# Combines: Search (Phase 3) + Answer Generation (Phase 4)

import ollama
import chromadb
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────

print("⏳ Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model ready!")
print()

# Connect to our existing ChromaDB database
chroma_client = chromadb.PersistentClient(path="embeddings/chroma_db")
collection = chroma_client.get_collection("lextech_knowledge_base")

print(f"✅ Connected to ChromaDB!")
print(f"✅ Knowledge base has {collection.count()} chunks")
print()


# ─────────────────────────────────────────────────
# STEP 1: RETRIEVE RELEVANT CHUNKS
# ─────────────────────────────────────────────────

def retrieve_chunks(question, n_results=3):
    """
    Converts question to numbers and finds
    the most relevant chunks in ChromaDB.

    This is the R in RAG = RETRIEVAL
    """

    # Convert question to embedding numbers
    question_embedding = model.encode(question).tolist()

    # Search ChromaDB for closest chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )

    # Package results nicely
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "content": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "heading": results["metadatas"][0][i]["heading"],
            "score": results["distances"][0][i],
        })

    return chunks


# ─────────────────────────────────────────────────
# STEP 2: BUILD THE PROMPT
# ─────────────────────────────────────────────────

def build_prompt(question, chunks):
    """
    Builds the instruction we send to the AI.

    We give the AI:
    1. A role (support assistant)
    2. Rules (only use the context provided)
    3. The context (retrieved chunks)
    4. The question

    Daily life analogy:
    Like giving an employee a briefing document
    before they answer a customer call.
    They can ONLY answer based on that document.
    """

    # Format the chunks into readable context
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1}: {chunk['source']} "
            f"— {chunk['heading']}]\n"
            f"{chunk['content']}"
        )

    # Join all chunks with dividers
    context = "\n\n---\n\n".join(context_parts)

    # Build the full prompt
    prompt = f"""You are a helpful customer support assistant for LexTech, \
a cloud-based project management software.

IMPORTANT RULES:
1. Answer ONLY using the context provided below
2. If the answer is not in the context, say: \
"I don't have information about that in our documentation."
3. Always mention which document your answer comes from
4. Keep your answer clear and concise
5. Use bullet points for step-by-step instructions

CONTEXT FROM LEXTECH DOCUMENTATION:
{context}

CUSTOMER QUESTION: {question}

YOUR ANSWER:"""

    return prompt


# ─────────────────────────────────────────────────
# STEP 3: GENERATE ANSWER
# ─────────────────────────────────────────────────

def generate_answer(prompt):
    """
    Sends the prompt to our local Llama AI
    and gets back an answer.

    This is the G in RAG = GENERATION

    Ollama runs the AI model locally on your computer.
    No internet needed, completely free, completely private.
    """

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # Extract just the text from the response
    return response["message"]["content"]


# ─────────────────────────────────────────────────
# STEP 4: THE COMPLETE RAG PIPELINE
# ─────────────────────────────────────────────────

def ask(question):
    """
    The complete RAG pipeline in one function:

    1. RETRIEVE relevant chunks
    2. BUILD prompt with chunks
    3. GENERATE answer using AI
    4. Return answer with citations

    Daily life analogy:
    Like a smart librarian who:
    1. Finds the right books (retrieve)
    2. Opens them to the right pages (build prompt)
    3. Summarizes the answer for you (generate)
    4. Tells you which books they used (citations)
    """

    print(f"\n{'='*55}")
    print(f"❓ QUESTION: {question}")
    print(f"{'='*55}")

    # Step 1: Find relevant chunks
    print("\n🔍 Step 1: Searching knowledge base...")
    chunks = retrieve_chunks(question, n_results=3)

    print(f"   Found {len(chunks)} relevant chunks:")
    for i, chunk in enumerate(chunks):
        print(f"   {i+1}. {chunk['source']} — "
              f"{chunk['heading']} "
              f"(score: {chunk['score']:.3f})")

    # Step 2: Build the prompt
    print("\n📝 Step 2: Building prompt...")
    prompt = build_prompt(question, chunks)
    print(f"   Prompt length: {len(prompt)} characters")

    # Step 3: Generate the answer
    print("\n🤖 Step 3: Generating answer with Llama...")
    print("   (this takes 10-30 seconds)")
    answer = generate_answer(prompt)

    # Step 4: Show everything
    print(f"\n{'─'*55}")
    print("💬 ANSWER:")
    print(f"{'─'*55}")
    print(answer)

    print(f"\n{'─'*55}")
    print("📚 SOURCES USED:")
    print(f"{'─'*55}")
    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}] {chunk['source']} "
              f"— {chunk['heading']}")

    print(f"\n{'='*55}")

    return {
        "question": question,
        "answer": answer,
        "sources": chunks
    }


# ─────────────────────────────────────────────────
# MAIN: Test the complete RAG pipeline
# ─────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("🤖 LexTech RAG Pipeline — Phase 4")
    print("   Complete Question Answering System")
    print("=" * 55)
    print()

    # Test with 4 different questions
    questions = [
        "How do I reset my password?",
        "Can I get a refund after 30 days?",
        "How do I invite someone to my project?",
        "What happens if I upload a file that is too large?",
    ]

    for question in questions:
        result = ask(question)
        print()
        input("Press Enter for next question...")
        # input() pauses and waits for you to press Enter
        # so you can read each answer before moving on