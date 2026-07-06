# embedder.py — converts chunks into embeddings
# using FREE local sentence-transformers model
# No API key needed!

import json
import os
from pathlib import Path

# sentence-transformers runs locally on your computer
# completely free, no API key needed
from sentence_transformers import SentenceTransformer

# Import ChromaDB — our vector database
import chromadb

# ─────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────

print("⏳ Loading embedding model...")
print("   (this takes ~30 seconds the first time)")
print()

# Load the free local embedding model
# "all-MiniLM-L6-v2" is a small but powerful model
# It converts text → 384 numbers (vs OpenAI's 1536)
# Downloads automatically first time (~90MB)
model = SentenceTransformer("all-MiniLM-L6-v2")

print("✅ Model loaded successfully!")
print()

# Create ChromaDB client
# Stores our vector database locally in embeddings/ folder
chroma_client = chromadb.PersistentClient(path="embeddings/chroma_db")


# ─────────────────────────────────────────────────
# LOAD CHUNKS
# ─────────────────────────────────────────────────

def load_chunks(chunks_file="chunks/heading_chunks.json"):
    """
    Loads chunks from the JSON file we saved in Phase 2.
    """
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"📂 Loaded {len(chunks)} chunks from {chunks_file}")
    return chunks


# ─────────────────────────────────────────────────
# GENERATE EMBEDDING
# ─────────────────────────────────────────────────

def get_embedding(text):
    """
    Converts text into a list of numbers using
    our FREE local sentence-transformers model.

    Daily life analogy:
    Like GPS coordinates but for MEANING.
    "password reset" and "forgot password" will have
    very similar coordinates because they mean the same thing.

    Returns: a list of 384 numbers
    """

    # Clean the text
    text = text.replace("\n", " ").strip()

    # Generate embedding using local model
    # .encode() does the conversion
    # .tolist() converts numpy array to regular Python list
    embedding = model.encode(text).tolist()

    return embedding


# ─────────────────────────────────────────────────
# STORE IN CHROMADB
# ─────────────────────────────────────────────────

def store_chunks_in_chromadb(chunks):
    """
    Stores all chunks AND their embeddings in ChromaDB.

    ChromaDB is a vector database — specially designed
    to store and search through lists of numbers (embeddings).

    Regular database  = finds by exact match
    Vector database   = finds by meaning/similarity
    """

    # Delete existing collection if it exists
    # This prevents duplicate data if we run this script twice
    try:
        chroma_client.delete_collection("lextech_knowledge_base")
        print("🗑️  Cleared existing collection")
    except:
        pass
        # pass means "do nothing if this fails"
        # collection might not exist yet — that's fine

    # Create fresh collection
    collection = chroma_client.create_collection(
        name="lextech_knowledge_base",
        metadata={"description": "LexTech support documents"}
    )

    print(f"📦 Storing {len(chunks)} chunks in ChromaDB...")
    print()

    # Process each chunk
    for i, chunk in enumerate(chunks):

        # Show progress
        print(f"   🔄 [{i+1}/{len(chunks)}] "
              f"{chunk['source']} — {chunk['heading'][:35]}")

        # Step 1: Convert chunk text to numbers
        embedding = get_embedding(chunk["content"])

        # Step 2: Store in ChromaDB
        collection.add(
            ids=[chunk["chunk_id"]],
            # unique ID for this chunk

            embeddings=[embedding],
            # the 384 numbers representing meaning

            documents=[chunk["content"]],
            # the actual text

            metadatas=[{
                "source": chunk["source"],
                "heading": chunk["heading"],
                "strategy": chunk["strategy"],
                "word_count": chunk["word_count"],
                "char_count": chunk["char_count"],
            }]
            # extra info stored alongside
        )

    print()
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB!")
    print(f"✅ Total chunks in database: {collection.count()}")

    return collection


# ─────────────────────────────────────────────────
# TEST SEARCH
# ─────────────────────────────────────────────────

def test_search(collection, query, n_results=3):
    """
    Tests our vector database by searching with a question.

    This is the MAGIC of embeddings:
    We search by MEANING not by exact keywords!

    Example:
    Query: "I forgot my password"
    Finds: chunks about "password reset" and "account locked"
    Even though the words are different!
    """

    print(f"\n🔍 Query: '{query}'")
    print("-" * 50)

    # Step 1: Convert question to numbers
    query_embedding = get_embedding(query)

    # Step 2: Search ChromaDB for closest chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    # Step 3: Display results
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(documents)):
        print(f"\n  📄 Result {i+1}:")
        print(f"     Source  : {metadatas[i]['source']}")
        print(f"     Heading : {metadatas[i]['heading']}")
        print(f"     Score   : {distances[i]:.4f} "
              f"(lower = more similar)")
        print(f"     Preview : {documents[i][:120]}...")


# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("🧠 LexTech Embedder — Phase 3")
    print("   Using FREE local sentence-transformers model")
    print("=" * 55)
    print()

    # STEP 1: Load chunks from Phase 2
    print("📂 Step 1: Loading chunks...")
    chunks = load_chunks("chunks/heading_chunks.json")
    print()

    # STEP 2: Embed and store in ChromaDB
    print("🧠 Step 2: Generating embeddings and storing...")
    print()
    collection = store_chunks_in_chromadb(chunks)

    # STEP 3: Test semantic search
    print()
    print("=" * 55)
    print("🧪 Step 3: Testing Semantic Search")
    print("=" * 55)
    print()
    print("Watch how it finds relevant chunks by MEANING")
    print("even when the exact words don't match!")
    print()

    # Test question 1
    test_search(
        collection,
        "How do I reset my password?"
    )

    # Test question 2
    test_search(
        collection,
        "Can I get my money back?"
    )

    # Test question 3
    test_search(
        collection,
        "How much does the service cost?"
    )

    # Test question 4 — tricky one!
    test_search(
        collection,
        "I can't log into my account"
    )

    print()
    print("=" * 55)
    print("✅ Phase 3 Complete!")
    print()
    print("What just happened:")
    print("  1. Loaded 35 chunks from Phase 2")
    print("  2. Converted each chunk to 384 numbers")
    print("  3. Stored numbers in ChromaDB")
    print("  4. Searched by meaning — not just keywords!")
    print("=" * 55)