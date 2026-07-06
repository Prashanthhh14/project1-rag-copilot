# chunker.py — splits documents into smaller searchable pieces

import json
from pathlib import Path
from loader import load_documents  # reuse our loader from Phase 1!


# ─────────────────────────────────────────────────
# STRATEGY 1: Heading-Based Chunking
# ─────────────────────────────────────────────────

def chunk_by_headings(document):
    """
    Splits a markdown document at ## headings.
    Each section becomes one chunk.

    Example:
        ## What is LexTech?        ← chunk 1 starts
        LexTech is a cloud app...
        ## How do I sign up?       ← chunk 2 starts
        Go to lextech.com...
    """

    chunks = []          # empty list to collect chunks
    content = document["content"]   # the full text
    filename = document["filename"] # e.g. "faq.md"

    # Split content into individual lines
    lines = content.split("\n")
    # "\n" means newline — the invisible character when you press Enter

    # We'll build each chunk line by line
    current_heading = "Introduction"  # default if no heading yet
    current_lines = []                # lines collected for current chunk

    for line in lines:

        # Is this line a section heading?
        # "##" in markdown = a heading
        # startswith() checks if line BEGINS with those characters
        if line.startswith("##"):

            # Before starting new chunk, SAVE the current one
            if current_lines:
                chunk_text = "\n".join(current_lines).strip()
                # "\n".join() glues lines back together with newlines
                # .strip() removes blank spaces at start and end

                if chunk_text:  # only save if not empty
                    chunks.append({
                        "chunk_id": f"{filename}_chunk_{len(chunks)}",
                        "source": filename,
                        "heading": current_heading,
                        "content": chunk_text,
                        "strategy": "heading",
                        "word_count": len(chunk_text.split()),
                        "char_count": len(chunk_text),
                    })

            # Start fresh for new chunk
            current_heading = line.strip("#").strip()
            # strip("#") removes the ## symbols
            # strip() removes leftover spaces
            current_lines = [line]

        else:
            # Regular line — add to current chunk
            current_lines.append(line)

    # ⚠️ IMPORTANT: Save the LAST chunk
    # The loop ends before saving the final chunk
    # so we have to do it manually here
    if current_lines:
        chunk_text = "\n".join(current_lines).strip()
        if chunk_text:
            chunks.append({
                "chunk_id": f"{filename}_chunk_{len(chunks)}",
                "source": filename,
                "heading": current_heading,
                "content": chunk_text,
                "strategy": "heading",
                "word_count": len(chunk_text.split()),
                "char_count": len(chunk_text),
            })

    return chunks


# ─────────────────────────────────────────────────
# STRATEGY 2: Fixed-Size Chunking
# ─────────────────────────────────────────────────

def chunk_by_size(document, chunk_size=150, overlap=30):
    """
    Splits document into fixed word-count chunks with overlap.

    chunk_size = max words per chunk (default: 150)
    overlap    = words repeated between chunks (default: 30)

    Why overlap?
    Without overlap:
        chunk1: "...the refund must be requested within"
        chunk2: "14 days of purchase by emailing..."
        ← sentence got CUT in half — AI loses context!

    With overlap:
        chunk1: "...the refund must be requested within"
        chunk2: "requested within 14 days of purchase..."
        ← repeated words keep the context connected ✅
    """

    chunks = []
    filename = document["filename"]

    # Split ALL content into individual words
    words = document["content"].split()
    total_words = len(words)

    start = 0         # where current chunk starts
    chunk_number = 0  # counter for naming chunks

    # Keep making chunks until we reach the end of the document
    while start < total_words:

        # Where does this chunk end?
        end = start + chunk_size

        # Grab the words for this chunk
        # [start:end] = "give me words from position start up to end"
        chunk_words = words[start:end]

        # Glue words back into readable text
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "chunk_id": f"{filename}_size_{chunk_number}",
            "source": filename,
            "heading": f"chunk_{chunk_number}",
            "content": chunk_text,
            "strategy": "fixed_size",
            "word_count": len(chunk_words),
            "char_count": len(chunk_text),
            "start_word": start,
            "end_word": min(end, total_words),
            # min() picks the smaller of two numbers
            # prevents going past the end of the document
        })

        chunk_number += 1

        # Move forward — but subtract overlap so chunks share words
        # Example: chunk_size=150, overlap=30
        # chunk1: words 0-150
        # chunk2: words 120-270  (starts 30 words back = overlap)
        # chunk3: words 240-390
        start += chunk_size - overlap

    return chunks


# ─────────────────────────────────────────────────
# SAVE CHUNKS TO FILE
# ─────────────────────────────────────────────────

def save_chunks(chunks, output_file):
    """
    Saves chunks to a JSON file for later use.

    JSON = JavaScript Object Notation
    It's just a way to save structured data as text.
    Like a spreadsheet but saved as a text file.

    Example JSON:
    [
        {
            "chunk_id": "faq.md_chunk_0",
            "content": "What is LexTech?...",
            ...
        }
    ]
    """

    # Make sure chunks/ folder exists
    # exist_ok=True means don't crash if it already exists
    Path("chunks").mkdir(exist_ok=True)

    # Open file for writing and save chunks as JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
        # indent=2 makes the file nicely formatted and readable

    print(f"💾 Saved {len(chunks)} chunks → {output_file}")


# ─────────────────────────────────────────────────
# MAIN: Run Everything
# ─────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("✂️  LexTech Document Chunker — Phase 2")
    print("=" * 55)
    print()

    # STEP 1: Load documents using our Phase 1 loader
    print("📂 Step 1: Loading documents...")
    print()
    documents = load_documents()
    print()

    # STEP 2: Chunk every document using both strategies
    print("✂️  Step 2: Chunking documents...")
    print()

    all_heading_chunks = []  # collect ALL heading chunks
    all_size_chunks = []     # collect ALL size chunks

    for doc in documents:

        # Apply both chunking strategies
        heading_chunks = chunk_by_headings(doc)
        size_chunks = chunk_by_size(doc, chunk_size=150, overlap=30)

        # Add results to our master lists
        all_heading_chunks.extend(heading_chunks)
        all_size_chunks.extend(size_chunks)

        # Show results for this document
        print(f"  📄 {doc['filename']}")
        print(f"     Heading chunks : {len(heading_chunks)}")
        print(f"     Fixed-size chunks : {len(size_chunks)}")
        print()

    # STEP 3: Show totals
    print("=" * 55)
    print(f"📊 TOTAL HEADING CHUNKS   : {len(all_heading_chunks)}")
    print(f"📊 TOTAL FIXED-SIZE CHUNKS: {len(all_size_chunks)}")
    print("=" * 55)
    print()

    # STEP 4: Save both sets of chunks to files
    print("💾 Step 3: Saving chunks...")
    print()
    save_chunks(all_heading_chunks, "chunks/heading_chunks.json")
    save_chunks(all_size_chunks,    "chunks/size_chunks.json")

    # STEP 5: Show a sample chunk so we can inspect it
    print()
    print("=" * 55)
    print("🔍 SAMPLE CHUNK (first heading chunk):")
    print("=" * 55)
    print()

    if all_heading_chunks:
        sample = all_heading_chunks[0]
        print(f"  chunk_id : {sample['chunk_id']}")
        print(f"  source   : {sample['source']}")
        print(f"  heading  : {sample['heading']}")
        print(f"  words    : {sample['word_count']}")
        print(f"  chars    : {sample['char_count']}")
        print()
        print(f"  content preview:")
        print(f"  {sample['content'][:300]}")
        print()

    print("✅ Phase 2 Complete!")