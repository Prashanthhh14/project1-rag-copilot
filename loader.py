# loader.py — reads all documents from the docs folder

# import means "bring in a tool someone else built"
import os  # os = operating system tool, lets us work with files and folders
from pathlib import Path  # Path = a smarter way to handle file paths

# python-dotenv reads our .env file and loads the API key safely
from dotenv import load_dotenv

# Load the .env file so our API key is available
load_dotenv()

# --- FUNCTION: Load all documents ---
# A function is a reusable block of code
# "def" means "define a new function"
# "load_documents" is the name we're giving it
# "docs_folder" is the input it expects (like an ingredient)

def load_documents(docs_folder="docs"):
    """
    Reads all .md files from the docs folder.
    Returns a list of dictionaries with filename and content.
    """
    # An empty list to store our documents
    # Think of it like an empty box we'll fill with papers
    documents = []

    # Path() turns our folder name into a proper file path
    # that works on Windows, Mac, and Linux
    folder_path = Path(docs_folder)

    # Check if the folder actually exists
    # If someone deleted it by accident, we warn them instead of crashing
    if not folder_path.exists():
        print(f"Error: Folder '{docs_folder}' not found!")
        return documents  # Return empty list

    # Loop through every file in the docs folder
    # "glob" means "find files matching a pattern"
    # "*.md" means "any file ending in .md"
    for file_path in folder_path.glob("*.md"):

        # Open the file and read its contents
        # "r" means "read mode" (not write mode)
        # "encoding=utf-8" means handle special characters properly
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Create a dictionary (like a labeled box) with info about this doc
        # A dictionary stores key:value pairs
        document = {
            "filename": file_path.name,        # e.g. "faq.md"
            "filepath": str(file_path),         # e.g. "docs/faq.md"
            "content": content,                 # the actual text inside
            "char_count": len(content),         # how many characters
            "word_count": len(content.split()), # how many words
        }

        # Add this document to our list
        documents.append(document)

        # Print a message so we know it worked
        print(f"✅ Loaded: {file_path.name} ({len(content)} characters)")

    return documents  # Send the list back to whoever called this function


# --- MAIN: Run the loader ---
# This block only runs when you execute THIS file directly
# It won't run if another file imports this one
if __name__ == "__main__":

    print("=" * 50)
    print("📚 LexTech Support Knowledge Base — Document Loader")
    print("=" * 50)
    print()

    # Call our function and store the result
    docs = load_documents()

    print()
    print("=" * 50)
    print(f"📊 SUMMARY: Loaded {len(docs)} documents")
    print("=" * 50)
    print()

    # Loop through each document and show its details
    for i, doc in enumerate(docs):
        # enumerate gives us a counter (i) alongside each item
        print(f"Document {i+1}: {doc['filename']}")
        print(f"  Words: {doc['word_count']}")
        print(f"  Characters: {doc['char_count']}")
        print(f"  Preview: {doc['content'][:100]}...")
        # [:100] means "show only the first 100 characters"
        print()