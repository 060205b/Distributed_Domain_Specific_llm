import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import AutoTokenizer
import pypdf2
from docx import Document
import json

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = pypdf2.PdfReader(file)
            num_pages = len(reader.pages)
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text += page.extract_text()
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
    return text

def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        full_text = []
        for paragraph in doc.paragraphs:
            full_text.append(paragraph.text)
        return "\n".join(full_text)
    except Exception as e:
        print(f"Error processing DOCX {docx_path}: {e}")
        return ""

def clean_text(text):
    text = text.strip()
    text = " ".join(text.split())
    text = text.lower()
    return text

def preprocess_file(filepath, tokenizer, chunk_size=512, overlap=128):
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()

    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    elif ext == ".pdf":
        text = extract_text_from_pdf(filepath)
    elif ext == ".docx":
        text = extract_text_from_docx(filepath)
    else:
        print(f"Unsupported file type: {filepath}")
        return []

    text = clean_text(text)
    sentences = sent_tokenize(text)
    all_chunks = []

    current_chunk = []
    current_length = 0

    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        sentence_length = len(tokens)

        if current_length + sentence_length + 1 <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length + 1
        else:
            all_chunks.append(" ".join(current_chunk))
            # Handle overlap.  Start the new chunk with the last 'overlap' tokens of the previous chunk.
            previous_tokens = tokenizer.tokenize(" ".join(current_chunk))
            overlap_tokens = previous_tokens[-overlap:]
            current_chunk = [tokenizer.convert_tokens_to_string(overlap_tokens)]
            current_length = len(overlap_tokens)
            current_chunk.append(sentence)
            current_length += sentence_length + 1

    if current_chunk:  # Add the last chunk
        all_chunks.append(" ".join(current_chunk))

    return all_chunks


def main():
    model_name = "mistralai/Mistral-7B-v0.1"  # Or your chosen model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_dir = "data"  # Change this to your data directory
    output_dir = "data/preprocessed"
    os.makedirs(output_dir, exist_ok=True)
    preprocessed_data = []

    for filename in os.listdir(input_dir):
        if filename.startswith("dummy"): #Just to skip training with all files.
            continue
        filepath = os.path.join(input_dir, filename)
        chunks = preprocess_file(filepath, tokenizer)
        print(f"Processed {filepath}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
          preprocessed_data.append({
              "text": chunk
          })

    #Save the output
    output_file = os.path.join(output_dir,"preprocessed.json")
    with open(output_file, "w") as f:
      json.dump(preprocessed_data, f, indent=4)
    print("Preprocessing Done!!")



if __name__ == "__main__":
    main()
