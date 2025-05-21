import os
from pathlib import Path
import PyPDF2
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pickle
from tqdm import tqdm
import json
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import re

class DocumentProcessor:
    def __init__(self, pdf_folder="pdfs", output_folder="processed_data"):
        self.pdf_folder = pdf_folder
        self.output_folder = output_folder
        self.metadata_file = os.path.join(output_folder, "chunk_metadata.json")
        self.vector_store_file = os.path.join(output_folder, "vector_store.faiss")
        self.embeddings_file = os.path.join(output_folder, "embeddings.pkl")
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Load CSV data for metadata
        self.csv_data = self.load_csv_data()
        
        # Set the embedding model - using a more powerful model for medical/technical content
        self.embedding_model_name = "pritamdeka/S-PubMedBert-MS-MARCO"
        
        # Save which model was actually used
        self.used_model_name = self.embedding_model_name
        
    def load_csv_data(self):
        """Load and process the CSV file with additional information"""
        try:
            df = pd.read_csv('combined_list_with_filenames.csv')
            return df.set_index('psg filename').to_dict('index')
        except Exception as e:
            print(f"Error loading CSV file: {str(e)}")
            return {}

    def clean_text(self, text):
        """Clean and normalize text for better embedding quality"""
        if not text:
            return ""
            
        # Replace multiple spaces, newlines, tabs with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Fix hyphenated words that might be split across lines
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        
        # Remove unnecessary characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z0-9.,;:?!()\[\]{}\'"%-]', ' ', text)
        
        # Normalize whitespace again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def extract_text_with_metadata(self, pdf_path):
        """Extract text from PDF with page numbers"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                pages = []
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():  # Only add non-empty pages
                        # Clean the text before saving
                        clean_text = self.clean_text(text)
                        if clean_text:
                            pages.append({
                                'text': clean_text,
                                'page': page_num + 1,
                                'filename': os.path.basename(pdf_path)
                            })
                return pages
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {str(e)}")
            return []

    def create_embeddings_with_progress(self, texts):
        """Create embeddings with progress tracking using the specified model"""
        print(f"Loading embedding model: {self.embedding_model_name}...")
        try:
            model = SentenceTransformer(self.embedding_model_name)
            
            print("Creating embeddings...")
            embeddings = []
            batch_size = 16  # Smaller batch size for the larger model
            
            for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
                batch = texts[i:i + batch_size]
                batch_embeddings = model.encode(batch, show_progress_bar=False)
                embeddings.extend(batch_embeddings)
            
            # We successfully used the primary model
            self.used_model_name = self.embedding_model_name
            return np.array(embeddings)
        except Exception as e:
            print(f"Error with primary embedding model: {str(e)}")
            print("Falling back to default embedding model: all-MiniLM-L6-v2")
            
            # Fallback to simpler model
            fallback_model_name = "all-MiniLM-L6-v2"
            model = SentenceTransformer(fallback_model_name)
            embeddings = []
            batch_size = 32
            
            for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
                batch = texts[i:i + batch_size]
                batch_embeddings = model.encode(batch, show_progress_bar=False)
                embeddings.extend(batch_embeddings)
            
            # We had to use the fallback model
            self.used_model_name = fallback_model_name
            return np.array(embeddings)

    def extract_keywords_from_text(self, text):
        """Extract potential technical terms or keywords from text"""
        # Simple pattern for technical terms (can be expanded)
        patterns = [
            r'(?<!\w)[A-Z][a-z]+(?: [A-Z][a-z]+)*(?!\w)',  # Capitalized terms
            r'(?<!\w)[A-Z]{2,}(?!\w)',  # Acronyms
            r'(?<!\w)(?:\d+\.?\d*|\d*\.?\d+) ?(?:mg|kg|ml|g|mcg|μg|ng)(?!\w)',  # Measurements
            r'(?<!\w)(?:narrow therapeutic index|therapeutic index|drug|active ingredient|dissolution|bioequivalence|in vitro|in vivo|pharmacokinetic|pharmacodynamic)(?!\w)'  # Domain specific
        ]
        
        keywords = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            keywords.extend(matches)
        
        # Remove duplicates and sort by length (descending)
        keywords = list(set(keywords))
        keywords.sort(key=len, reverse=True)
        
        return keywords[:20]  # Limit to most significant keywords

    def process_documents(self):
        """Process all PDF documents and create vector store"""
        print("Starting document processing...")
        
        # Get list of PDF files
        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.lower().endswith('.pdf')]
        if not pdf_files:
            print(f"No PDF files found in {self.pdf_folder}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each PDF file
        all_pages = []
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            pdf_path = os.path.join(self.pdf_folder, pdf_file)
            pages = self.extract_text_with_metadata(pdf_path)
            all_pages.extend(pages)
        
        print(f"Extracted text from {len(all_pages)} pages")
        
        # Split text into chunks with improved parameters for technical content
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Smaller chunks for more precise retrieval
            chunk_overlap=500,  # Decent overlap to avoid splitting concepts
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", ".\n", "!\n", "?\n", ";", ":", ",", " ", ""],
            keep_separator=True
        )
        
        # Prepare texts and metadata
        texts = []
        metadatas = []
        
        print("Splitting text into chunks...")
        for page in tqdm(all_pages, desc="Creating chunks"):
            # Extract potential keywords from the page text
            keywords = self.extract_keywords_from_text(page['text'])
            
            # Split the text into chunks
            chunks = text_splitter.split_text(page['text'])
            for chunk in chunks:
                # Check if chunk is not too short
                if len(chunk.strip()) < 50:
                    continue
                    
                texts.append(chunk)
                # Create metadata for each chunk
                metadata = {
                    'filename': page['filename'],
                    'page': page['page'],
                    'chunk_index': len(texts) - 1,
                    'chunk_length': len(chunk),
                    'keywords': ';'.join(k for k in keywords if k.lower() in chunk.lower())
                }
                # Add CSV data if available
                if page['filename'] in self.csv_data:
                    metadata.update(self.csv_data[page['filename']])
                metadatas.append(metadata)
        
        print(f"Created {len(texts)} chunks")
        
        # Create an augmented text list for embedding
        augmented_texts = []
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            # For each chunk, add keywords to the beginning to emphasize important terms
            keywords = metadata.get('keywords', '').split(';') if metadata.get('keywords') else []
            if keywords:
                prefix = ' '.join(keywords[:5]) + ': '  # Use up to 5 keywords
                augmented_text = prefix + text
            else:
                augmented_text = text
            augmented_texts.append(augmented_text)
        
        # Create embeddings with progress tracking
        embeddings = self.create_embeddings_with_progress(augmented_texts)
        
        # Create and save vector store using the SAME model that created the embeddings
        print("Creating vector store...")
        
        # Create the HuggingFaceEmbeddings object with the model we actually used
        hf_embeddings = HuggingFaceEmbeddings(model_name=self.used_model_name)
        
        # Create the vector store
        vector_store = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings)),  # Use original texts for storage
            embedding=hf_embeddings,  # Use the same model that generated the embeddings
            metadatas=metadatas
        )
        
        # Save vector store
        print("Saving vector store...")
        vector_store.save_local(self.vector_store_file)
        
        # Save metadata
        print("Saving metadata...")
        with open(self.metadata_file, 'w') as f:
            json.dump({
                'chunk_metadata': metadatas,
                'processing_date': datetime.now().isoformat(),
                'total_chunks': len(texts),
                'total_documents': len(pdf_files),
                'embedding_model': self.used_model_name  # Save the model we actually used
            }, f, indent=2)
        
        # Save the model name to a separate file for the application to use
        with open(os.path.join(self.output_folder, "embedding_model.txt"), 'w') as f:
            f.write(self.used_model_name)
        
        print("Processing complete!")
        print(f"Vector store saved to: {self.vector_store_file}")
        print(f"Metadata saved to: {self.metadata_file}")
        print(f"Used embedding model: {self.used_model_name}")

def main():
    processor = DocumentProcessor()
    processor.process_documents()

if __name__ == "__main__":
    main() 