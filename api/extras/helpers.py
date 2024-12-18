import os
from fastapi import UploadFile
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from grobid_client.grobid_client import GrobidClient
import xml.etree.ElementTree as ET
import json

from uuid import uuid4

async def process_pdfs(files, qclient_, collection_name, emb_model, upload_dir_):
    """
    Process uploaded PDF files, extracting text, splitting into chunks, embedding, and upserting to Qdrant.

    Args:
        files (list[UploadFile]): List of PDF files to process.
        qclient_ (QdrantClient): Qdrant client instance for vector database operations.
        collection_name (str): Name of the Qdrant collection to upsert points into.
        emb_model (EmbeddingModel): Model instance for generating embeddings.
        upload_dir_ (str): Directory path to save the uploaded PDF files.

    Returns:
        dict: Information about uploaded files, including chunk count and upsert response.
    """
    uploaded_files_info = {}

    for file in files:
        # Save the uploaded PDF file
        file_path = os.path.join(upload_dir_, file.filename)
        os.makedirs(upload_dir_, exist_ok=True)  # Ensure the directory exists

        # Check if the file already exists
        if os.path.exists(file_path):
            print(f"Skipping {file.filename}: already exists in {upload_dir_}.")
            continue  # Skip to the next file if the PDF already exists

        with open(file_path, "wb") as f:
            f.write(await file.read())
        print(f"PDF saved to: {file_path}")

        # Extract text from the PDF
        pdf_text = extract_text_from_pdf(file_path)

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2100, chunk_overlap=210)
        text_chunks = text_splitter.split_text(pdf_text)

        # Generate embeddings for the text chunks
        embeddings = emb_model.embed_documents(text_chunks)

        pdf_id = file.filename
        points = [
            models.PointStruct(
                id=str(uuid4()),  # Ensure unique ID by using a UUID
                payload={
                    
                    'metadata':{'pdf_id': pdf_id, "text": text_chunks[i],}                },
                vector=embedding # Ensure this is a list
            )
            for i, embedding in enumerate(embeddings)
        ]

        # Upsert points into Qdrant collection
        upsert_response = qclient_.upsert(collection_name=collection_name, points=points)

        # Store the response information
        uploaded_files_info[file.filename] = {
            "file_name": file.filename,
            "total_chunks": len(text_chunks),
            "upsert_response": upsert_response
        }

    return uploaded_files_info


def extract_text_from_pdf(file_path) -> str:
    pdf_document = fitz.open(file_path)
    text = ""
    for page in pdf_document:
        text += page.get_text()
    pdf_document.close()
    return text


# Function to remove namespace from XML tags
def strip_namespace(tag):
    return tag.split('}')[-1]  # Get everything after the '}' character

# Function to convert XML elements to a nested string representation
def xml_to_string(element, indent=0):
    result = ""
    # Process child elements
    for child in element:
        child_tag = strip_namespace(child.tag)
        child_string = xml_to_string(child, indent + 2)  # Recursively call for child elements
        
        # If the element has text, add it directly without the 'text' key
        if child.text and child.text.strip():
            result += ' ' * indent + f"{child_tag}: {child.text.strip()}\n"
        
        # If the child has sub-elements, include the constructed string
        if child_string:
            result += ' ' * indent + f"{child_tag}:\n{child_string}"

    cleaned_result = result.replace('\n', ' ')
    cleaned_string = ' '.join(cleaned_result.split())  # This removes extra spaces

        
    return cleaned_string



async def process_pdf_files_texts(files, qclient_, gclient_, collection_name, emb_model, upload_dir_, metas_dir_):
    uploaded_files_info = {}

    for file in files:
        # Define the paths for the PDF and XML files
        pdf_file_path = os.path.join(upload_dir_, file.filename)
        xml_file_path = os.path.join(metas_dir_, f'{file.filename.split(".")[0]}.grobid.xml')

        # Step 1: Check if the PDF file already exists
        if os.path.exists(pdf_file_path):
            print(f"Found {file.filename}: PDF already exists in {upload_dir_}.")

            # Check if the corresponding XML file exists
            if os.path.exists(xml_file_path):
                print(f"Skipping {file.filename}: corresponding XML already exists in {metas_dir_}.")
                continue  # Skip processing if both PDF and XML exist
        else:
            # Step 2: Save the uploaded PDF file since it doesn't exist
            with open(pdf_file_path, "wb") as f:
                f.write(await file.read())
            print(f"PDF saved to: {pdf_file_path}")

        # Step 3: Create XML from the PDF using GROBID if the XML doesn't exist
        if not os.path.exists(xml_file_path):
            pdf_f, status, pdf_t = gclient_.process_pdf(
                "processFulltextDocument",
                pdf_file_path,
                generateIDs=False,
                consolidate_header=True,
                consolidate_citations=True,
                include_raw_citations=True,
                include_raw_affiliations=True,
                tei_coordinates=True,
                segment_sentences=False,
            )

            # Save the XML to the 'metas' directory
            with open(xml_file_path, 'w', encoding='utf-8') as xml_file:
                xml_file.write(pdf_t)
            print(f"XML saved to: {xml_file_path}")

        # Step 4: Read the XML and get the string representation
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        xml_string = xml_to_string(root)  # Ensure xml_to_string is defined

        # Step 5: Perform text splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
        text_chunks = text_splitter.split_text(xml_string)

        # Step 6: Generate embeddings for the text chunks
        embeddings = emb_model.embed_documents(text_chunks)

        # Step 7: Upsert points into Qdrant collection
        points = [
            models.PointStruct(
                id=str(uuid4()),
                payload={
                    "pdf_id": file.filename,
                    "text": text_chunks[i]
                },
                vector=embedding  # Ensure this is a list
            )
            for i, embedding in enumerate(embeddings)
        ]

        upsert_response = qclient_.upsert(collection_name=collection_name, points=points)

        uploaded_files_info[file.filename] = {
            "file_name": file.filename,
            "total_chunks": len(text_chunks),
            "upsert_response": upsert_response
        }

    return uploaded_files_info