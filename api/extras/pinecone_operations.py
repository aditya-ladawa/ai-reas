# api/pinecone_operations.py

import os
from pinecone import Pinecone, ServerlessSpec  # Import the new Pinecone class
from dotenv import load_dotenv

class PineconeOperations:

    def __init__(self, dimension=768, index_name="aireas-index"):
        load_dotenv()  # Load environment variables

        api_key = os.getenv('PINECONE_API_KEY')  # Update to use the correct env var
        self.pinecone = Pinecone(api_key=api_key)  # Create an instance of the Pinecone class
        self.index = None
        self.connect_index(dimension, index_name)  # Connect to the Pinecone index

    def create_index(self, index_name, dimension) -> list:
        indexes = self.pinecone.list_indexes().names()  # Use the new method to list indexes
        if index_name not in indexes:
            self.pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud='aws', region='us-east-1')  # Adjust according to your needs
            )
        return indexes

    def connect_index(self, dimension, index_name):
        self.create_index(index_name, dimension)
        self.index = self.pinecone.Index(index_name)  # Connect to the created index

    def upsert_pdf_vectors(self, pdf_id, filename, vectors):
        # Prepare data for upsert, each vector needs an ID
        data = [{"id": f"{pdf_id}_{i}", "values": vector} for i, vector in enumerate(vectors)]
        return self.index.upsert(vectors=data, namespace=filename.split('.')[0])


    # Other methods can be added here as needed
