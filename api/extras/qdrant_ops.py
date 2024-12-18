from qdrant_client import QdrantClient
from qdrant_client.http import models

def connect_to_qdrant():
    try:
        connection = QdrantClient("http://localhost:6333")

        if connection:
            print('Started Qdrant client.')

        # Check if the collection already exists
        existing_collections = connection.get_collections().collections
        if "aireas-local" not in [collection.name for collection in existing_collections]:
            connection.create_collection(
                collection_name="aireas-local",
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
            )
            print("Collection 'aireas-local' created successfully.")
            
            # Add index to the 'pdf_id' field in payload
            connection.create_payload_index(
                collection_name="aireas-local",
                field_name="pdf_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            print("Index created on 'pdf_id' field for efficient filtering.")
            
        else:
            print("Collection 'aireas-local' already exists.")
            
            # Check if 'pdf_id' index exists, add if missing
            collection_info = connection.get_collection(collection_name="aireas-local")
            if "pdf_id" not in collection_info.payload_schema:
                connection.create_payload_index(
                    collection_name="aireas-local",
                    field_name="pdf_id",
                    field_schema="keyword"
                )
                print("Index created on 'pdf_id' field for efficient filtering.")
            else:
                print("Index on 'pdf_id' field already exists.")

        return connection
    except Exception as e:
        print(f"Connection error: {e}")
        return None