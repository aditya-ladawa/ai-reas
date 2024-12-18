import os
import fitz
import re
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.store.postgres.aio import PoolConfig



def extract_text_until_introduction(text):
    """
    Extracts text from the beginning of a document up to the last encountered 'Introduction' section.

    Args:
        text (str): The full text of the document.

    Returns:
        str: The text up to the last 'Introduction' section (matching various formats),
            or the full text if no 'Introduction' is found.
    """
    # Define a regex pattern to match 'Introduction' with possible prefixes
    pattern = r"(?i)(?:(?:^|\n)(?:\d+\.\s*|\(\d+\)\s*|I{1,3}\.\s*)?)Introduction"

    # Find all matches and their positions
    matches = list(re.finditer(pattern, text))

    if matches:
        # Get the last match's start position
        last_match = matches[-1]
        return text[:last_match.start()]

    # If no matches, return the full text
    return text

def extract_paper_metadata_with_chain(chain, files):
    """
    Process a list of PDF files, extract metadata using the given chain, and handle errors gracefully.

    Args:
        llm: The language model used for metadata extraction.
        chain: The chain to invoke for metadata extraction.
        files: A list of file paths to process (from FastAPI or similar sources).

    Returns:
        dict: A dictionary where keys are file names and values are either metadata results or error messages.
    """



    results = {}

    for file_path in files:
        try:
            # Extract the file name
            file_name = os.path.basename(file_path)
            print(f"Processing file: {file_name}")

            # Open the PDF document
            pdf_document = fitz.open(file_path)

            # Extract text from the PDF
            text = ""
            for page in pdf_document:
                text += page.get_text()
            pdf_document.close()

            # Extract text until the 'Introduction' section
            extracted_text = extract_text_until_introduction(text)

            # Invoke the chain to extract metadata
            result = chain.invoke({"content": extracted_text})

            # Store the result
            results[file_name] = result

        except Exception as e:
            # Handle any errors during processing
            error_message = f"Error processing {file_path}: {str(e)}"
            print(error_message)
            results[file_name] = {"error": error_message}

    return results


def make_json_serializable(data):
    """Ensure data is JSON-serializable."""
    if isinstance(data, (dict, list, str, int, float, bool)) or data is None:
        return data
    elif isinstance(data, set):
        return list(data)  # Convert sets to lists
    elif isinstance(data, bytes):
        return data.decode()  # Convert bytes to string
    else:
        return str(data)  # Fallback: Convert other types to string

def ensure_serializable_dict(data):
    """Recursively ensure all dictionary values are JSON-serializable."""
    if isinstance(data, dict):
        return {k: ensure_serializable_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [ensure_serializable_dict(item) for item in data]
    return make_json_serializable(data)


DB_URI_CHECKPOINTER = "postgresql://adi:root@localhost:5432/chat_memory?sslmode=disable"
DB_URI_STORE = "postgresql://adi:root@localhost:5432/chat_store?sslmode=disable"


connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

store_pool_config = PoolConfig(min_size=5,max_size=20)

async def get_checkpointer():
    async with AsyncConnectionPool(conninfo=DB_URI_CHECKPOINTER, max_size=20, kwargs=connection_kwargs) as pool:
        checkpointer = AsyncPostgresSaver(pool)

        await checkpointer.setup()

        return checkpointer



# async def get_store():
#     async with AsyncPostgresStore.from_conn_string(conn_string=DB_URI_STORE, pool_config=store_pool_config) as store:
#         await store.setup()

#         return store



# # Correct way to call get_checkpointer() inside an async function
# async def main():
#     checkpointer = await get_checkpointer()
#     print(checkpointer)

# # Running the async function with asyncio
# import asyncio
# asyncio.run(main())
