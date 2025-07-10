import os
import fitz
import re
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.store.postgres.aio import PoolConfig

# from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig

# from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph

from typing_extensions import Annotated, Optional
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, trim_messages


from .token_counter import tiktoken_counter


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


async def prepare_system_message(state, config: RunnableConfig, store: BaseStore):
    user_id = config.get("configurable", {}).get("user_id")
    conversation_id = config.get("configurable", {}).get("thread_id")
    
    user_data_item = await store.aget(("users",), user_id)
    user_data = user_data_item.value if user_data_item else {}

    conversation_metadata = await store.asearch(("conversation_metadata", user_id, conversation_id))
    metadata_info = "\n".join(
        f"{item.key.replace('metadata_', '')} | "
        f"{item.value.get('title', 'Unknown')} | "
        f"{', '.join(item.value.get('authors', []))} | "
        f"{item.value.get('description', 'Unknown')}"
        for item in conversation_metadata
    )

    conversation_memories = await store.asearch(("conversation_memory", user_id, conversation_id))
    memory_info = "\n".join(
        f"{item.value}" if item.value is not None else "Unknown"
        for item in conversation_memories
    )

    conversation_info_item = await store.aget(("conversation_info", user_id), conversation_id)
    conversation_info = conversation_info_item.value if conversation_info_item else {}

    user_info = f"Name: {user_data.get('name', 'Unknown')}, Email: {user_data.get('email', 'Unknown')}."
    conv_info = f"Topic: {conversation_info.get('topic', 'Unknown')}, Conversation Name: {conversation_info.get('conversation_name', 'Unknown')}."
    
    prompt = "You are a helpful research assistant. Help user to the best of your abilities. Provide concise but accurate and up to point answers. As of now you have these tools in your arsenal: qdrant_retriever_tool (content retrieval from vector database), arxiv_search_tool (search research papers), tavily_search tool (internet search). If you do not know the answer, then simply say 'I don't know. If you need clarification on what exactly user wants, then ask the user again. If you know the answer to user's query then answer yourself, else you can also rely on tools you have.\n"

    msg = f"User Info: {user_info}\nConversation Info: {conv_info}\nFiles in vector DB and storage:\n{metadata_info}, Conversation memories: {memory_info}"

    system_msg = prompt + msg

    trimmed_mesgs = trim_messages(
        messages=state['messages'],
        max_tokens=100000,
        strategy="last",
        token_counter=tiktoken_counter,
        include_system=True,
        allow_partial=False,
    )

    return [{"role": "system", "content": system_msg}] + trimmed_mesgs



# def prepare_system_message(state: StateGraph, config: RunnableConfig, store: BaseStore):
#     user_id = config.get("configurable", {}).get("user_id")
#     conversation_id = config.get("configurable", {}).get("thread_id")
    
#     # Synchronous version of 'aget' -> 'get'
#     user_data_item = store.get(("users",), user_id)
#     user_data = user_data_item.value if user_data_item else {}

#     # Synchronous version of 'asearch' -> 'search'
#     conversation_metadata = store.search(("conversation_metadata", user_id, conversation_id))
#     metadata_info = "\n".join(
#         f"{item.key.replace('metadata_', '')} | "
#         f"{item.value.get('title', 'Unknown')} | "
#         f"{', '.join(item.value.get('authors', []))} | "
#         f"{item.value.get('description', 'Unknown')}"
#         for item in conversation_metadata
#     )

#     conversation_memories = store.search(("conversation_memory", user_id, conversation_id))
#     memory_info = "\n".join(
#         f"{item.value}" if item.value is not None else "Unknown"
#         for item in conversation_memories
#     )

#     conversation_info_item = store.get(("conversation_info", user_id), conversation_id)
#     conversation_info = conversation_info_item.value if conversation_info_item else {}

#     # Construct user and conversation info
#     user_info = f"Name: {user_data.get('name', 'Unknown')}, Email: {user_data.get('email', 'Unknown')}."
#     conv_info = f"Topic: {conversation_info.get('topic', 'Unknown')}, Conversation Name: {conversation_info.get('conversation_name', 'Unknown')}."
    
#     # Define the prompt and messages
#     prompt = "You are a helpful research assistant. Help user to the best of your abilities. Provide concise but accurate and up to point answers. As of now you have these tools in your arsenal: qdrant_retriever_tool (content retrieval from vector database), arxiv_search_tool (search research papers), tavily_search tool (internet search). If you do not know the answer, then simply say 'I don't know. If you need clarification on what exactly user wants, then ask the user again. If you know the answer to user's query then answer yourself, else you can also rely on tools.\n"

#     msg = f"User Info: {user_info}\nConversation Info: {conv_info}\nFiles in vector DB and storage:\n{metadata_info}, Conversation memories: {memory_info}"

#     # Combine the prompt with the message and return it
#     system_msg = prompt + msg

#     return [{"role": "system", "content": system_msg}] + state['messages']


# async def get_checkpointer():
#     async with AsyncConnectionPool(conninfo=DB_URI_CHECKPOINTER, max_size=20, kwargs=connection_kwargs) as pool:
#         checkpointer = AsyncPostgresSaver(pool)

#         await checkpointer.setup()

#         return checkpointer



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
