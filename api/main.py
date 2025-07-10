from fastapi import FastAPI, Depends
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.store.postgres.aio import PoolConfig
from langchain_core.runnables import RunnableConfig
from typing import List
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator
from contextlib import asynccontextmanager


DB_URI_STORE = "postgresql://adi:root@localhost:5432/chat_store?sslmode=disable"
store_pool_config = PoolConfig(min_size=5, max_size=20)

class TheState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]  # Corrected to a direct list

@asynccontextmanager
async def lifespan(app: FastAPI):



    async with AsyncPostgresStore.from_conn_string(conn_string=DB_URI_STORE, pool_config=store_pool_config) as store:
        await store.setup()
        app.state.store = store
        print('\nInitialized Postgres Store\n')
        yield
        del app.state.store




app = FastAPI(lifespan=lifespan)

async def get_psql_store():
    return app.state.store  


# Function to prepare system message from store content
async def prepare_system_message(state, config: RunnableConfig, store: AsyncPostgresStore = Depends(get_psql_store)):
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
    
    prompt = "You are a helpful research assistant. Help user to the best of your abilities. Provide concise but accurate and up to point answers. As of now you have these tools in your arsenal: qdrant_retriever_tool (content retrieval from vector database), arxiv_search_tool (search research papers), tavily_search tool (internet search). If you do not know the answer, then simply say 'I don't know'. If you need clarification on what exactly user wants, then ask the user again. If you know the answer to user's query then answer yourself, else you can also rely on tools you have.\n"

    msg = f"User Info: {user_info}\nConversation Info: {conv_info}\nFiles in vector DB and storage:\n{metadata_info}, Conversation memories: {memory_info}"

    system_msg = prompt + msg 
    return [{"role": "system", "content": system_msg}]

# FastAPI route to fetch and print content from the store
@app.post("/get_conversation_data")
async def get_conversation_data(user_id: str, thread_id: str, store: AsyncPostgresStore = Depends(get_psql_store)):
    # store = app.state.store  # Access the store from app.state
    config = RunnableConfig(configurable={"user_id": user_id, "thread_id": thread_id})

    system_message = await prepare_system_message(TheState, config, store)
    
    return {"message": "Data printed to console", "system_message": system_message[0]["content"]}
