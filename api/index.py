from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Response, Request, Query, status, WebSocket, WebSocketDisconnect

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from uuid import uuid4
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage

from typing import List

from api.qdrant_cloud_ops import process_pdfs, qclient_, EMBEDDING_MODEL, delete_file_from_qdrant

# sql_ops imports
from api.sql_ops import init_db, create_user, get_user_by_email, verify_password, generate_jwt_token, validate_password_strength, get_user_by_id
from fastapi.responses import JSONResponse
from api.pydantic_models import *

from api.chat_handlers import assign_chat_topic_chain, llm, react_agent, metadata_extraction_chain

from fastapi.security import OAuth2PasswordBearer

import jwt
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError
from jwt import decode

from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timedelta


from sqlalchemy.exc import SQLAlchemyError

import re

import traceback


from api.redis_ops import add_conversation, initialize_redis, close_redis_connection, fetch_user_conversations, fetch_conversation, update_conversation_files, delete_file_from_redis, delete_conversation_from_redis

from fastapi.responses import FileResponse


from api.store_memory_ops import get_checkpointer, extract_paper_metadata_with_chain, DB_URI_CHECKPOINTER, DB_URI_STORE, store_pool_config, connection_kwargs, make_json_serializable, ensure_serializable_dict
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore

from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.store.postgres.aio import PoolConfig


from api.file_ops import delete_file_from_storage

from typing_extensions import Optional

load_dotenv()

# Services
SECRET_KEY = os.environ.get('JWT_SECRET_KEY')
ALGORITHM = "HS256"
EMBEDDING_MODEL = EMBEDDING_MODEL
COLLECTION_NAME = 'aireas-cloud'
qdrant_client = qclient_
APIS = os.path.join(os.getcwd(), 'api')



@asynccontextmanager
async def lifespan(app: FastAPI):

    await init_db()
    print('\nStarted SQL db')


    await initialize_redis()
    print('\nStarted Redis connection')


    checkpointer = await get_checkpointer()
    app.state.checkpointer = checkpointer
    print('\nInitialized Postgres Checkpointer\n')

    async with AsyncPostgresStore.from_conn_string(conn_string=DB_URI_STORE, pool_config=store_pool_config) as store:
        await store.setup()
        app.state.store = store

        yield {'app.state.store': app.state.store}

    yield


    await close_redis_connection()

# Initialize FastAPI
app = FastAPI(docs_url="/api/docs", openapi_url="/api/openapi.json", debug=True, lifespan=lifespan)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://localhost:3000"],  # Update as per your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Custom handler for HTTP exceptions to structure the error response.
    """
    # Provide detailed error information for debugging
    error_detail = {
        "detail": exc.detail,
        "status_code": exc.status_code,
        "message": "An error occurred during the request.",
    }

    # Include stack trace if in debug mode
    if app.debug:
        error_detail["stack_trace"] = traceback.format_exc()

    return JSONResponse(
        status_code=exc.status_code,
        content=error_detail,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Custom handler for uncaught exceptions.
    """
    error_detail = {
        "detail": str(exc),
        "status_code": 500,
        "message": "Internal server error occurred.",
    }

    # Include stack trace if in debug mode
    if app.debug:
        error_detail["stack_trace"] = traceback.format_exc()

    return JSONResponse(
        status_code=500,
        content=error_detail,
    )

# # Define directories for file uploads
# static_dir = Path("api/static/")
# static_dir.mkdir(exist_ok=True)

# metas_dir = "api/metas/"
# os.makedirs(metas_dir, exist_ok=True)

# # Mount static files
# app.mount("/static", StaticFiles(directory=static_dir), name="static")


# # Initialize the sql database at startup
# @app.on_event("startup")
# async def sql_startup_event():
#     await init_db()
#     print('\nStarted SQL db')

# # Initialize the redis database at startup
# @app.on_event("startup")
# async def redis_startup_event():
#     await initialize_redis()

# @app.on_event("shutdown")
# async def redis_shutdown_event():
#     await close_redis_connection()




async def get_psql_checkpointer():
    return app.state.checkpointer

async def get_psql_store():
    return app.state.store

def get_authenticated_user(request: Request):
    token = request.cookies.get("auth_token")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    try:
        payload = decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        exp = payload.get("exp")
        
        if exp and datetime.utcfromtimestamp(exp) < datetime.utcnow():
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")

        return {
            "user_id": payload.get("user_id"),
            "email": payload.get("email"),
        }

    except ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except DecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Malformed token")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error: {str(e)}")


@app.post("/api/upload/{conversation_id}")
async def upload_files(conversation_id: str, files: List[UploadFile] = File(...), current_user: dict = Depends(get_authenticated_user), store = Depends(get_psql_store)):
    """
    Upload and process PDF files for the current authenticated user.
    """
    try:
        user_id = current_user.get("user_id")
        email = current_user.get("email")  # Assuming email is in the user dict

        if not user_id or not email:
            raise HTTPException(status_code=401, detail="User authentication failed.")

        # Define the user's directory
        user_storage_base = os.path.join(APIS, "users_storage")
        user_dir = os.path.join(user_storage_base, user_id)
        conversation_dir = os.path.join(user_dir, conversation_id)

        os.makedirs(conversation_dir, exist_ok=True)

        # Process uploaded files
        result = await process_pdfs(
            files=files,
            qclient_=qclient_,
            collection_name="aireas-cloud",
            emb_model=EMBEDDING_MODEL,
            user_id=user_id,
            email=email,
            conversation_dir=conversation_dir,
            conversation_id=conversation_id
        )

        if result and "uploaded_files" in result and result["uploaded_files"]:
            uploaded_files_info = result["uploaded_files"]
            file_paths = [file_info["file_path"] for file_info in uploaded_files_info.values()]

            try:
                # Extract metadata and update the store
                metadata_results = extract_paper_metadata_with_chain(metadata_extraction_chain, file_paths)
                for k, v in metadata_results.items():
                    serializable_value = ensure_serializable_dict(v)
                    await store.aput(
                        namespace=("conversation_metadata", user_id, conversation_id),
                        key=f"metadata_{k}",
                        value=serializable_value
                    )

                # Update Redis DB
                update_response = await update_conversation_files(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    uploaded_files=uploaded_files_info
                )

                return {
                    "message": "Files uploaded and metadata extracted successfully.",
                    "details": uploaded_files_info,
                    "metadata": metadata_results
                }

            except Exception as e:
                # Clean up files if an error occurs
                for file_info in uploaded_files_info.values():
                    file_path = file_info["file_path"]
                    if os.path.exists(file_path):
                        os.remove(file_path)
                raise HTTPException(status_code=500, detail=f"Error during post-upload processing: {str(e)}")

        else:
            raise HTTPException(status_code=400, detail="No files were processed successfully.")

    except HTTPException as e:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.get("/api/get_uploaded_files/{conversation_id}")
async def get_uploaded_files(conversation_id: str, current_user: dict = Depends(get_authenticated_user)):
    """
    Retrieve uploaded files for a specific conversation and the current authenticated user.
    """
    try:
        # Retrieve user-specific details
        user_id = current_user.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="User authentication failed.")

        # Define the conversation-specific directory
        user_storage_base = os.path.join(APIS, "users_storage")
        user_dir = os.path.join(user_storage_base, user_id)
        conversation_dir = os.path.join(user_dir, conversation_id)

        # Check if the directory exists
        if not os.path.exists(conversation_dir):
            return JSONResponse(content={"files": [], "message": "No files found for this conversation."}, status_code=200)

        # Retrieve all files within the conversation directory
        files = [
            file for file in os.listdir(conversation_dir) if os.path.isfile(os.path.join(conversation_dir, file))
        ]

        if not files:
            return JSONResponse(content={"files": [], "message": "No files found for this conversation."}, status_code=200)

        # Prepare a detailed response with file paths
        file_details = [
            {"file_name": file, "file_path": os.path.join(conversation_dir, file)} for file in files
        ]

        return JSONResponse(content={"files": file_details, "message": "Files retrieved successfully."}, status_code=200)

    except HTTPException as e:
        print(f"HTTPException: {e.detail}")
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


# @app.post('/api/retrieve')
# def retrieve(query_request: QueryRequest, store: Depends(get_psql_store)):
#     """Retrieves relevant PDF information based on the query."""
#     try:
#         # Get the embeddings for the query
#         query_embeddings = EMBEDDING_MODEL.embed_query(query_request.query)

#         # Query points from Qdrant
#         search_result = qdrant_client.query_points(
#             collection_name=COLLECTION_NAME,
#             query=query_embeddings,
#             with_payload=True,
#             limit=query_request.top_k,
#         )

#         # Extracting necessary details
#         results = []
#         if hasattr(search_result, 'points'):
#             for point in search_result.points:  # Access the points attribute
#                 results.append({
#                     "id": point.id,
#                     "score": point.score,
#                     "pdf_name": point.payload.get('pdf_name', 'N/A'),  # Use get to avoid KeyError
#                     "text": point.payload.get('text', 'N/A'),
#                 })

#         return {"points": results}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/signup", status_code=201)
async def signup(user: UserCreate, store = Depends(get_psql_store)):
    existing_user = await get_user_by_email(user.email.lower())
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="This email is already registered. Please log in.",
        )

    # Validate password strength
    if not validate_password_strength(user.password):
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least 8 characters, including an uppercase letter, a number, and a special character.",
        )

    try:
        new_user = await create_user(
            user_name=user.name.strip(),
            raw_password=user.password,
            email=user.email.strip().lower(),
        )

        signed_up_user_id = new_user.user_id
        signed_up_user_name = new_user.user_name
        signed_up_user_email = new_user.email

        await store.aput(
            namespace=('users'),
            key=signed_up_user_id,
            value={'name': signed_up_user_name, 'email': signed_up_user_email}
        )

        return JSONResponse(
            content={
                "message": f"User {new_user.user_name} successfully registered.",
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during signup. Please try again later. {e}",
        )


@app.post("/api/login", status_code=200)
async def login(user: UserLogin, response: Response):
    try:
        db_user = await get_user_by_email(user.email.strip().lower())
        if db_user is None or not verify_password(user.password, db_user.password):
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        token = generate_jwt_token(user_id=db_user.user_id, email=db_user.email)

        response.set_cookie(
            key="auth_token",
            value=token,
            httponly=True,
            secure=False,
            samesite='Strict'
        )

        return {"message": "Login successful"}

    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Database error")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.get('/api/get_auth_user')
async def get_active_user(current_user: dict = Depends(get_authenticated_user)):
    return {'user': current_user}


@app.post("/api/logout")
async def logout(response: Response):
    try:
        response.set_cookie(
            key="auth_token",
            value="", 
            expires="Thu, 01 Jan 1994 00:00:00 GMT",
            max_age=0, 
            httponly=True,
            secure=False,
            samesite="Strict",
            path="/",
        )
        return {"message": "Logout successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during logout: {str(e)}")


@app.post('/api/add_conversation')
async def add_conversation_route(request: AssignTopic, current_user: dict = Depends(get_authenticated_user), store = Depends(get_psql_store)):
    """
    API route to add a conversation.
    """
    try:
        
        # Extract user details from the authenticated user
        user_id = current_user.get("user_id")
        email = current_user.get("email")

        if not user_id or not email:
            raise HTTPException(status_code=400, detail="User ID or email is missing from the request.")

        # Extract conversation details from the request body
        name = request.conversation_name
        description = request.conversation_description

        if not name.strip() or not description.strip():
            raise HTTPException(status_code=400, detail="Both conversation_name and conversation_description are required.")

        assigned_topic = assign_chat_topic_chain.invoke(description)

        # Add conversation to Redis
        result = await add_conversation(user_id, email, name, description, assigned_topic)

        await store.aput(namespace=('conversation_info', result['user_id']), key=result['conversation_id'], value={'conversation_name': name, 'topic': result['conversation_data']['topic']})


        conversation_info = await store.aget(namespace=('conversation_info', result['user_id']), key=result['conversation_id'])
        # Return success message
        return {
            "message": "Conversation added successfully",
            "conversation_id": result["conversation_id"],
            "assigned_topic": assigned_topic,
            'store': conversation_info.value,
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get('/api/fetch_conversations')
async def fetch_conversations(current_user: dict = Depends(get_authenticated_user)):
    user_id = current_user.get("user_id")
    user_email = current_user.get("email")

    if not user_id or not user_email:
        raise HTTPException(status_code=400, detail="Invalid user details.")

    conversations = await fetch_user_conversations(user_id)

    if not conversations:
        return {"message": "No conversations found.", "conversations": []}

    return {"conversations": conversations}


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str,  current_user: str = Depends(get_authenticated_user)):
    user_id = current_user.get("user_id")
    return await fetch_conversation(user_id=user_id, conversation_id=conversation_id)


@app.get("/api/view_pdf/{conversation_id}/{file_name}")
async def view_pdf(conversation_id: str, file_name: str, current_user: dict = Depends(get_authenticated_user)):
    user_id = current_user.get('user_id')
    selected_file_path = os.path.join(APIS, 'users_storage', user_id, conversation_id, file_name)

    if not os.path.exists(selected_file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Use FileResponse and specify headers for inline display
    return FileResponse(
        selected_file_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{file_name}"'}
    )


async def get_authenticated_user_websocket(websocket: WebSocket):
    token = websocket.cookies.get("auth_token")

    if not token:
        raise HTTPException(status_code=401, detail="Authorization token missing")

    try:
        payload = decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        exp = payload.get("exp")
        if exp and datetime.utcfromtimestamp(exp) < datetime.utcnow():
            raise HTTPException(status_code=401, detail="Token has expired")

        return {
            "user_id": payload.get("user_id"),
            "email": payload.get("email"),
        }

    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.websocket("/api/llm_chat/{conversation_id}")
async def websocket_llm_chat(conversation_id: str, websocket: WebSocket, current_user: dict = Depends(get_authenticated_user_websocket)):
    user_id = current_user.get('user_id')
    config = {"configurable": {'user_id': user_id ,"thread_id": conversation_id}}
    await websocket.accept()
    try:
        await websocket.send_text("Connected to LLM WebSocket! Start sending your queries.")

        seen_tool_calls = set()

        while True:
            user_query = await websocket.receive_text()

            query = {'messages': [HumanMessage(content=user_query)]}

            async for event in react_agent.astream(query, stream_mode='values', config=config):
                if 'messages' not in event:
                    continue

                for msg in event['messages']:
                    if isinstance(msg, HumanMessage):
                        continue

                    elif isinstance(msg, AIMessage) and not msg.content:
                        tool_calls = msg.additional_kwargs['tool_calls']

                        for tool_call in tool_calls:
                            tool_name = tool_call['function']['name']
                            args = tool_call['function']['arguments']

                            tool_call_id = (tool_name, str(args))
                            if tool_call_id not in seen_tool_calls:
                                seen_tool_calls.add(tool_call_id)
                                await websocket.send_text(f"Calling tool: {tool_name}\nTool arguments: {args}")

                    elif isinstance(msg, AIMessage):
                        if msg.content:
                            await websocket.send_text(msg.content)

    except WebSocketDisconnect:
        print("WebSocket connection closed.")


# @app.get('/api/get_conversation_memories_from_store')
# async def get_conversation_memories_from_store(conversation_id, store = Depends(get_psql_store), current_user = Depends(get_authenticated_user)):
#     user_id = current_user.get('user_id')

#     r = await store.asearch(("conversation_metadata", user_id, conversation_id))
    
#     r_data = [item.dict() for item in r]

#     return {'store data': r_data}



# @app.get('/api/get_user_data_from_store')
# async def get_user_data_from_store(store = Depends(get_psql_store), current_user = Depends(get_authenticated_user)):
#     user_id = current_user.get('user_id')

#     p = await store.aget(('users'), key=user_id)
#     p_data = p.value
    
#     return {'user_data': p_data}


@app.get('/api/get_data_from_store')
async def get_data_from_store(store=Depends(get_psql_store), current_user=Depends(get_authenticated_user), conversation_id: Optional[str] = None):
    user_id = current_user.get('user_id')

    try:
        p = await store.aget(('users',), key=user_id)
        if p is None:
            raise HTTPException(
                status_code=404,
                detail="User data not found in the store."
            )
        user_data = p.value

        if conversation_id:
            try:
                r = await store.asearch(("conversation_metadata", user_id, conversation_id))
                r_data = [item.dict() for item in r]

                t = await store.aget(namespace=('conversation_info',), key=conversation_id)
                t_data = t.value if t else None

                return {
                    'user_data': user_data,
                    'conversation_metadata': r_data,
                    'conversation_info': t_data
                }
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error while retrieving conversation data: {str(e)}"
                )

        return {'user_data': user_data}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )


# @app.delete('/api/delete_store_data')
# async def delete_store_data(conversation_id: str, store = Depends(get_psql_store), current_user = Depends(get_authenticated_user)):
#     user_id = current_user.get('user_id')
    
#     namespace = ("conversation_metadata", user_id, conversation_id)
    
#     try:
#         r = await store.asearch(namespace)
        
#         if not r:
#             return {'message': 'No items found in the namespace to delete'}
        
#         deleted_items_count = 0
#         for item in r:
#             key = item.key if hasattr(item, 'key') else None  # Use .key if item has it
#             if key:
#                 await store.adelete(namespace=namespace, key=key)
#                 deleted_items_count += 1
        
#         # Return a response with the number of deleted items
#         return {'message': f'{deleted_items_count} item(s) deleted from the namespace successfully'}

#     except Exception as e:
#         # Handle any exceptions that occur during the process
#         return {'error': f'An error occurred: {str(e)}'}



@app.delete('/api/purge_files')
async def purge_file(file_name: str, conversation_id: str, store = Depends(get_psql_store), current_user = Depends(get_authenticated_user)):
    try:
        user_id = current_user.get('user_id')
        file_name = file_name + '.pdf'
        file_name = file_name.lower()

        del_msg = ''

        # Step 1: Delete from storage (File System)
        file_path = os.path.join(APIS, 'users_storage', user_id, conversation_id, file_name) 
        try:
            await delete_file_from_storage(file_path)
            del_msg += '\ndeleted from File storage'
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting file from storage: {str(e)}")


        # Step 2: Delete from PostgreSQL store (Metadata)
        try:
            await store.adelete(
                namespace=("conversation_metadata", user_id, conversation_id),
                key=f"metadata_{file_name}",
            )
            del_msg += '\ndeleted from Conversation Memory storage'
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting metadata from store: {str(e)}")


        # Step 3: Delete from Redis
        try:
            await delete_file_from_redis(user_id, conversation_id, file_name)
            del_msg += '\ndeleted from Redis storage'
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting file from Redis: {str(e)}")


        # Step 4: Delete from Qdrant
        try:
            await delete_file_from_qdrant(qclient_= qclient_,file_name=file_name)
            del_msg += '\ndeleted from Qdrant Vector db'
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting file from Qdrant: {str(e)}")


        return {"message": f"File {file_name} purged successfully: {del_msg} "}

    except HTTPException as http_error:
        raise http_error

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# @app.delete('/api/delete_conversation_info')
# async def delete_conversation_info(conversation_id: str, store=Depends(get_psql_store), current_user=Depends(get_authenticated_user)):
#     user_id = current_user.get('user_id')
#     namespace = ('conversation_info', user_id)

#     try:
#         infos = await store.asearch(namespace)

#         if not infos:
#             return {'message': "No items to delete"}


#         deleted_items_count = 0
#         for item in infos:
#             key = item.key if hasattr(item, 'key') else None  # Use .key if item has it
#             if key:
#                 await store.adelete(namespace=namespace, key=conversation_id)
#                 deleted_items_count += 1
        
#         return {'message': f'{deleted_items_count} item(s) deleted from the namespace successfully'}

#     except Exception as e:
#         return {'error': f'An error occurred: {str(e)}'}



@app.delete('/api/delete_conversation')
async def delete_conversation(conversation_id: str, store=Depends(get_psql_store), current_user=Depends(get_authenticated_user)):
    """
    Deletes all data related to a given conversation ID from storage, metadata store, and Redis.
    Checks if the conversation exists in Redis, local file system, and store before attempting deletion.
    """
    user_id = current_user.get('user_id')
    del_msg = ''
    final_msg = {"storage": "", "metadata_in_store": "", "conversation_info_in_store": "", "redis": ""}

    try:
        # Step 1: Check if the conversation exists in Redis, File storage, and Store (Conversation info)
        try:
            await fetch_conversation(user_id, conversation_id)
            final_msg["redis"] = "Conversation found in Redis."
        except ValueError as e:
            final_msg["redis"] = f"Conversation not found in Redis: {str(e)}"

        # Check File storage
        user_storage_path = os.path.join(APIS, 'users_storage', user_id, conversation_id)
        if os.path.exists(user_storage_path) and os.listdir(user_storage_path):
            final_msg["storage"] = "Conversation found in File storage."
        else:
            final_msg["storage"] = "Conversation not found in File storage."

        # Check metadata in PostgreSQL store
        metadata_keys = await store.asearch(("conversation_metadata", user_id, conversation_id))
        metadata_keys = [key for key in metadata_keys if key.startswith("metadata_")]
        final_msg["metadata_in_store"] = "Metadata found in Conversation Memory storage." if metadata_keys else "No metadata found."

        # Check conversation info in store
        conversation = await store.aget(('conversation_info', user_id), key=conversation_id)
        final_msg["conversation_info_in_store"] = "Conversation info found in Memory." if conversation else "Conversation info not found"

        # If conversation not found in any of the sources, return early
        if final_msg["redis"] == "Conversation not found in Redis." and \
           final_msg["storage"] == "Conversation not found in File storage." and \
           final_msg["metadata_in_store"] == "No metadata found." and \
           final_msg["conversation_info_in_store"] == "Conversation info not found":
            return {"message": f"Conversation {conversation_id} not found in any of the sources. No data to delete."}

        # Step 2: Delete files from storage (File System) and Qdrant
        if final_msg["storage"] == "Conversation found in File storage.":
            try:
                for file_name in os.listdir(user_storage_path):
                    file_path = os.path.join(user_storage_path, file_name)
                    try:
                        await delete_file_from_storage(file_path)
                        del_msg += f"\nDeleted file '{file_name}' from file storage."
                    except Exception as e:
                        del_msg += f"\nError deleting file '{file_name}' from file storage: {str(e)}"
                        
                    try:
                        await delete_file_from_qdrant(qclient_, file_name=file_name)
                        del_msg += f"\nDeleted file '{file_name}' from Qdrant."
                    except Exception as qdrant_error:
                        del_msg += f"\nError deleting file '{file_name}' from Qdrant: {str(qdrant_error)}"

                try:
                    os.rmdir(user_storage_path)  # Remove the conversation directory if empty
                    final_msg["storage"] = "Deleted files from File storage."
                except OSError as e:
                    del_msg += f"\nError deleting empty directory '{user_storage_path}': {str(e)}"
            except Exception as e:
                del_msg += f"\nError during file deletion: {str(e)}"

        # Step 3: Delete metadata from PostgreSQL store
        if final_msg["metadata_in_store"] == "Metadata found in Conversation Memory storage.":
            try:
                for key in metadata_keys:
                    try:
                        await store.adelete(("conversation_metadata", user_id, conversation_id), key)
                        del_msg += f"\nDeleted metadata key '{key}' from Conversation Memory storage."
                    except Exception as e:
                        del_msg += f"\nError deleting metadata key '{key}': {str(e)}"
                final_msg["metadata_in_store"] = "Deleted metadata from Conversation Memory storage."
            except Exception as e:
                del_msg += f"\nError deleting metadata: {str(e)}"

        # Step 4: Delete conversation info from store
        if final_msg["conversation_info_in_store"] == "Conversation info found in Memory.":
            try:
                await store.adelete(('conversation_info', user_id), conversation_id)
                del_msg += f"\nDeleted {conversation_id} from conversation_info."
                final_msg["conversation_info_in_store"] = "Deleted conversation info from store."
            except Exception as e:
                del_msg += f"\nError deleting conversation info: {str(e)}"

        # Step 5: Delete data from Redis
        if final_msg["redis"] == "Conversation found in Redis.":
            try:
                redis_response = await delete_conversation_from_redis(user_id, conversation_id)
                del_msg += f"\n{redis_response['message']}"
                final_msg["redis"] = "Deleted conversation data from Redis."
            except Exception as e:
                del_msg += f"\nError deleting conversation data from Redis: {str(e)}"
                final_msg["redis"] = f"Failed to delete from Redis: {str(e)}"

        # Final summary message
        final_msg_str = f"Final Summary: \n" \
                        f"Storage: {final_msg['storage']}\n" \
                        f"Metadata: {final_msg['metadata_in_store']}\n" \
                        f"Conversation Info: {final_msg['conversation_info_in_store']}\n" \
                        f"Redis: {final_msg['redis']}"

        # return {"message": f"Conversation {conversation_id} purged successfully: {del_msg}\n{final_msg_str}"}
        return {"message": f"{final_msg_str}"}

    except Exception as e:
        return {"message": f"Unexpected error: {str(e)}"}



# @app.delete('/api/delete_user')
# async def delete_user(current_user =  Depends(get_authenticated_user), store = Depends(get_psql_store)):

#     # delete user's storage dir along with conversations

#     # delete vectors from db associated with the user

#     # delete memories from store

#     # delete user and conversations metadata from redis

#     # delete memory from checkpointer

#     # delete from sql db

#     pass




