import uuid
from datetime import datetime
import redis.asyncio as redis
import json
from typing_extensions import List

redis_client = None  # Global Redis client for shared use

async def initialize_redis():
    """
    Initialize the Redis connection.
    """
    global redis_client
    if not redis_client:
        redis_client = redis.Redis.from_url(
            "redis://localhost:6379", decode_responses=True,
        )
        print("Redis connection initialized.")


async def close_redis_connection():
    """
    Gracefully close the Redis connection.
    """
    global redis_client
    if redis_client:
        await redis_client.aclose()
        redis_client = None
        print("Redis connection closed.")



async def add_conversation(user_id: str, email: str, name: str, description: str, topic: str):
    """
    Add a conversation to the Redis database with the specified structure.
    """
    user_key = f"user:{user_id}"
    conversation_id = str(uuid.uuid4())  # Unique conversation ID
    timestamp = datetime.now().isoformat()

    # Ensure the user exists with their email
    await redis_client.hset(user_key, mapping={"user_email": email})

    # Structure for the conversation
    conversation_data = {
        "name": name,
        "description": description,
        "timestamp": timestamp,
        "topic": topic,
        "files": json.dumps({})
    }

    # Add the conversation under 'user:{user_id}:conversations'
    conversations_key = f"{user_key}:conversations"
    await redis_client.hset(conversations_key, conversation_id, json.dumps(conversation_data))

    return {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "conversation_data": conversation_data,
    }


async def fetch_user_conversations(user_id: str) -> list:
    user_conversations_key = f"user:{user_id}:conversations"
    conversations_data = await redis_client.hgetall(user_conversations_key)

    if not conversations_data:
        return []

    conversations = []
    for conversation_id, conversation_json in conversations_data.items():
        try:
            conversation = json.loads(conversation_json)
            conversation["id"] = conversation_id
            # Format the timestamp
            if 'timestamp' in conversation:
                timestamp = datetime.fromisoformat(conversation['timestamp'])
                conversation['created_at'] = timestamp.strftime('%Y-%m-%d %H:%M')
                del conversation['timestamp']
            conversations.append(conversation)
        except json.JSONDecodeError:
            continue  # Skip invalid JSON

    return conversations


async def fetch_conversation(user_id: str, conversation_id: str) -> dict:
    user_conversations_key = f"user:{user_id}:conversations"
    conversation_json = await redis_client.hget(user_conversations_key, conversation_id)
    
    if conversation_json is None:
        raise ValueError(f"Conversation with ID {conversation_id} not found for user {user_id}.")

    try:
        conversation = json.loads(conversation_json)
        conversation["id"] = conversation_id

        if 'timestamp' in conversation:
            timestamp = datetime.fromisoformat(conversation['timestamp'])
            conversation['created_at'] = timestamp.strftime('%Y-%m-%d %H:%M')
            del conversation['timestamp']
        
        return conversation

    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding conversation data: {str(e)}")


async def update_conversation_files(user_id: str, conversation_id: str, uploaded_files):
    """
    Update the conversation files for a specific user and conversation ID.
    If a file with the same name already exists, replace its information with the new data.
    """
    user_conversations_key = f"user:{user_id}:conversations"

    # Fetch the existing conversation data from Redis
    conversation_json = await redis_client.hget(user_conversations_key, conversation_id)
    if not conversation_json:
        raise ValueError(f"Conversation with ID {conversation_id} not found for user {user_id}.")

    try:
        conversation_data = json.loads(conversation_json)

        # Load existing files data or initialize as empty
        files_data = json.loads(conversation_data.get("files", "{}"))

        for file_info in uploaded_files.values():
            file_name = file_info["file_name"]
            file_path = file_info["file_path"]

            # Replace or add the file's information
            files_data[file_name] = {
                "file_path": file_path,
                "upload_timestamp": file_info["timestamp"],  # Ensure the key is 'timestamp'
            }

        # Update the conversation data with the new files
        conversation_data["files"] = json.dumps(files_data)

        # Save the updated conversation data back to Redis
        await redis_client.hset(user_conversations_key, conversation_id, json.dumps(conversation_data))

    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding conversation data: {str(e)}")

    return {"message": "Conversation files updated successfully."}


async def delete_file_from_redis(user_id: str, conversation_id: str, file_name: str):
    """
    Deletes the specified file from the Redis conversation data.

    Args:
        user_id (str): The user ID.
        conversation_id (str): The conversation ID.
        file_name (str): The name of the file to delete.

    Raises:
        HTTPException: If an error occurs during the deletion process.
    """
    user_conversations_key = f"user:{user_id}:conversations"
    
    try:
        # Fetch the conversation data from Redis
        conversation_json = await redis_client.hget(user_conversations_key, conversation_id)
        
        if not conversation_json:
            raise Exception(f"Conversation with ID {conversation_id} not found.")
        
        conversation_data = json.loads(conversation_json)
        files_data = json.loads(conversation_data.get("files", "{}"))
        
        # Check if the file exists in the conversation data
        if file_name not in files_data:
            raise Exception(f"File {file_name} not found in the conversation data.")
        
        # Remove the file from the conversation's files data
        del files_data[file_name]
        conversation_data["files"] = json.dumps(files_data)
        
        # Save the updated conversation data back to Redis
        await redis_client.hset(user_conversations_key, conversation_id, json.dumps(conversation_data))
        
        return {"message": f"File {file_name} deleted successfully from the conversation data."}

    except Exception as e:
        raise Exception(f"Error deleting file from Redis: {str(e)}")