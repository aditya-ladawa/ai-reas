import os

# deletes .pdf files for the current user
async def delete_file_from_storage(file_path: str):
    """
    Deletes a file from the local file system.

    Args:
        file_path (str): The path to the file to be deleted.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)  # Delete the file
        else:
            raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error deleting file from storage: {str(e)}")

