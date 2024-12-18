from grobid_client.grobid_client import GrobidClient
import os

def connect_to_grobid():
    try:
        grobid_client = GrobidClient(config_path="api/config.json")
        print('Started Grobid client')

        return grobid_client

    except Exception as e:
        print(f"Connection error: {e}")
        return None

