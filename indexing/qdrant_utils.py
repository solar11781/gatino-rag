from qdrant_client.http import models as qm

from config import (
    QDRANT_COLLECTION,
    EMBED_DIM,
)


def init_qdrant_collection(client, collection_name: str = QDRANT_COLLECTION, vector_size: int = EMBED_DIM):
    # Delete existing collection if exists
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    # Create new collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qm.VectorParams(
            size=vector_size,
            distance=qm.Distance.COSINE,
        ),
    )

    print(f"[INFO] Collection '{collection_name}' (dim={vector_size}) ready.")