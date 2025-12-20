from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from config import QDRANT_URL, QDRANT_COLLECTION, EMBED_DIM
from embeddings import embed_text


def init_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int = EMBED_DIM):
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


def search_example(query: str, limit: int = 5):
    client = QdrantClient(url=QDRANT_URL)

    # Embed query
    q_emb = embed_text(query)

    # Perform vector search
    result = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=q_emb,
        limit=limit,
    )

    print(f"\n[SEARCH] Top {limit} results for query: '{query}'")

    for point in result.points:
        p = point.payload
        score = point.score

        print(
            f"- score={score:.4f} | "
            f"{p.get('repo_name')} :: {p.get('file_path')} "
            f"({p.get('start_line')}-{p.get('end_line')})"
        )
