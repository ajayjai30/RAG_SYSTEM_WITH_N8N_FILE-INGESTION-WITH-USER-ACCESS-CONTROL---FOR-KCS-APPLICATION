import os
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Use fastembed for BM25 computation
try:
    from fastembed import SparseTextEmbedding
except ImportError:
    print("Please install fastembed: pip install fastembed")
    exit(1)

client = QdrantClient("http://localhost:6333")
COLLECTION_NAME = "app_rag_docs"

def build_hybrid_index():
    print("1. Initializing SparseTextEmbedding (BM25)...")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    print("\n2. Fetching all existing points from Qdrant to calculate BM25...")
    points = []
    offset = None
    while True:
        resp, next_page = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            with_payload=True,
            with_vectors=True, # Need vectors to re-upsert properly
            offset=offset
        )
        points.extend(resp)
        offset = next_page
        if offset is None:
            break

    print(f" -> Found {len(points)} points.")

    print("\n3. Recreating Collection with Hybrid Config...")
    client.delete_collection(COLLECTION_NAME)
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=768, 
            distance=models.Distance.COSINE
        ),
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        }
    )
    print(" -> Success: Rebuilt Collection.")

    print(f" -> Found {len(points)} points.")

    print("\n4. Encoding text and Upserting Sparse Vectors in batches...")
    
    batch_size = 50
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        
        texts = [p.payload.get("content", "") for p in batch]
        # Calculate BM25 Sparse Vectors
        sparse_embeddings = list(sparse_model.embed(texts))
        
        updated_points = []
        for p, s_emb in zip(batch, sparse_embeddings):
            # Keep original unnamed dense vector, add named "sparse" vector
            vectors_dict = {
                "": p.vector, # unnamed dense 
                "sparse": models.SparseVector(
                    indices=s_emb.indices.tolist(),
                    values=s_emb.values.tolist()
                )
            }
            
            updated_points.append(
                models.PointStruct(
                    id=p.id,
                    vector=vectors_dict,
                    payload=p.payload
                )
            )
        
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=updated_points
        )
        print(f" -> Upserted batch {i//batch_size + 1}")

    print("\n✅ Hybrid Index successfully built! The DB now supports Dense + BM25 Fusion Queries.")

if __name__ == "__main__":
    build_hybrid_index()
