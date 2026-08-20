"""Check vector database status."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from vector_db_faiss import VectorDatabase

try:
    db = VectorDatabase(persist_directory="./chroma_db")
    count = db.count()
    print(f"✓ Vector database loaded")
    print(f"✓ Document chunks: {count}")
    
    if count == 0:
        print("\n⚠ Warning: No documents in database!")
        print("Run: python ingest_medical_docs.py")
    else:
        # Test search with lower threshold
        from embeddings import get_embedding_model
        model = get_embedding_model()
        query_embedding = model.encode("What is cancer?")
        
        # Try with no threshold first
        results = db.search(query_embedding, top_k=5, threshold=0.0)
        print(f"\n✓ Search test (no threshold)")
        print(f"  Found {len(results)} results")
        if results:
            print(f"  Top scores: {[r['similarity_score'] for r in results[:3]]}")
            print(f"  Top result: {results[0]['metadata']['document_name']}")
            print(f"  Text preview: {results[0]['text'][:100]}...")
            
        # Try with threshold 0.3
        results_filtered = db.search(query_embedding, top_k=5, threshold=0.3)
        print(f"\n✓ Search test (threshold=0.3)")
        print(f"  Found {len(results_filtered)} results")
            
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
