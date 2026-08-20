"""
Script to ingest provided medical documents into the vector database.
"""

import os
from pathlib import Path
from ingestion import IngestionService
from vector_db_faiss import VectorDatabase
from config import config


def main():
    """Ingest all medical documents from backend directory."""
    print("Medical Document Ingestion Script")
    print("=" * 50)
    
    # Initialize services
    vector_db = VectorDatabase(persist_directory=config.vector_db_path)
    ingestion_service = IngestionService(
        vector_db=vector_db,
        chunk_size=config.rag.chunk_size,
        overlap=config.rag.chunk_overlap,
        embedding_model_name=config.embedding_model
    )
    
    # Medical documents to ingest
    documents = [
        "breast_cancer.pdf.docx",
        "prevention.pdf.pdf",
        "screening.pdf.docx",
        "world_cancer.pdf.pdf"
    ]
    
    total_chunks = 0
    successful = 0
    failed = 0
    
    for doc_name in documents:
        doc_path = Path(doc_name)
        
        if not doc_path.exists():
            print(f"\n❌ File not found: {doc_name}")
            failed += 1
            continue
        
        print(f"\n📄 Processing: {doc_name}")
        print(f"   Size: {doc_path.stat().st_size / 1024:.1f} KB")
        
        try:
            result = ingestion_service.ingest_file(str(doc_path))
            
            if result.success:
                print(f"   ✅ Success: {result.chunks_created} chunks created")
                total_chunks += result.chunks_created
                successful += 1
            else:
                print(f"   ❌ Failed: {result.error}")
                failed += 1
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("Ingestion Summary:")
    print(f"  Total documents processed: {len(documents)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total chunks created: {total_chunks}")
    print(f"  Vector DB total count: {vector_db.count()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
