"""Test the full RAG pipeline end-to-end."""

import sys
import os
import asyncio
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from config import config
from vector_db_faiss import VectorDatabase
from llm_service import LLMService
from rag_pipeline import RAGPipeline

async def test_pipeline():
    print("=== Testing Full RAG Pipeline ===\n")
    
    # Initialize components
    print("1. Initializing components...")
    vector_db = VectorDatabase(persist_directory="./chroma_db")
    print(f"   ✓ Vector DB: {vector_db.count()} chunks")
    
    llm_service = LLMService(
        groq_api_key=config.groq_api_key,
        model_name=config.rag.model_name,
        temperature=config.rag.temperature,
        max_tokens=config.rag.max_tokens
    )
    print(f"   ✓ LLM Service: {'Groq' if llm_service.use_groq else 'OpenAI'}")
    
    rag_pipeline = RAGPipeline(
        vector_db=vector_db,
        llm_service=llm_service,
        embedding_model_name=config.embedding_model,
        top_k=config.rag.top_k,
        similarity_threshold=config.rag.similarity_threshold
    )
    print(f"   ✓ RAG Pipeline (threshold={config.rag.similarity_threshold})")
    
    # Test query
    print("\n2. Testing query: 'What is cancer?'")
    query = "What is cancer?"
    
    # Retrieve context
    context_chunks, _ = rag_pipeline.retrieve_context(query)
    print(f"   ✓ Retrieved {len(context_chunks)} chunks")
    if context_chunks:
        print(f"   ✓ Top score: {context_chunks[0]['similarity_score']:.3f}")
        print(f"   ✓ Source: {context_chunks[0]['metadata']['document_name']}")
    
    # Generate response
    print("\n3. Generating response...")
    print("   Response: ", end="", flush=True)
    
    token_count = 0
    async for token in rag_pipeline.process_query(query):
        print(token, end="", flush=True)
        token_count += 1
    
    print(f"\n\n   ✓ Generated {token_count} tokens")
    
    # Get citations
    citations = rag_pipeline.get_citations(query)
    print(f"\n4. Citations: {len(citations)} sources")
    for i, citation in enumerate(citations[:3], 1):
        print(f"   {i}. {citation.document_name} (page {citation.page_number})")
    
    print("\n✓ Pipeline test complete!")

if __name__ == "__main__":
    asyncio.run(test_pipeline())
