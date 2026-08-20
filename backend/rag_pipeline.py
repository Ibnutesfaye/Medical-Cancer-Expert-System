"""
RAG pipeline orchestration.

Combines retrieval and generation for question answering.
Falls back to external search (Wikipedia/PubMed) when internal docs have no relevant results.
"""

from typing import List, Dict, Any, AsyncGenerator, Optional
from pydantic import BaseModel
from embeddings import get_embedding_model
from vector_db_faiss import VectorDatabase
from llm_service import LLMService
from external_search import ExternalSearchService
from textblob import TextBlob


class Citation(BaseModel):
    """Citation for a source document or external URL."""
    document_name: Optional[str] = None
    page_number: Optional[int] = None
    chunk_text: str
    relevance_score: float
    source_type: str = "document"  # "document" | "wikipedia" | "pubmed" | "external"
    url: Optional[str] = None


class Message(BaseModel):
    """Conversation message."""
    role: str  # 'user' or 'assistant'
    content: str


class RAGPipeline:
    """
    RAG pipeline with external search fallback.
    """

    def __init__(
        self,
        vector_db: VectorDatabase,
        llm_service: LLMService,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        similarity_threshold: float = 0.2,  # Lowered from 0.3 to let more relevant chunks pass
        fallback_enabled: bool = True
    ):
        self.vector_db = vector_db
        self.llm_service = llm_service
        self.embedding_model = get_embedding_model(embedding_model_name)
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.fallback_enabled = fallback_enabled
        self.external_search = ExternalSearchService()

        # Track last query source for citations
        self._last_source = "document"
        self._last_external_result = None

    def retrieve_context(self, query: str) -> tuple[List[Dict[str, Any]], bool]:
        """Retrieve relevant context from vector DB."""
        query_embedding = self.embedding_model.encode(query)
        results = self.vector_db.search(
            query_embedding,
            top_k=self.top_k,
            threshold=self.similarity_threshold
        )
        results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)
        return results, False

    def construct_prompt(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        conversation_history: Optional[List[Message]] = None,
        external_result: Optional[dict] = None
    ) -> str:
        """Construct prompt for LLM using internal docs or external search result."""

        history_text = ""
        if conversation_history:
            recent = conversation_history[-6:]
            history_text = "\n\nConversation History:\n" + "\n".join(
                f"{m.role.capitalize()}: {m.content}" for m in recent
            )

        base_system_prompt = (
            "You are a highly specialized medical AI assistant focused ONLY on cancer and oncology.\n\n"
            "The system is a Medical Cancer Expert System that combines deep learning image analysis and a retrieval-augmented generation (RAG) system for answering medical questions.\n\n"
            "⚠️ Core Limitations (must be clearly acknowledged by the system)\n"
            "* The model is NOT clinically approved and is for educational/research purposes only.\n"
            "* Predictions may be inaccurate due to limited dataset size and variability in medical images.\n"
            "* The system does not replace professional medical diagnosis.\n"
            "* Training data quality and class imbalance may affect performance.\n\n"
            "🧠 Model Training Requirements & 📊 Image Analysis & Admin Dashboard Logging\n"
            "The image classification model must be trained using ResNet18 or equivalent CNN architecture. "
            "The number of training epochs must be greater than 20 (recommended: 20–30 epochs) to improve accuracy and generalization. "
            "During training, the system must compute and store: Training accuracy, Validation accuracy, Loss values, Final model performance metrics. "
            "After every image analysis, the system automatically sends and stores the prediction details (predicted cancer type/non-cancer, confidence score %, model used), "
            "model performance metadata (training accuracy, validation accuracy, loss value), and evaluation information in the Admin Dashboard. "
            "The admin dashboard displays image analysis history, model performance metrics over time, training vs validation accuracy comparison, and model version/configuration.\n\n"
            "🧾 Model Evaluation Explanation (must be included in responses)\n"
            "When asked, you should explain:\n"
            "* How the model was trained (CNN/ResNet18)\n"
            "* How accuracy is calculated (Training accuracy = correct predictions on training set; Validation accuracy = performance on unseen data)\n"
            "* Why validation accuracy is important (to detect overfitting)\n"
            "* How the system decides final predictions (argmax of output probabilities based on learned features from training data)\n\n"
            "🧠 Conversation Memory (VERY IMPORTANT)\n"
            "You MUST remember previous messages. If a user previously mentioned a symptom and now mentions another, combine them.\n"
            "Example: 'Previously you mentioned headache, and now nausea. These combined symptoms may indicate conditions such as...'\n\n"
            "💬 Symptom-Based Reasoning & Explanation (MANDATORY)\n"
            "When a user describes symptoms (even with spelling mistakes):\n"
            "1. NEVER give a final diagnosis. Use phrases like: 'may be', 'could be related to', 'possible condition'.\n"
            "2. Explain your reasoning explicitly: 'This suggestion is based on: [list of symptoms extracted]. These symptoms are commonly associated with [condition types].'\n"
            "3. Format it similarly to: 'Based on your symptoms (...), this could possibly be related to conditions such as ..., but it is not certain. You should consult a medical professional for proper diagnosis.'\n\n"
            "🚀 Goal of the System\n"
            "The system is designed to: Detect cancer types from medical images, Provide AI-based medical Q&A using RAG, Show transparent model evaluation metrics, Help users understand AI decision-making in healthcare.\n\n"
            "STRICT RULES:\n"
            "* You ONLY answer questions related to cancer, tumors, oncology, diagnosis, and treatment.\n"
            "* If the question is NOT related to cancer, you MUST refuse politely.\n"
            "* Do NOT answer general knowledge questions.\n"
            "* Do NOT go outside the medical cancer domain.\n"
            "* ⚠️ SAFETY RULE (MANDATORY): EVERY medical response MUST include this exact text at the end: 'This is not a medical diagnosis. Please consult a qualified doctor.'\n\n"
            "If the question is unrelated, respond ONLY with:\n"
            "\"I am a cancer-specialized assistant and can only help with cancer-related questions.\"\n\n"
            "Always keep responses medically accurate, simple, and safe.\n"
            "Always recommend consulting a healthcare professional.\n\n"
            "If the question IS related to cancer, always structure your answer exactly like this:\n\n"
            "1. **Summary Line** — One sentence definition. Bold the subject + main descriptor.\n\n"
            "2. 🔬 **Simple Explanation**\n"
            "One short intro line.\n"
            "3–5 bullets explaining how it works. Use **bold verbs** in each bullet.\n\n"
            "3. 🧬 **Types / Examples**\n"
            "Format each as: **Type** – short description\n"
            "Include at least one real-world example.\n\n"
            "4. ⚠️ **Key Features / Signs**\n"
            "5 short bullet points.\n"
        )

        # --- Internal document context ---
        if context_chunks:
            system_prompt = base_system_prompt
            context_text = "\n\n".join([
                f"[Source: {c['metadata']['document_name']}, Page {c['metadata']['page_number']}]\n{c['text']}"
                for c in context_chunks
            ])
            return f"""{system_prompt}

Context from medical documents:
{context_text}
{history_text}

User Question: {query}

Answer:"""

        # --- External search context ---
        if external_result:
            source_label = external_result.get("source", "external").capitalize()
            system_prompt = f"The following information was retrieved from {source_label}.\n\n" + base_system_prompt
            return f"""{system_prompt}

Information from {source_label} ({external_result.get('url', '')}):
{external_result['text']}
{history_text}

User Question: {query}

Answer:"""

        # --- LLM general knowledge fallback ---
        system_prompt = base_system_prompt
        return f"""{system_prompt}
{history_text}

User Question: {query}

Answer:"""

    def extract_citations(
        self,
        context_chunks: List[Dict[str, Any]],
        external_result: Optional[dict] = None
    ) -> List[Citation]:
        """Extract citations from internal chunks or external result."""

        if context_chunks:
            doc_citations: Dict[str, dict] = {}
            for chunk in context_chunks:
                doc_name = chunk['metadata']['document_name']
                page_num = chunk['metadata']['page_number']
                score = chunk['similarity_score']
                if doc_name not in doc_citations:
                    doc_citations[doc_name] = {
                        'pages': set(), 'score': score, 'text': chunk['text'][:200]
                    }
                doc_citations[doc_name]['pages'].add(page_num)

            citations = [
                Citation(
                    document_name=name,
                    page_number=sorted(info['pages'])[0],
                    chunk_text=info['text'],
                    relevance_score=info['score'],
                    source_type="document"
                )
                for name, info in doc_citations.items()
            ]
            return sorted(citations, key=lambda x: x.relevance_score, reverse=True)

        if external_result:
            return [Citation(
                document_name=None,
                page_number=None,
                chunk_text=external_result.get('text', '')[:200],
                relevance_score=1.0,
                source_type=external_result.get('source', 'external'),
                url=external_result.get('url')
            )]

        return []

    async def process_query(
        self,
        query: str,
        conversation_history: Optional[List[Message]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Process query: try internal docs first, fall back to external search.

        Yields:
            Response tokens, then a special [SOURCE] marker at the end.
        """
        # 0. Automatically fix spelling mistakes
        original_query = query
        query = str(TextBlob(query).correct())
        if query != original_query:
            print(f"Corrected spelling: '{original_query}' -> '{query}'")

        # 1. Try internal vector DB
        context_chunks, _ = self.retrieve_context(query)
        external_result = None
        source = "document"

        # 2. If FAISS index is empty, fall back to external search
        # Fallback only happens when FAISS is truly empty (not due to filtering issues)
        if self.vector_db.count() == 0 and self.fallback_enabled:
            external_result = await self.external_search.search(query)
            source = external_result.get("source", "external") if external_result else "llm"
        elif not context_chunks:
            # If filtering somehow removed all chunks but DB isn't empty, LLM handles it as fallback
            source = "llm"

        # Store for get_citations()
        self._last_source = source
        self._last_external_result = external_result

        # 3. Build prompt and stream answer
        prompt = self.construct_prompt(query, context_chunks, conversation_history, external_result)

        async for token in self.llm_service.generate_streaming(prompt):
            yield token

        # 4. Yield source marker so the caller can include it in the response
        yield f"[SOURCE_MARKER]{source}"

    def get_citations(self, query: str) -> List[Citation]:
        """Return citations from the last processed query."""
        context_chunks, _ = self.retrieve_context(query)
        return self.extract_citations(context_chunks, self._last_external_result)

    def get_last_source(self) -> str:
        """Return the source used in the last query."""
        return self._last_source
