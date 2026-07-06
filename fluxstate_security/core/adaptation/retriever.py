import time
import numpy as np

# Adjust imports safely based on module paths
try:
    from ..agent import MLXModelPool
    from ..forensics import ForensicDatabase
except ImportError:
    pass

class SemanticRetriever:
    """
    Semantic Retrieval layer for Edge Adaptive Intelligence.
    Retrieves visually/semantically similar past events to build context
    for the Vision-Language Model.
    
    IMPORTANT: This layer MUST NOT auto-suppress alerts. It only supplies 
    context to the VLM to make the final determination.
    """
    def __init__(self, db, agent_pool, similarity_threshold=0.85):
        self.db = db
        self.agent_pool = agent_pool
        self.similarity_threshold = similarity_threshold

    def get_contextual_memories(self, scene_description: str, top_k: int = 3) -> list:
        """
        1. Encodes the current scene description using the lightweight embedder.
        2. Queries the ContextLedger (SQLite + Numpy Cache) for similar past events.
        3. Returns the raw memory dictionaries for prompt injection.
        """
        if not scene_description or not isinstance(scene_description, str):
            return []
            
        try:
            # 1. Generate float32 embedding via MLXModelPool
            query_embedding = self.agent_pool.generate_embedding(scene_description)
            
            if query_embedding is None or not isinstance(query_embedding, np.ndarray):
                return []
                
            # 2. Query ContextLedger (which implements fast numpy cosine similarity)
            # We explicitly pass our threshold
            similar_events = self.db.search_similar_events(
                query_embedding, 
                top_k=top_k, 
                threshold=self.similarity_threshold
            )
            
            return similar_events
        except Exception as e:
            print(f"[SemanticRetriever] Error retrieving context: {e}")
            return []
