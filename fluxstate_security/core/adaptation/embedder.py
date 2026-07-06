import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[Warning] sentence-transformers not found. Embeddings will fallback to mock mode.")

class LightweightEmbedder:
    """
    Generates 384-d semantic embeddings for RAG lookups.
    Uses all-MiniLM-L6-v2, optimized for Apple Silicon (MPS) if available.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = None
        self.enabled = False
        
        if TRANSFORMERS_AVAILABLE:
            try:
                device = "mps" if torch.backends.mps.is_available() else "cpu"
                print(f"[Embedder] Initializing {model_name} on {device}...")
                self.model = SentenceTransformer(model_name, device=device)
                self.enabled = True
                print("[Embedder] Successfully loaded into memory.")
            except Exception as e:
                print(f"[Embedder Error] Could not load embedding model: {e}")
                
    def embed_text(self, text: str) -> np.ndarray:
        """Generates a normalized float32 embedding."""
        if not self.enabled or self.model is None:
            # Fallback to deterministic mock mode
            import hashlib
            h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            rs = np.random.RandomState(h)
            emb = rs.rand(384).astype(np.float32)
            return emb / np.linalg.norm(emb)
            
        try:
            # sentence_transformers natively returns numpy array
            embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return embedding.astype(np.float32)
        except Exception as e:
            print(f"[Embedder Error] Failed to generate embedding: {e}")
            raise
