"""
Agentic VLM Reasoner (Vision-Language Model Orchestrator)
This module acts as the bridge between the lower-level FluxState tracking/detection
and higher-level semantic reasoning.

It intercepts complex temporal events and passes them to a Vision-Language Model
for deep contextual understanding.

V1.3.0 Update: Unified Memory Execution
This agent now leverages the MLX framework to execute VLMs (like Qwen2.5-VL)
directly on the Neural Engine and GPU of Apple Silicon hardware.
"""
import base64
import cv2
import threading
import numpy as np

try:
    from mlx_vlm import load, generate
    import mlx.core as mx
    MLX_AVAILABLE = True
except Exception as e:
    MLX_AVAILABLE = False
    print(f"[Warning] MLX VLM dependencies failed to load: {e}. Agents will fallback to mock/offline mode.")


class SemanticAgent:
    """
    LEGACY SemanticAgent.
    Preserved for strict backward compatibility.
    """
    def __init__(self, model_path="qwen/Qwen2.5-VL-3B-Instruct"):
        """
        Initializes the VLM locally on Apple Silicon Unified Memory using MLX.
        Downloads the model from HuggingFace if not present.
        """
        self.enabled = False
        self.model = None
        self.processor = None
        self.config = None
        
        if MLX_AVAILABLE:
            try:
                print(f"[VLM] Initializing {model_path} via MLX on Apple Silicon GPU...")
                self.model, self.processor = load(model_path, trust_remote_code=True)
                self.enabled = True
                print("[VLM] Model loaded into unified memory successfully.")
            except Exception as e:
                print(f"[VLM Error] Could not load MLX model: {e}")

    def investigate_scene(self, context_log, frame, prompt="Initiate Visual Threat Analysis. Scan the optical feed for tactical anomalies. Classify any handheld objects and assess the subject's intent."):
        if not self.enabled or frame is None:
            return "[VLM Offline] Fallback to standard rule-engine."
            
        import uuid
        import os
        try:
            temp_img_path = f"/tmp/flux_vlm_frame_{uuid.uuid4().hex}.jpg"
            cv2.imwrite(temp_img_path, frame)
            
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Context Log: {context_log}. Query: {prompt}"}
                ]}
            ]
            
            formatted_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            response = generate(self.model, self.processor, formatted_prompt, [temp_img_path], verbose=False)
            return response.text if hasattr(response, 'text') else str(response)
            
        except Exception as e:
            return f"[VLM Error] {str(e)}"
        finally:
            if 'temp_img_path' in locals() and os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            
    def query_temporal_memory(self, sql_results, natural_language_query):
        if not self.enabled:
            return "[VLM Offline] Temporal summary requires an active reasoning agent."
            
        try:
            context_block = "\n".join([f"[{row[0]}] {row[1]}" for row in sql_results])
            prompt = (
                f"You are the FluxState Autonomous Intelligence Nexus, an advanced OSINT and Threat Analysis system.\n"
                f"Perform forensic analysis on the provided telemetry logs. Respond with tactical precision, identifying critical events, target trajectories, and behavioral anomalies.\n\n"
                f"Operator Query: {natural_language_query}\n\n"
                f"Telemetry Logs:\n{context_block}"
            )
            
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            response = generate(self.model, self.processor, formatted_prompt, verbose=False)
            return response.text if hasattr(response, 'text') else str(response)
            
        except Exception as e:
            return f"[Agent Error] {str(e)}"


class MLXModelPool:
    """
    V2 Adaptive Architecture: Singleton-style manager that safely multiplexes 
    a 4-bit VLM and a lightweight embedding model on Apple Silicon Unified Memory 
    with thread-safe locks and explicit Metal cache clearing.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MLXModelPool, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
            
    def __init__(self, vlm_path="mlx-community/Qwen2.5-VL-3B-Instruct-4bit", enable_adaptive=True):
        if self._initialized:
            return
            
        self.vlm_path = vlm_path
        self.enable_adaptive = enable_adaptive
        self.vlm_enabled = False
        self.vlm_model = None
        self.vlm_processor = None
        
        self.embedder = None
        self._gpu_lock = threading.Lock()
        
        if enable_adaptive:
            try:
                from .adaptation.embedder import LightweightEmbedder
                self.embedder = LightweightEmbedder()
            except ImportError as e:
                print(f"[MLXModelPool] Failed to import embedder: {e}")
            
        self._lazy_load_vlm()
        self._initialized = True
        
    def _lazy_load_vlm(self):
        if MLX_AVAILABLE and not self.vlm_enabled:
            with self._gpu_lock:
                try:
                    print(f"[MLXModelPool] Lazy-loading {self.vlm_path} via MLX...")
                    self.vlm_model, self.vlm_processor = load(self.vlm_path, trust_remote_code=True)
                    self.vlm_enabled = True
                    print("[MLXModelPool] VLM successfully loaded into Unified Memory.")
                except Exception as e:
                    print(f"[MLXModelPool Error] Could not load MLX model: {e}")
                    
    def _clear_metal_cache(self):
        """Forces MLX/Metal to release unused buffers to prevent OOM."""
        if MLX_AVAILABLE:
            try:
                mx.clear_cache()
            except Exception:
                pass
            
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generates a 384-d semantic embedding using the lightweight model."""
        if not self.embedder:
            import hashlib
            h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            rs = np.random.RandomState(h)
            emb = rs.rand(384).astype(np.float32)
            return emb / np.linalg.norm(emb)
            
        with self._gpu_lock:
            emb = self.embedder.embed_text(text)
            self._clear_metal_cache()
            return emb
            
    def investigate_scene(self, context_log, frame, prompt="Initiate Visual Threat Analysis.", historical_context=None):
        """
        Takes a flagged event, a raw frame, and optional semantic RAG historical context, 
        and asks the local MLX VLM to reason about it safely via the GPU lock.
        """
        if not self.vlm_enabled or frame is None:
            return "[VLM Offline] Fallback to standard rule-engine."
            
        # Build prompt incorporating historical context if provided (The Semantic RAG injection)
        final_prompt = f"Context Log: {context_log}\n"
        if historical_context:
            final_prompt += f"Historical Semantic Memory: {historical_context}\n"
        final_prompt += f"Query: {prompt}"
        
        import uuid
        import os
        with self._gpu_lock:
            temp_img_path = f"/tmp/flux_pool_frame_{uuid.uuid4().hex}.jpg"
            try:
                cv2.imwrite(temp_img_path, frame)
                
                messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": final_prompt}
                    ]}
                ]
                
                formatted_prompt = self.vlm_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                response = generate(self.vlm_model, self.vlm_processor, formatted_prompt, [temp_img_path], verbose=False)
                res_text = response.text if hasattr(response, 'text') else str(response)
                
                # Explicit cleanup after heavy generation
                self._clear_metal_cache()
                return res_text
            except Exception as e:
                self._clear_metal_cache()
                return f"[VLM Error] {str(e)}"
            finally:
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
