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

try:
    from mlx_vlm import load, generate
    MLX_AVAILABLE = True
except Exception as e:
    MLX_AVAILABLE = False
    print(f"[Warning] MLX VLM dependencies failed to load: {e}. SemanticAgent will fallback to mock/offline mode.")


class SemanticAgent:
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
                # In production, we load this asynchronously or as a singleton.
                # For this release architecture, we scaffold the MLX loading:
                self.model, self.processor = load(model_path, trust_remote_code=True)
                self.enabled = True
                print("[VLM] Model loaded into unified memory successfully.")
            except Exception as e:
                print(f"[VLM Error] Could not load MLX model: {e}")

    def investigate_scene(self, context_log, frame, prompt="Initiate Visual Threat Analysis. Scan the optical feed for tactical anomalies. Classify any handheld objects and assess the subject's intent."):
        """
        Takes a flagged event and a raw frame, and asks the local MLX VLM to reason about it.
        This provides true 'Scene Understanding' beyond simple bounding boxes.
        """
        if not self.enabled or frame is None:
            return "[VLM Offline] Fallback to standard rule-engine."
            
        try:
            # Encode frame to disk temporarily for MLX (or pass numpy array if supported)
            temp_img_path = "/tmp/flux_vlm_frame.jpg"
            cv2.imwrite(temp_img_path, frame)
            
            # Construct the Qwen-VL specific prompt format
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Context Log: {context_log}. Query: {prompt}"}
                ]}
            ]
            
            formatted_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Execute inference on the Apple Silicon GPU
            response = generate(self.model, self.processor, formatted_prompt, [temp_img_path], verbose=False)
            return response.text if hasattr(response, 'text') else str(response)
            
        except Exception as e:
            return f"[VLM Error] {str(e)}"
            
    def query_temporal_memory(self, sql_results, natural_language_query):
        """
        Allows an operator to ask: 'What happened between 2 PM and 3 PM?'
        Takes the SQLite output and uses the local MLX VLM to generate a human-readable summary.
        """
        if not self.enabled:
            return "[VLM Offline] Temporal summary requires an active reasoning agent."
            
        try:
            # Format the raw SQL forensics into a context block
            context_block = "\n".join([f"[{row[0]}] {row[1]}" for row in sql_results])
            
            prompt = (
                f"You are the FluxState Autonomous Intelligence Nexus, an advanced OSINT and Threat Analysis system.\n"
                f"Perform forensic analysis on the provided telemetry logs. Respond with tactical precision, identifying critical events, target trajectories, and behavioral anomalies.\n\n"
                f"Operator Query: {natural_language_query}\n\n"
                f"Telemetry Logs:\n{context_block}"
            )
            
            # Since this is a text-only query, we don't pass images.
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Execute inference on the Apple Silicon GPU
            response = generate(self.model, self.processor, formatted_prompt, verbose=False)
            return response.text if hasattr(response, 'text') else str(response)
            
        except Exception as e:
            return f"[Agent Error] {str(e)}"
