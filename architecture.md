# FluxState SDK Architecture Overview

![FluxState Architecture Diagram](assets/fluxstate_architectural_diagram.png)
FluxState operates as a **Camera-Agnostic Intelligent Video Recorder (IVR)**. It ingests video from existing infrastructure and processes it through a layered inference pipeline to generate semantic metadata.

## 1. Ingestion Layer (RTSP & Local)
*(Handled by `EdgeVideoStream` in `fluxstate_security/utils/video_stream.py`)*

To prevent the AI inference loop from bottlenecking due to network latency, camera ingestion is decoupled into a background daemon thread. 
- The thread continuously pulls frames from RTSP streams or local hardware into a circular buffer.
- The inference loop requests `frame.copy()`, minimizing the latency differential between capture and processing.

## 2. Extraction Pipeline
When sufficient kinetic motion is detected, the SDK fires a parallel extraction pipeline:

- **A. Object Detection (YOLOv8)**: Classifies general objects and assigns persistent ByteTrack IDs.
- **B. Skeletal Kinetics (MediaPipe)**: Extracts 33 3D skeletal joints to map physical limb geometry into semantic actions (`"ARMS_RAISED"`, `"WALKING"`).
- **C. Audio Analysis (PyAudio)**: Calculates RMS volume in a background thread to detect sudden acoustic anomalies.
- **D. Identity Hashing (GPASS)**: Extracts upper-body geometric crops and hashes them to create anonymous, privacy-compliant tracking IDs across frames.

## 3. Inference Engine
*(Handled by `FluxInferenceEngine` in `fluxstate_security/core/engine.py`)*

Telemetry from the extraction pipeline is fed to the `FluxInferenceEngine` and cross-referenced with thresholds defined in `intelligence_policy.json`.
- It fuses the modalities (e.g., matching kinetic action with acoustic volume).
- It generates a formatted natural-language summary (the `Reality Graph` observation).

## 4. Temporal Forensic Memory
*(Handled by `ForensicDatabase` in `fluxstate_security/core/forensics.py`)*

Instead of saving heavy video files to disk, FluxState saves the semantic output into a local SQLite database (`swarm_ledger.db`).
- This creates a searchable text-based ledger of all physical events.
- Operators can perform fast SQL queries or keyword searches (e.g., finding specific actions or anomalies) across months of historical data.

## 5. Agentic VLM Reasoner (Vision-Language Orchestration)
*(Handled by `SemanticAgent` in `fluxstate_security/core/agent.py`)*

To achieve deep scene understanding, FluxState implements an Agentic Reasoning bridge.
- The `SemanticAgent` intercepts flagged SQLite events and executes inference locally using the **MLX framework directly on Apple Silicon unified memory**, removing the need for cloud API calls.
- **Visual Grounding:** To enhance small-model fidelity (e.g., Qwen2.5-VL-3B), the engine draws high-contrast red bounding boxes around anomalies before VLM inference. This forces the model to focus exclusively on target pixels, preventing hallucinations.
- **Tactical Threat Analysis:** The VLM operates using highly structured OSINT/Tactical prompts, strictly ruling out civilian objects and evaluating kinetic threat indicators without hardcoding biases.
- It also enables Natural Language querying against the temporal database, allowing operators to ask: *"What happened between 2 PM and 3 PM?"*

## 6. C-Level Memory Mitigation
*(Handled in the `poll_telemetry` loop in `fluxstate_security/app.py`)*

To address data-retention concerns, the system attempts to clear images from RAM as quickly as possible.
- Python's standard garbage collector does not immediately zero out deallocated memory, meaning images can theoretically linger in the heap.
- To mitigate this, FluxState utilizes `ctypes.memset` to manually overwrite the underlying C-level memory buffer of the numpy image array with zeros immediately after inference is complete.
- *Note: This mitigates lingering Python heap data, but does not prevent the OS, camera firmware, or GPU drivers from maintaining their own underlying buffers.*

## 7. Integration Webhooks
*(Handled by `_push_to_integration_bus` in `fluxstate_security/app.py`)*

When an anomaly surpasses the defined thresholds, the SDK can instantly fire an HTTP `POST` request to the client's proprietary Video Management System (VMS) webhook. The payload contains the full JSON telemetry, allowing the client to log the event in their own systems.

## 8. Containerized Deployment
*(Handled by `Dockerfile`)*

FluxState is packaged as an edge-ready container. This guarantees native C++ dependencies (Tesseract OCR, PortAudio) are perfectly locked, bypassing OS-level package manager constraints and enabling immediate deployment on Kubernetes or Docker Swarm.
