# FluxState Edge SDK: System Architecture

The following diagram illustrates the inference pipeline, memory management, and agentic reasoning data flow of the FluxState Edge SDK. This diagram is designed for public documentation, architectural reviews, and README integration.

```mermaid
graph TD
    %% Styling
    classDef hardware fill:#1e1e1e,stroke:#444,stroke-width:2px,color:#fff
    classDef extraction fill:#0e4b75,stroke:#1a7fc4,stroke-width:2px,color:#fff
    classDef engine fill:#3a4b60,stroke:#5c728e,stroke-width:2px,color:#fff
    classDef memory fill:#145224,stroke:#27ae60,stroke-width:2px,color:#fff
    classDef vlm fill:#802121,stroke:#e74c3c,stroke-width:2px,color:#fff
    classDef vms fill:#4b2b73,stroke:#9b59b6,stroke-width:2px,color:#fff
    classDef policy fill:#755913,stroke:#f1c40f,stroke-width:2px,color:#fff

    %% Components
    Camera[RTSP/Local Camera Feed]:::hardware
    Mic[Hardware Audio Feed]:::hardware

    subgraph "Ingestion Layer"
        VideoStream[EdgeVideoStream<br/>(Daemon Thread / Ring Buffer)]:::hardware
    end

    subgraph "Extraction Pipeline (Parallel)"
        YOLO[YOLOv8 Nano<br/>Object Detection]:::extraction
        Tracker[ByteTrack<br/>Persistent Tracking]:::extraction
        MediaPipe[MediaPipe<br/>3D Skeletal Kinetics]:::extraction
        GPASS[GPASS<br/>Privacy Identity Hashing]:::extraction
        Acoustic[PyAudio RMS<br/>Acoustic Analysis]:::extraction
    end

    subgraph "Inference Engine (core/engine.py)"
        Policy[(intelligence_policy.json)]:::policy
        Engine[FluxInferenceEngine<br/>Reality Graph Generation]:::engine
    end

    subgraph "Temporal Memory (core/forensics.py)"
        SQLite[(swarm_ledger.db<br/>Forensic Ledger)]:::memory
    end

    subgraph "Agentic Reasoning (core/agent.py)"
        VisualGrounding[Visual Grounding<br/>Target Annotation (Red Box)]:::vlm
        VLM[SemanticAgent (Qwen2.5-VL)<br/>MLX Unified Memory Execution]:::vlm
    end

    Webhook[Enterprise VMS / Webhook Bus]:::vms

    %% Data Flow
    Camera --> VideoStream
    VideoStream -->|frame.copy()| YOLO
    YOLO --> Tracker
    VideoStream -->|frame.copy()| MediaPipe
    Tracker --> GPASS
    Mic --> Acoustic

    Tracker --> Engine
    MediaPipe --> Engine
    GPASS --> Engine
    Acoustic --> Engine
    Policy --> Engine

    Engine -->|Semantic Observation| SQLite
    
    SQLite -->|Flagged Threat / Anomaly| VisualGrounding
    VideoStream -->|Raw Frame| VisualGrounding
    VisualGrounding -->|Annotated Frame + OSINT Prompt| VLM
    
    VLM -->|Tactical Assessment| Webhook
    Engine -->|Reality Graph JSON| Webhook
    
    %% Memory Mitigation Note
    VLM -.->|ctypes.memset| MemoryMitigation[C-Level Pixel Buffer Zeroing]
```

### Key Architectural Highlights
1. **Threaded Ingestion**: `EdgeVideoStream` pulls RTSP frames into a circular buffer on a background thread to prevent network I/O from bottlenecking inference.
2. **Deterministic Extraction**: YOLOv8, MediaPipe, and PyAudio extract physical geometries and acoustic metadata deterministically at a high frame rate.
3. **Temporal Forensic Ledger**: All events are logged as a *Reality Graph* into SQLite, allowing operators to run SQL/Natural Language queries on historical physical events instead of storing heavy video files.
4. **Agentic Interception**: If `FluxInferenceEngine` flags an anomaly based on the `intelligence_policy.json`, it triggers the VLM.
5. **Visual Grounding**: The system draws explicit targeting boxes on the anomalous pixels before sending the frame to the VLM, preventing hallucination.
6. **Unified Memory Execution**: The VLM executes natively on Apple Silicon unified memory using MLX, bypassing the need for cloud-based APIs.
