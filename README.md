<div align="center">

<img src="https://raw.githubusercontent.com/contact-us-gdinexus/fluxstate/main/assets/images/hero-banner-v1.1.png" alt="FluxState Edge Hero Banner" width="100%">

# 🦅 FluxState Edge SDK

**Privacy-Preserving Contextual Edge Video Analytics for Enterprise Security**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge&color=2e2e2e)](https://opensource.org/licenses/MIT)
[![Tests: Passing](https://img.shields.io/badge/Tests-Passing-brightgreen.svg?style=for-the-badge&color=2e2e2e)]()

<br>

*FluxState Edge* is an extensible, camera-agnostic video analytics SDK designed for direct integration into proprietary security backends. It processes RTSP streams locally on the edge, extracting behavioral metadata via object detection, 3D skeletal posing, audio processing, and VLM semantic reasoning with our v1.1 Adaptive Intelligence layer.

</div>

---

<br>

## 🚀 Quick Install

```bash
pip install fluxstate-security
```

## 🎥 Installation Demo

See the SDK installation and verification in under a minute.

▶️ [Watch Installation Demo (Google Drive)](https://drive.google.com/file/d/1_KX3aYMqBTP7uCZmRAn4qMwj1M6CpZxd/view?usp=sharing)

<br>

## ⚡ System Capabilities

| Capability | Technical Implementation |
| :--- | :--- |
| 🧠 **Agentic VLM Reasoner** | Runs Vision-Language Models (e.g., `Qwen2.5-VL`) locally via MLX on Apple Silicon unified memory for deep contextual scene understanding. Bypasses cloud APIs entirely. |
| 🔄 **Adaptive Intelligence (v1.1)** | Seamlessly loads and unloads MLX models (`MLXModelPool`) to manage unified memory. Provides a Feedback API to suppress false positives on-device without retraining. |
| 📚 **Semantic Retrieval (RAG)** | Embeds scene telemetry and queries an on-device `ContextLedger` using cosine similarity to inject historical context into the VLM prompt. |
| ⏱️ **Episodic Memory** | Buffers the last 10 minutes of scene states (`EpisodicMemoryBuffer`) with O(1) operations, allowing delayed asynchronous operator feedback. |
| 🎯 **Tactical Visual Grounding** | Automatically injects high-contrast red bounding boxes over anomalies. This guides the VLM toward the detected region, reducing irrelevant reasoning and improving contextual grounding. |
| 🛡️ **Privacy-by-Design** | Actively zeroes out image buffers post-inference via C-level `memset`. Designed with privacy-first principles to prevent sensitive pixel data from lingering in the system heap. |
| 🗄️ **Temporal Forensics** | Behavioral anomalies are serialized into a local SQLite database (`fluxstate_security/core/forensics.py`), creating a searchable text-based ledger of physical events. |
| 📹 **Hardware Agnostic** | Ingests existing IP cameras via standard RTSP URLs. No proprietary recording hardware required. |
| 🐳 **Edge Containerization** | Ships with a highly optimized `Dockerfile` for enterprise edge deployments (Kubernetes/Docker Swarm), permanently locking native OS dependencies. |

<br>

## 🖼️ Architectural Vision & Use Cases

<div align="center">
  <img src="https://raw.githubusercontent.com/contact-us-gdinexus/fluxstate/main/assets/images/architecture-v1.1.png" alt="FluxState Architecture Diagram" width="90%">
  <br><br>
  <img src="https://raw.githubusercontent.com/contact-us-gdinexus/fluxstate/main/assets/images/system-overview-v1.1.png" alt="System Overview" width="90%">
</div>

### Sub-Pipelines

<div align="center">
  <img src="https://raw.githubusercontent.com/contact-us-gdinexus/fluxstate/main/assets/images/adaptive-pipeline.png" alt="Adaptive Pipeline" width="30%">
  <img src="https://raw.githubusercontent.com/contact-us-gdinexus/fluxstate/main/assets/images/rag-flow.png" alt="RAG Flow" width="30%">
  <img src="https://raw.githubusercontent.com/contact-us-gdinexus/fluxstate/main/assets/images/feedback-loop.png" alt="Feedback Loop" width="30%">
  <br><br>
  <img src="https://raw.githubusercontent.com/contact-us-gdinexus/fluxstate/main/assets/images/tech-stack.png" alt="Tech Stack" width="90%">
</div>

<br>

## 🛠️ Minimal SDK Integration

FluxState is designed to fade into the background. Drop it into your existing backend and attach a webhook.

```python
import time
from fluxstate_security import FluxStateNode

# 1. Initialize the SDK
sdk = FluxStateNode()

# 2. Define your integration hook
def handle_threat(event_payload):
    print(f"\n[INTEGRATION BUS] Escalating to VMS...")
    print(f"Target Identity: {event_payload['entities']}")
    print(f"Behavioral Vector: {event_payload['context_log']}")
    print(f"Feedback Event ID: {event_payload.get('event_id', 'None')}")

# 3. Bind the hook
sdk.on_threat_detected = handle_threat

# 4. Deploy Headlessly (Runs as a background daemon)
sdk.start_headless_daemon()
sdk.start_api_server(port=8000) # Starts the JSON API & Feedback loop

try:
    while True: time.sleep(1)
except KeyboardInterrupt:
    sdk.stop() # 5. Clean shutdown
```

<br>

## 📦 Deployment Strategies

<details>
<summary><b>Option A: Enterprise Docker Deployment (Recommended)</b></summary>
<br>
For production environments, use the provided Docker container to guarantee system dependencies (Tesseract, PortAudio) are perfectly locked across the cluster.

```bash
docker build -t fluxstate-security .
docker run -d --name fluxstate-security fluxstate-security
```
</details>

<details>
<summary><b>Option B: Local Python Development</b></summary>
<br>

```bash
pip install fluxstate-security

# macOS dependencies
brew install tesseract portaudio

# Linux dependencies
sudo apt-get install tesseract-ocr libportaudio2 libportaudiocpp0 portaudio19-dev
```
</details>

<br>

## 🧪 Forensics & Testing
FluxState ships with an automated `pytest` suite covering the Forensic SQLite ledger, semantic retrievers, and JSON intelligence policies.
```bash
pytest tests/
```

---

## 📡 API Reference

### `POST /api/v2/feedback`
Submit operator feedback to adapt the system.
**Payload:**
```json
{
  "event_id": "uuid-string",
  "human_label": "FALSE_POSITIVE",
  "operator_id": "guard-123",
  "operator_role": "admin"
}
```

---

<div align="center">
  <i>For a deeper dive into the threading model, VLM orchestration, and the SQLite schema, see <a href="https://github.com/contact-us-gdinexus/fluxstate/blob/main/architecture_adaptive_edge.md">architecture_adaptive_edge.md</a>.</i>
</div>

---

<div align="center">
  <b>⭐ <a href="https://github.com/contact-us-gdinexus/fluxstate">GitHub Repository</a></b> • 
  <b>📦 <a href="https://pypi.org/project/fluxstate-security/">PyPI Package</a></b> • 
  <b>📖 <a href="https://github.com/contact-us-gdinexus/fluxstate/blob/main/architecture_adaptive_edge.md">Documentation</a></b>
</div>
