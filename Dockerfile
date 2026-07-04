FROM python:3.11-slim

# Install system-level dependencies for Edge AI
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir build pytest

# Copy SDK Source
COPY . .

# Run tests on build to guarantee integrity
RUN pytest tests/

# Set the entrypoint to the SDK
CMD ["python", "-c", "from fluxstate_security import FluxStateNode; sdk = FluxStateNode(); sdk.start_headless_daemon(); import time; time.sleep(86400)"]
