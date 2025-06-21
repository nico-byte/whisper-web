# Whisper Realtime Transcriber

## Table of Contents

- [Whisper Realtime Transcriber](#whisper-realtime-transcriber)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Quick Start](#quick-start)
    - [Model Installation and Management](#model-installation-and-management)
    - [Supported Model IDs](#supported-model-ids)
    - [Troubleshooting Model Installation](#troubleshooting-model-installation)
    - [How it works](#how-it-works)
    - [Documentation](#documentation)
  - [Deployment](#deployment)
    - [Overview](#overview)
    - [Project Structure](#project-structure)
  - [Environment Configuration](#environment-configuration)
    - [Environment Variables Template](#environment-variables-template)
    - [Makefile Tasks](#makefile-tasks)
  - [Deployment Instructions](#deployment-instructions)
    - [Prerequisites (Docker)](#prerequisites-docker)
    - [Quick Start](#quick-start-1)
    - [Individual Service Management](#individual-service-management)
    - [Profile-Based Deployments](#profile-based-deployments)
    - [Local Development (without Docker)](#local-development-without-docker)
  - [Troubleshooting](#troubleshooting)
    - [Build Issues](#build-issues)
    - [Service Status and Connectivity](#service-status-and-connectivity)
    - [Common Issues](#common-issues)
  - [Monitoring and Maintenance](#monitoring-and-maintenance)
    - [Health Checks](#health-checks)
  - [Security Considerations](#security-considerations)

## Project Overview

The Whisper Web Transcription Server is a Python-based real-time speech-to-text transcription system powered by
OpenAI's Whisper models. It leverages state-of-the-art models like Distil-Whisper to transcribe audio input
in real-time.

Key Features:
* Real-time Transcription: Transcribes spoken words into text almost instantaneously, making it ideal for
use cases like live captioning, real-time subtitles, or interactive voice-driven applications.
* Customizable Model Configurations: The system can be fine-tuned with various model configurations to
accommodate different use cases, such as adjusting the sampling rate, block size, or selecting from
different Whisper model sizes.
* API Integration: Exposes a RESTful API for easy integration with other services. You can send and retrieve
transcriptions via HTTP requests, allowing for real-time updates in web or mobile apps.
* Multi-threaded Asynchronous Processing: Leverages asynchronous programming (via asyncio) for optimal
performance, allowing the transcription engine to handle high volumes of audio input while processing
transcription results concurrently.
* Memory-Efficient and Scalable: Designed to work efficiently even with resource-intensive models, offering
scalable transcription performance with lower resource consumption.

By combining cutting-edge machine learning models and real-time audio processing, this project enables fast,
accurate, and flexible transcription solutions for various audio-driven applications.

### Prerequisites

- [Python 3.12](https://www.python.org) installed on the machine, either standalone or via a package manager like uv. We recommend to use [uv](https://docs.astral.sh/uv/getting-started/installation/).
- Optional: Microphone connected to the machine.

### Installation

Install dependencies via a package manager like uv, in project root:

```bash
uv sync --all-groups --extra cpu # or --extra cu128 for torch with CUDA support
```

Setup .env and model cache dir:

```bash
sh setup.sh
```

### Quick Start

After completing the installation, one can now use the transcriber.

For microphone input the cli client can be used. For File Uploads the streamlit upload client can be used.
There is also a streamlit viewer client, that shows all active sessions and their transcriptions on the server.

Simply start the server and cli/streamlit clients via the Makefile tasks:

```bash
make local.run.upload # for server plus streamlit upload app - doesn't support mic input
make local.run.cli # for server plus cli client - supports mic input
```

Alternatively one can run the docker images:

```bash
make docker.up.server-cpu-streamlit # for cpu server
make docker.up.server-cuda-streamlit # for cuda server
``` 

More on the Makefile tasks in the [Makefile Tasks](#makefile-tasks) section.

### Model Installation and Management

On first startup, no models are installed by default. You can install Whisper models accordingly:

1. Access the Streamlit Upload App at http://localhost:8501
2. In the model selection field, enter one of the supported model IDs listed below
3. Click "Create Session" to download and initialize the model
4. **Important**: Model downloading happens in the background without visual feedback. Session creation may take 2-10 minutes depending on your internet connection and the model size.

#### Supported Model IDs

Whisper Web supports all whisper models available at huggingface but here are some reccomendations:

**English Models:**
- `distil-whisper/distil-large-v3.5` - **Recommended** - Best accuracy, larger size (~1.5GB)
- `distil-whisper/distil-medium.en` - Good balance of speed and accuracy (~760MB)
- `distil-whisper/distil-small.en` - Fastest, smallest size (~240MB)

**German Models:**
- `primeline/distil-whisper-large-v3-german` - German-optimized model (~1.5GB)

**Multilingual Models:**
- `openai/whisper-tiny` - Ultra-fast, 39MB, supports 99 languages
- `openai/whisper-base` - Fast, 140MB, supports 99 languages
- `openai/whisper-small` - Balanced, 460MB, supports 99 languages

#### Troubleshooting Model Installation

**If session creation takes too long:**
- Check your internet connection
- Monitor server logs: `docker compose logs -f server-cpu`
- Verify disk space availability in the `.models` directory

**If model download fails:**
- Ensure the model ID is spelled correctly
- Check HuggingFace Hub connectivity: `curl -I https://huggingface.co`
- Verify Docker container has internet access

**Storage requirements:**
- Reserve at least 2GB free disk space for model storage
- Models are cached in `.models/` directory and persist between restarts
- Use `du -sh .models/` to check current model storage usage

### How it works

- The transcriber consists of two modules: a Inputstream Generator and a Whisper Model.
- The implementation of the Inputstream Generator is based on this [implementation](https://github.com/tobiashuttinger/openai-whisper-realtime).
- The Inputstream Generator reads the microphone input and passes it over an event bus to the Whisper Model. The Whisper Model then generates the transcriptions and passes them via the event bus to the server.
- This is happening in an async event loop so that the Whisper Model can continuously generate
transcriptions from the provided audio input, generated and processed by the Inputstream Generator.

### Documentation

Documentation can be found [here](https://nico-byte.github.io/whisper-web/).

## Deployment

This document provides comprehensive instructions for deploying the Whisper Web transcription system using Docker containers with both CPU and CUDA GPU support.

### Overview

The Whisper Web system consists of five main components:
- **Whisper Server (CPU/CUDA)**: Core transcription service with WebSocket support, available in both CPU and CUDA variants
- **Streamlit Upload App**: Web interface for file uploads and WebRTC streaming
- **Streamlit Viewer App**: Read-only transcription viewer
- **CLI Client**: Command-line client with microphone access
- **Docker Compose Profiles**: Flexible deployment configurations for different use cases

### Project Structure

```
whisper-web/
├── app/
│   ├── __init__.py
│   ├── cli.py
|   ├── helper.py
│   ├── server.py
│   ├── streamlit_upload_client.py
│   └── streamlit_viewer_client.py
├── docker/
│   ├── DOCKERFILE.server.cpu
│   ├── DOCKERFILE.server.cuda
│   ├── DOCKERFILE.streamlit_upload
│   └── DOCKERFILE.streamlit_viewer
├── tests/
│   ├── test_api_pytest.py
│   ├── test_multi_client_pytest.py
├── whisper_web/
│   └── [core package files]
├── .dockerignore
├── .env
├── .env.example
├── .gitignore
├── .python-version
├── docker-compose.yml
├── install_uv.sh
├── LICENSE
├── Makefile
├── pyproject.toml
├── pytest.ini
├── README.md
├── ruff.toml
├── setup.sh
└── uv.lock
```

## Environment Configuration

### Environment Variables Template
Create a `.env` file in the project root with the following variables, if not done via setup.sh:

```env
# Server Configuration
SERVER_TYPE=cpu  # or 'cuda' for GPU acceleration
HOST=0.0.0.0
CUDA_VISIBLE_DEVICES=0
HF_HOME=./.models

# Streamlit variables
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ADDRESS=${HOST}
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
STREAMLIT_UPLOAD_PORT=8501
STREAMLIT_VIEWER_PORT=8502

# Network Configuration
DOCKER_NETWORK_NAME=whisper-web
```

### Makefile Tasks
The project includes a comprehensive Makefile for common operations:

```makefile
# Development tasks
make format         # Format code with ruff
make lint           # Check code with ruff
make lintfix        # Fix linting issues
make pytest         # Run tests

# Docker tasks
make docker.up.server-cpu-streamlit   # Run cpu server + streamlit apps
make docker.up.server-cuda-streamlit  # Run cuda server ++ streamlit apps
make docker.build.server-cpu          # Build cpu server
make docker.build.server-cuda         # Build cuda server
make docker.build.streamlit           # Build streamlit apps
make docker.build.all-cpu             # Build all images (cpu server)
make docker.build.all-cuda            # Build all images (cuda server)

# Local development tasks
make local.server     # Run server locally
make local.cli        # Run CLI locally
make local.viewer     # Run viewer locally
make local.upload     # Run upload app locally
make local.run.cli    # Run server + CLI + streamlit viewer app
make local.run.upload # Run server + streamlit upload app
```

## Deployment Instructions

### Prerequisites (Docker)
- Docker 20.10+ with Docker Compose V2
- At least 4GB RAM available (8GB+ recommended)
- For GPU support: NVIDIA Docker runtime and CUDA-compatible GPU
- Python 3.12 (for local development)
- uv package manager (recommended)

### Quick Start

1. **Prepare the environment:**
```bash
cp .env.example .env
# Edit .env file to configure your deployment
```

2. **Build all Docker images:**
```bash
make docker.build.all-cpu # or ...all-cuda if driver is present
```

3. **Start services based on your hardware:**

**CPU-only deployment with Streamlit interface:**
```bash
make docker.up.server-cpu-streamlit
```

**GPU-accelerated deployment (requires NVIDIA GPU):**
```bash
make docker.up.server-cuda-streamlit
```

1. **Access the applications:**
- **Whisper Web Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Streamlit Upload App**: http://localhost:8501 
- **Streamlit Viewer App**: http://localhost:8502

### Individual Service Management

```bash
# Start only CPU server
SERVER_TYPE=cpu docker compose --profile cpu up -d

# Start CUDA server + Streamlit apps
SERVER_TYPE=cuda docker compose --profile cuda --profile streamlit up -d
```

### Profile-Based Deployments

The docker-compose configuration uses profiles for flexible deployments:

- `cpu`: CPU-only server
- `cuda`: GPU-accelerated server  
- `streamlit`: Streamlit web interfaces

**Common deployment combinations:**
```bash
# Web interface with CPU server
SERVER_TYPE=cpu docker compose --profile cpu --profile streamlit up -d

# Web interface with GPU server
SERVER_TYPE=cuda docker compose --profile cuda --profile streamlit up -d
```

### Local Development (without Docker)

For development, you can run services locally using uv:

```bash
# Install dependencies
uv sync

# Start individual services
make local.server    # Whisper Web server
make local.cli       # CLI client
make local.upload    # Streamlit upload app
make local.viewer    # Streamlit viewer app

# Start combined services
make local.run.cli     # Server + CLI + Viewer
make local.run.upload  # Server + Upload app
```

## Troubleshooting

### Build Issues

**CUDA build failures:**
```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi

# Check CUDA availability
nvidia-smi
```

### Service Status and Connectivity

1. **Check service status:**
```bash
docker compose ps
curl -f http://localhost:8000/status
```

2. **Network connectivity issues:**
```bash
# Test internal network connectivity
docker compose exec streamlit-upload curl -f http://server-cpu:8000/status

# Check port bindings
docker compose port server-cpu 8000
```

### Common Issues

1. **Models not downloading:** 
   - Check internet connection and disk space
   - Verify `.models` directory permissions
   - Check HuggingFace Hub connectivity

2. **GPU not detected:** 
   - Verify NVIDIA Docker runtime installation
   - Check CUDA_VISIBLE_DEVICES environment variable
   - Ensure GPU has sufficient memory (4GB+ recommended)

3. **Streamlit apps not loading:** 
   - Verify server status and connectivity
   - Check firewall settings for ports 8501/8502

4. **Port conflicts:** 
   - Modify port mappings in docker-compose.yml or .env file
   - Check for other services using ports 8000, 8501, 8502

5. **Memory issues:**
   - Increase Docker memory limits
   - Use smaller Whisper models (tiny, base, small)
   - Monitor container resource usage with `docker stats`

## Monitoring and Maintenance

### Health Checks
All services include comprehensive health checks:

**Server Health:**
```bash
# Check server status
curl -f http://localhost:8000/status
curl -f http://localhost:8000/sessions

# Monitor via Docker
docker compose ps
docker compose logs -f server-cpu  # or server-cuda
```

**Streamlit App Health:**
```bash
# Check Streamlit health endpoints
curl -f http://localhost:8501/_stcore/health
curl -f http://localhost:8502/_stcore/health
```

**System Resource Monitoring:**
```bash
# Monitor container resource usage
docker stats

# Check disk usage for models
du -sh .models/
docker system df
```

## Security Considerations

1. **Network isolation:** Services communicate through internal Docker network
2. **Volume mounts:** Only necessary directories are mounted
3. **User permissions:** Client runs as non-root user when possible

This deployment provides a complete, scalable solution for the Whisper Web transcription system with proper containerization and networking.

## TODOs
  - Fix runtime error when using multilingual Whisper models:
    RuntimeError: Tried to instantiate class '__path__._path', but it does not exist!
    Ensure the model is correctly registered with torch::class_.
    Also resolve input shape errors like
    Whisper expects the mel input features to be of length 3000, but found 2500
    by padding mel features to 3000, only important for multilingual model support.