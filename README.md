# Beautiful Batches AI Engine

**The core AI backend for [Beautiful Batches](https://github.com/supSugam/beautiful-batches)**

This module provides specialized AI processing capabilities as an integrated addon for the Beautiful Batches desktop application. It handles complex image manipulation tasks using local machine learning models, specifically optimized for watermark removal and background segmentation.

---

## Core Capabilities

### Watermark & Object Removal
- **Detection**: Powered by **Microsoft Florence-2** for high-accuracy identification of logos, text, and watermarks.
- **Inpainting**: Utilizes the **LaMA (Large Mask Inpainting)** model to reconstruct background details with structural consistency.

### Background Removal
- **Segmentation**: Uses **rembg (ISNet)** to isolate primary subjects from their backgrounds.
- **Transparency**: Outputs high-quality alpha channels directly to PNG format.

### Hardware Acceleration
The engine automatically selects the most efficient execution provider for your hardware:
- **CUDA**: Nvidia GPUs
- **CoreML**: Apple Silicon (M1/M2/M3)
- **DirectML/OpenVINO**: Windows and Intel/AMD integrated graphics (Turbo Mode)

---

## Technical Integration

This engine operates as a sidecar process. The Beautiful Batches application communicates with it through a JSON-RPC bridge provided in `bridge.py`.

### Communication Protocol
The bridge listens on `stdin` for JSON commands:
- `load`: Initializes the vision and inpainting models.
- `process`: Executes the detection and removal pipeline on a specific file path.
- `remove_bg`: Executes the background removal pipeline.
- `ping`: Health check and model status.

---

## Setup for Development

### Prerequisites
- Python 3.10+
- FFmpeg (required for video processing extensions)

### Installation
```bash
git clone https://github.com/supSugam/beautiful-batches.git
cd beautiful-batches/WatermarkRemover-AI
./setup.sh # or .\setup.ps1 for Windows
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
