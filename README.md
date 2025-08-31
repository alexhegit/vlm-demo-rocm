# AMD ROCm™ Vision Language Model (VLM) Demo

This repository contains a web-based demonstration of a Vision Language Model (VLM) optimized for AMD ROCm™ hardware. The demo allows users to interact with a VLM that can analyze images and generate text responses based on visual content.

## What is this?

This project demonstrates a Vision Language Model that combines:
- Computer vision capabilities to analyze images
- Natural language processing to understand and respond to text prompts
- AMD ROCm™ hardware acceleration for improved performance

The demo uses the Qwen2.5-VL-3B-Instruct model, which is a multimodal model capable of understanding both text and images simultaneously.

## Features

- Real-time camera feed integration
- Text-based prompting for image analysis
- Configurable processing intervals
- Responsive web interface with AMD branding
- ROCm™ hardware acceleration support

## Prerequisites

Before using this demo, you'll need:
1. AMD GPU with ROCm™ support
2. ROCm™ platform installed on your system
3. Required model files (Qwen2.5-VL-3B-Instruct and mmproj files)
4. Web browser with camera access permissions

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/alexhegit/vlm-demo-rocm.git
cd vlm-demo-rocm
```

### 2. Prepare Model Files
Download the required model files:
- Qwen2.5-VL-3B-Instruct-Q8_0.gguf
- mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf

Place these files in the `~/GGUF/Qwen2.5-VL-3B-Instruct-GGUF/` directory.

### 3. Make Service Script Executable
```bash
chmod +x service.sh
```

## How to Use

### Start the VLM Service
```bash
./service.sh
```

This will start the llama-server with the VLM model, listening on port 8080.

### Access the Demo Interface
Open `demo.html` in your web browser:
```bash
xdg-open demo.html
```

Or alternatively, you can serve the file using a simple HTTP server:
```bash
python3 -m http.server 8000
```
Then navigate to `http://localhost:8000/demo.html` in your browser.

### Using the Demo

1. Allow camera access when prompted by the browser
2. The camera feed will appear in the interface
3. Enter your text prompt in the "Instruction" field (default is "What do you see?")
4. Configure the interval between requests using the dropdown menu
5. Click "Start" to begin processing
6. View the AI-generated responses in the "Response" area
7. Click "Stop" to halt processing

## Web Demo (Alternative Interface)

This repository also includes a more advanced web-based demo interface that allows browsing and analyzing images from a local directory.

### Features
- Browse images from the `images/` directory
- Run VLM inference on individual images
- View inference results with timing metrics
- Keyboard navigation support (arrow keys, spacebar)
- Responsive web interface with modern UI

### How to Use
1. Ensure the VLM service is running (`./service.sh`)
2. Place your images in the `images/` directory
3. Run the web demo:
   ```bash
   ./web_demo.sh
   ```
4. The web interface will automatically open in your browser at `http://localhost:5000`
5. Use the interface to browse images and run inference

### Requirements
- Python 3.x
- Flask and Flask-CORS Python packages
- Images in the `images/` directory
- VLM service running on `http://localhost:8080`

### Additional Notes
- The web demo will automatically detect and use images in the `images/` directory
- Results are logged to `inference-result.json`
- The demo supports keyboard navigation for quick image browsing
- Press spacebar to run inference on the currently displayed image

## Configuration

The service.sh script can be modified to use different models:
- Uncomment the alternative model lines to switch to SmolVLM2-500M-Video-Instruct
- Adjust the HSA_OVERRIDE_GFX_VERSION if needed for your specific GPU

## Troubleshooting

### Camera Access Issues
- Ensure you're using HTTPS or localhost (required for camera access)
- Check browser permissions for camera access
- Verify your camera is properly connected and recognized by the system

### Model Loading Problems
- Verify that the model files exist in the specified paths
- Ensure sufficient system memory for the model
- Check that ROCm™ is properly installed and configured

### API Connection Issues
- Confirm the service is running (`./service.sh`)
- Verify the service is listening on port 8080
- Test connectivity with: `curl http://localhost:8080/v1/chat/completions`

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Usage Guide

### Quick Start

1. **Ensure prerequisites are met**:
   - AMD GPU with ROCm™ support
   - ROCm™ platform installed
   - Model files downloaded and placed in the correct location

2. **Start the VLM service**:
   ```bash
   ./service.sh
   ```

3. **Open the demo interface**:
   ```bash
   xdg-open demo.html
   ```

4. **Use the demo**:
   - Allow camera access when prompted
   - Enter your text prompt in the "Instruction" field
   - Select processing interval
   - Click "Start" to begin analysis
   - View AI responses in the "Response" area

### Detailed Usage

The demo interface consists of several components:
- **Camera Feed**: Shows real-time video from your webcam
- **API Endpoint Field**: Configurable URL for the VLM service (default: http://localhost:8080)
- **Instruction Input**: Text prompt to guide the VLM analysis
- **Response Area**: Displays AI-generated responses
- **Controls**: Interval selector and Start/Stop button

When you click "Start", the system will:
1. Capture frames from your camera
2. Send each frame with your instruction to the VLM service
3. Display the AI's response in the response area
4. Continue at the selected interval until you click "Stop"

## Acknowledgments

- AMD ROCm™ Platform
- Qwen2.5-VL-3B-Instruct Model
- llama.cpp for the server implementation
