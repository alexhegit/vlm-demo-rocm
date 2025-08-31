#!/usr/bin/env python3
"""
Web-based VLM inference demo that displays images one by one
"""

import os
import random
import base64
import requests
import time
import json
import re
import argparse
from datetime import datetime
import tempfile
import subprocess
import sys
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import threading
import webbrowser
import socket

# Configuration
IMAGE_DIR = "./images"  # Directory path of images
BASE_URL = "http://localhost:8080"  # Your API endpoint
INSTRUCTION = "Describe the images in 80 words within one sentence"  # Prompt for inference
LOG_FILE = "inference-result.json"  # Log file name
WEB_PORT = 5000  # Web server port

# Global tokenizer variable
tokenizer = None

# Import tokenizer libraries
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

def load_tokenizer(tokenizer_path=None):
    """Load tokenizer from local directory"""
    global tokenizer
    
    if tokenizer_path and os.path.exists(tokenizer_path) and TRANSFORMERS_AVAILABLE:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            print(f"Loaded tokenizer from local path: {tokenizer_path}")
            return True
        except Exception as e:
            print(f"Failed to load tokenizer from {tokenizer_path}: {e}")
            return False
    elif tokenizer_path and not TRANSFORMERS_AVAILABLE:
        print("transformers library not available. Install with: pip install transformers")
        return False
    else:
        print("No tokenizer path provided. Using approximate token counting.")
        return False

def count_tokens_with_tokenizer(text):
    """Count tokens using loaded tokenizer"""
    if tokenizer is None:
        return count_tokens_approximately(text)
    
    try:
        if hasattr(tokenizer, 'encode'):  # transformers tokenizer
            return len(tokenizer.encode(text))
        elif hasattr(tokenizer, 'encode_ordinary'):  # tiktoken tokenizer
            return len(tokenizer.encode_ordinary(text))
        else:
            return count_tokens_approximately(text)
    except Exception as e:
        print(f"Token counting error: {e}, falling back to approximation")
        return count_tokens_approximately(text)

def count_tokens_approximately(text):
    """Approximate token counting method"""
    if not text:
        return 0
    
    # Enhanced word-based approximation
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text.strip())
    
    # Apply rough correction factor for subword tokenization
    word_count = len([t for t in tokens if re.match(r'\w+', t)])
    punct_count = len(tokens) - word_count
    
    # Rough estimation: average 1.0 tokens per word for subword tokenization
    estimated_tokens = int(word_count * 1.0) + punct_count
    
    return estimated_tokens

def get_random_images(num_images):
    """Select random images and return their info"""
    if not os.path.exists(IMAGE_DIR):
        print(f"Directory {IMAGE_DIR} does not exist")
        return []

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.jpg')]
    if not image_files:
        print(f"No .jpg files found in directory {IMAGE_DIR}")
        return []

    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    
    image_info = []
    for img_file in selected_images:
        img_path = os.path.join(IMAGE_DIR, img_file)
        try:
            with open(img_path, "rb") as img_file:
                img_bytes = img_file.read()
            img_str = base64.b64encode(img_bytes).decode('utf-8')
            image_info.append({
                'path': img_path,
                'filename': os.path.basename(img_path),
                'base64': f"data:image/jpeg;base64,{img_str}"
            })
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            continue

    return image_info

def send_request(instruction, image_base64_list):
    """Send a request with multiple images and return the response"""
    payload = {
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    *[{"type": "image_url", "image_url": {"url": img}} for img in image_base64_list]
                ]
            }
        ],
        "stream": True  # Enable streaming response
    }

    start_time = time.time()
    first_token_time = None
    full_response = ""
    completion_tokens = 0
    prompt_tokens = 0
    total_tokens = 0

    try:
        with requests.Session() as session:
            response = session.post(f"{BASE_URL}/v1/chat/completions", json=payload, stream=True, timeout=60)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data:'):
                        data = line[6:].strip()
                        if data == '[DONE]':
                            break
                        try:
                            data_json = json.loads(data)
                            
                            # Extract content from delta
                            content = data_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                if first_token_time is None:
                                    first_token_time = time.time() - start_time
                                full_response += content
                            
                            # Extract token usage information if available
                            usage = data_json.get("usage")
                            if usage:
                                completion_tokens = usage.get("completion_tokens", 0)
                                prompt_tokens = usage.get("prompt_tokens", 0)
                                total_tokens = usage.get("total_tokens", 0)
                                
                        except json.JSONDecodeError:
                            continue
            
            latency = time.time() - start_time
            
            # Use API-provided token count if available, otherwise fallback to tokenizer
            if completion_tokens > 0:
                tokens = completion_tokens
            else:
                tokens = count_tokens_with_tokenizer(full_response)
            
            return {
                'success': True,
                'ttft': first_token_time,
                'latency': latency,
                'tokens': tokens,
                'content': full_response.strip(),
                'token_usage': {
                    'completion_tokens': completion_tokens,
                    'prompt_tokens': prompt_tokens,
                    'total_tokens': total_tokens
                }
            }
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
        return {'success': False, 'error': str(e)}
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return {'success': False, 'error': str(e)}

def write_log(inference_no, result, image_paths):
    """Write the inference result to the JSON file"""
    if not result['success']:
        log_entry = {
            "inference_no": inference_no,
            "success": False,
            "error": result.get('error', 'Unknown error'),
            "image_paths": image_paths,
            "timestamp": datetime.now().isoformat()
        }
    else:
        log_entry = {
            "inference_no": inference_no,
            "success": True,
            "ttft": round(result['ttft'], 4) if result['ttft'] else None,
            "latency": round(result['latency'], 4) if result['latency'] else None,
            "tokens": result['tokens'],
            "content": result['content'],
            "image_paths": image_paths,
            "timestamp": datetime.now().isoformat()
        }
    
    # If the file does not exist, create and write the first line
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([log_entry], f, ensure_ascii=False, indent=2)
    else:
        # Read existing data, append new result, and write back
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Remove stats if they exist (they'll be added at the end)
            if data and isinstance(data[-1], dict) and "total_latency" in data[-1]:
                data.pop()
            
            data.append(log_entry)
            
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, FileNotFoundError):
            # If file is corrupted or doesn't exist, create new
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                json.dump([log_entry], f, ensure_ascii=False, indent=2)

# Flask App
app = Flask(__name__)
CORS(app)

# Global state
current_inference = 0
last_inference_result = None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/start_inference', methods=['POST'])
def start_inference():
    """Start a new inference session"""
    global current_inference
    
    current_inference += 1
    
    return jsonify({'success': True, 'inference_id': current_inference})

@app.route('/api/get_images/<int:inference_id>')
def get_images(inference_id):
    """Get images for a specific inference"""
    images = get_random_images(9)  # Get up to 9 random images
    return jsonify({
        'success': True,
        'inference_id': inference_id,
        'images': images  # Return all images for sequential viewing
    })

@app.route('/api/run_inference', methods=['POST'])
def run_inference():
    """Run inference on selected images"""
    global last_inference_result
    
    data = request.get_json()
    image_base64_list = data.get('images', [])
    inference_id = data.get('inference_id')
    
    if not image_base64_list:
        return jsonify({'success': False, 'error': 'No images provided'})
    
    # Run inference
    result = send_request(INSTRUCTION, image_base64_list)
    
    # Log the result
    image_paths = [f"Image {i+1}" for i in range(len(image_base64_list))]
    write_log(inference_id, result, image_paths)
    
    # Store only the last result
    last_inference_result = {
        'inference_id': inference_id,
        'result': result,
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify({
        'success': True,
        'inference_id': inference_id,
        'result': result
    })


@app.route('/api/results')
def get_results():
    """Get the last inference result"""
    return jsonify({
        'success': True,
        'result': last_inference_result
    })


def is_port_in_use(port):
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except socket.error:
            return True

def open_browser():
    """Open browser to the web interface"""
    # Wait a moment for the server to start
    time.sleep(1.5)
    webbrowser.open(f'http://localhost:{WEB_PORT}')

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Web-based VLM inference demo")
    parser.add_argument("-t", "--tokenizer-path", type=str, help="Path to local tokenizer directory")
    parser.add_argument("-p", "--port", type=int, default=WEB_PORT, help="Web server port")
    args = parser.parse_args()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer_loaded = load_tokenizer(args.tokenizer_path)

    print(f"Starting web-based VLM inference demo")
    print(f"API endpoint: {BASE_URL}")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Log file: {LOG_FILE}")
    print(f"Web port: {args.port}")
    if tokenizer_loaded:
        print(f"Tokenizer: Loaded from {args.tokenizer_path}")
    else:
        print("Tokenizer: Using approximate counting")
    print("=" * 50)

    # Check if port is available
    if is_port_in_use(args.port):
        print(f"Port {args.port} is already in use. Please use a different port with -p option.")
        return

    # Start browser in background
    threading.Thread(target=open_browser, daemon=True).start()

    # Start Flask app
    app.run(host='localhost', port=args.port, debug=False)

if __name__ == "__main__":
    main()