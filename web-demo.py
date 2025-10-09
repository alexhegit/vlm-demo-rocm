#!/usr/bin/env python3
"""
Web-based VLM inference demo that displays images one by one
Fixed version with improved error handling and thread safety
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
from typing import List, Dict, Optional, Any
import threading
import webbrowser
import socket
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# Configuration Constants
IMAGE_DIR = "./images"
BASE_URL = "http://localhost:8080"
INSTRUCTION = "Describe the images in 80 words within one sentence"
LOG_FILE = "inference-result.json"
WEB_PORT = 5000
MAX_IMAGES = 9
MAX_TOKENS = 100
REQUEST_TIMEOUT = 60

# Global variables with thread safety
tokenizer = None
inference_lock = threading.Lock()
log_lock = threading.Lock()
current_inference = 0
last_inference_result = None

# Import tokenizer libraries
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def load_tokenizer(tokenizer_path: Optional[str] = None) -> bool:
    """Load tokenizer from local directory"""
    global tokenizer
    
    if not tokenizer_path:
        print("No tokenizer path provided. Using approximate token counting.")
        return False
    
    if not TRANSFORMERS_AVAILABLE:
        print("transformers library not available. Using approximate token counting.")
        return False
    
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer path does not exist: {tokenizer_path}")
        return False
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"✓ Loaded tokenizer from: {tokenizer_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to load tokenizer from {tokenizer_path}: {e}")
        return False


def count_tokens_with_tokenizer(text: str) -> int:
    """Count tokens using loaded tokenizer"""
    if tokenizer is None:
        return count_tokens_approximately(text)
    
    try:
        if hasattr(tokenizer, 'encode'):
            return len(tokenizer.encode(text))
        elif hasattr(tokenizer, 'encode_ordinary'):
            return len(tokenizer.encode_ordinary(text))
        else:
            return count_tokens_approximately(text)
    except Exception as e:
        print(f"Token counting error: {e}, falling back to approximation")
        return count_tokens_approximately(text)


def count_tokens_approximately(text: str) -> int:
    """Approximate token counting method"""
    if not text:
        return 0
    
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text.strip())
    word_count = len([t for t in tokens if re.match(r'\w+', t)])
    punct_count = len(tokens) - word_count
    
    # Rough estimation: average 1.0 tokens per word for subword tokenization
    estimated_tokens = int(word_count * 1.0) + punct_count
    
    return estimated_tokens


def get_random_images(num_images: int) -> List[Dict[str, str]]:
    """Select random images and return their info"""
    if not os.path.exists(IMAGE_DIR):
        print(f"✗ Directory {IMAGE_DIR} does not exist")
        return []

    image_files = [f for f in os.listdir(IMAGE_DIR) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"✗ No image files found in directory {IMAGE_DIR}")
        return []
    
    # 确保不会尝试获取超过实际文件数量的图像
    actual_num = min(num_images, len(image_files))
    if actual_num < num_images:
        print(f"⚠ Warning: Only {len(image_files)} images available, requested {num_images}")
    
    selected_images = random.sample(image_files, actual_num)
    print(f"📸 Selected {actual_num} random images from {len(image_files)} total")
    
    image_info = []
    for img_filename in selected_images:
        img_path = os.path.join(IMAGE_DIR, img_filename)
        try:
            with open(img_path, "rb") as img_file_handle:
                img_bytes = img_file_handle.read()
            
            # Check file size (limit to 10MB)
            if len(img_bytes) > 10 * 1024 * 1024:
                print(f"⚠ Skipping {img_filename}: file too large (>10MB)")
                continue
            
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            image_info.append({
                'path': img_path,
                'filename': img_filename,
                'base64': f"data:image/jpeg;base64,{img_base64}",
                'size': len(img_bytes)
            })
            print(f"   ✓ Loaded {img_filename} ({len(img_bytes) / 1024:.1f} KB)")
        except Exception as e:
            print(f"✗ Error reading image {img_path}: {e}")
            continue

    print(f"✓ Successfully loaded {len(image_info)} images")
    return image_info


def send_request(instruction: str, image_base64_list: List[str]) -> Dict[str, Any]:
    """Send a request with multiple images and return the response"""
    if not image_base64_list:
        return {'success': False, 'error': 'No images provided'}
    
    payload = {
        "max_tokens": MAX_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    *[{"type": "image_url", "image_url": {"url": img}} 
                      for img in image_base64_list]
                ]
            }
        ],
        "stream": True
    }

    start_time = time.time()
    first_token_time = None
    full_response = ""
    completion_tokens = 0
    prompt_tokens = 0
    total_tokens = 0
    
    print(f"\n🔄 Sending request to {BASE_URL}/v1/chat/completions")
    print(f"   Images: {len(image_base64_list)}")
    print(f"   Max tokens: {MAX_TOKENS}")

    try:
        with requests.Session() as session:
            response = session.post(
                f"{BASE_URL}/v1/chat/completions", 
                json=payload, 
                stream=True, 
                timeout=REQUEST_TIMEOUT
            )
            
            print(f"   Response status: {response.status_code}")
            
            # Check for non-200 status
            if response.status_code != 200:
                error_text = response.text[:500]
                print(f"   ✗ Error response: {error_text}")
                return {
                    'success': False, 
                    'error': f'HTTP {response.status_code}: {error_text}'
                }
            
            line_count = 0
            error_count = 0
            
            for line in response.iter_lines():
                if not line:
                    continue
                
                line_count += 1
                
                try:
                    line_str = line.decode('utf-8')
                except UnicodeDecodeError as e:
                    print(f"   ⚠ Line {line_count}: Decode error: {e}")
                    continue
                
                # Skip non-data lines
                if not line_str.startswith('data:'):
                    if line_count <= 3:
                        print(f"   Skipping line {line_count}: {line_str[:50]}...")
                    continue
                
                data = line_str[6:].strip()
                
                # Check for [DONE]
                if data == '[DONE]':
                    print(f"   ✓ Received [DONE] signal")
                    break
                
                # Skip empty data
                if not data:
                    continue
                
                # Parse JSON
                try:
                    data_json = json.loads(data)
                except json.JSONDecodeError as e:
                    error_count += 1
                    if error_count <= 3:
                        print(f"   ⚠ JSON error at line {line_count}: {e}")
                        print(f"      Raw: {data[:100]}...")
                    continue
                
                # Debug first response
                if line_count == 1:
                    print(f"   Response keys: {list(data_json.keys())}")
                    if 'choices' in data_json:
                        print(f"   Choices count: {len(data_json.get('choices', []))}")
                
                # Safely extract content
                try:
                    choices = data_json.get("choices")
                    
                    # Check if choices exists and is not empty
                    if not choices:
                        continue
                    
                    if not isinstance(choices, list):
                        print(f"   ⚠ Warning: 'choices' is not a list: {type(choices)}")
                        continue
                    
                    if len(choices) == 0:
                        continue
                    
                    # Get first choice
                    choice = choices[0]
                    if not isinstance(choice, dict):
                        print(f"   ⚠ Warning: choice is not a dict: {type(choice)}")
                        continue
                    
                    # Get delta
                    delta = choice.get("delta")
                    if not delta or not isinstance(delta, dict):
                        continue
                    
                    # Get content
                    content = delta.get("content")
                    if content and isinstance(content, str):
                        if first_token_time is None:
                            first_token_time = time.time() - start_time
                            print(f"   ✓ First token at {first_token_time:.3f}s")
                        full_response += content
                    
                    # Extract token usage
                    usage = data_json.get("usage")
                    if usage and isinstance(usage, dict):
                        completion_tokens = usage.get("completion_tokens", 0)
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        total_tokens = usage.get("total_tokens", 0)
                        
                except (KeyError, TypeError, AttributeError) as e:
                    error_count += 1
                    if error_count <= 3:
                        print(f"   ⚠ Parsing error at line {line_count}: {e}")
                        print(f"      Structure: {json.dumps(data_json, indent=2)[:300]}...")
                    continue
            
            latency = time.time() - start_time
            
            # Use API-provided token count if available, otherwise fallback to tokenizer
            if completion_tokens > 0:
                tokens = completion_tokens
            else:
                tokens = count_tokens_with_tokenizer(full_response)
            
            print(f"   ✓ Complete in {latency:.2f}s")
            print(f"   Tokens: {tokens}, Response length: {len(full_response)} chars")
            print(f"   Total lines processed: {line_count}, Errors: {error_count}")
            
            if not full_response:
                print(f"   ⚠ Warning: No response content received!")
                return {
                    'success': False, 
                    'error': 'No content in API response. Check API server logs for errors.'
                }
            
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
            
    except requests.exceptions.Timeout:
        print(f"   ✗ Request timeout after {REQUEST_TIMEOUT}s")
        return {'success': False, 'error': f'Request timeout after {REQUEST_TIMEOUT}s'}
    except requests.exceptions.ConnectionError as e:
        print(f"   ✗ Connection error: {e}")
        return {'success': False, 'error': f'Cannot connect to API server at {BASE_URL}'}
    except requests.exceptions.HTTPError as e:
        print(f"   ✗ HTTP error: {e}")
        return {'success': False, 'error': f'HTTP error: {str(e)}'}
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Request failed: {e}")
        return {'success': False, 'error': f'Request failed: {str(e)}'}
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': f'Unexpected error: {str(e)}'}


def write_log(inference_no: int, result: Dict[str, Any], image_paths: List[str]) -> None:
    """Write the inference result to the JSON file (thread-safe)"""
    with log_lock:
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
        
        # Read existing data or create new
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Remove stats if they exist
                if data and isinstance(data[-1], dict) and "total_latency" in data[-1]:
                    data.pop()
            except (json.JSONDecodeError, FileNotFoundError):
                data = []
        else:
            data = []
        
        data.append(log_entry)
        
        # Write back
        try:
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"✗ Error writing log file: {e}")


# Flask App
app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/start_inference', methods=['POST'])
def start_inference():
    """Start a new inference session (thread-safe)"""
    global current_inference
    
    with inference_lock:
        current_inference += 1
        inference_id = current_inference
    
    return jsonify({'success': True, 'inference_id': inference_id})


@app.route('/api/get_images/<int:inference_id>')
def get_images(inference_id: int):
    """Get images for a specific inference"""
    if inference_id <= 0:
        return jsonify({'success': False, 'error': 'Invalid inference ID'}), 400
    
    print(f"\n📸 Loading images for inference #{inference_id}")
    images = get_random_images(MAX_IMAGES)
    
    if not images:
        error_msg = f'No images available in {IMAGE_DIR}'
        print(f"✗ {error_msg}")
        
        # 提供更详细的诊断信息
        if not os.path.exists(IMAGE_DIR):
            error_msg += f" (directory does not exist)"
        else:
            all_files = os.listdir(IMAGE_DIR)
            error_msg += f" (found {len(all_files)} files, but no valid images)"
        
        return jsonify({'success': False, 'error': error_msg}), 404
    
    print(f"✓ Returning {len(images)} images to client\n")
    
    return jsonify({
        'success': True,
        'inference_id': inference_id,
        'images': images,
        'total_images': len(images)
    })


@app.route('/api/run_inference', methods=['POST'])
def run_inference():
    """Run inference on selected images"""
    global last_inference_result
    
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    
    image_base64_list = data.get('images', [])
    inference_id = data.get('inference_id')
    
    if not inference_id or inference_id <= 0:
        return jsonify({'success': False, 'error': 'Invalid inference ID'}), 400
    
    if not image_base64_list:
        return jsonify({'success': False, 'error': 'No images provided'}), 400
    
    # Run inference
    result = send_request(INSTRUCTION, image_base64_list)
    
    # Log the result
    image_paths = [f"Image {i+1}" for i in range(len(image_base64_list))]
    write_log(inference_id, result, image_paths)
    
    # Store the last result (thread-safe)
    with inference_lock:
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
    with inference_lock:
        result = last_inference_result
    
    return jsonify({
        'success': True,
        'result': result
    })


@app.route('/api/config')
def get_config():
    """Get current configuration"""
    return jsonify({
        'success': True,
        'config': {
            'image_dir': IMAGE_DIR,
            'base_url': BASE_URL,
            'instruction': INSTRUCTION,
            'max_images': MAX_IMAGES,
            'max_tokens': MAX_TOKENS,
            'tokenizer_loaded': tokenizer is not None
        }
    })


def test_api_connection() -> bool:
    """Test if the API server is accessible"""
    try:
        print(f"🔍 Testing API connection to {BASE_URL}...")
        response = requests.get(f"{BASE_URL}/v1/models", timeout=5)
        response.raise_for_status()
        print(f"   ✓ API server is accessible")
        
        # Try to parse response
        try:
            models = response.json()
            print(f"   Available models: {models}")
        except:
            pass
        
        return True
    except requests.exceptions.ConnectionError:
        print(f"   ✗ Cannot connect to API server at {BASE_URL}")
        print(f"   Please ensure your VLM server is running")
        return False
    except requests.exceptions.Timeout:
        print(f"   ✗ Connection timeout")
        return False
    except Exception as e:
        print(f"   ⚠ Warning: {e}")
        return False


def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except socket.error:
            return True


def open_browser(port: int) -> None:
    """Open browser to the web interface"""
    time.sleep(1.5)
    webbrowser.open(f'http://localhost:{port}')


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Web-based VLM inference demo")
    parser.add_argument("-t", "--tokenizer-path", type=str, 
                       help="Path to local tokenizer directory")
    parser.add_argument("-p", "--port", type=int, default=WEB_PORT, 
                       help="Web server port")
    parser.add_argument("--no-browser", action="store_true",
                       help="Don't automatically open browser")
    args = parser.parse_args()

    # Load tokenizer
    print("=" * 50)
    print("VLM Inference Demo - Starting Up")
    print("=" * 50)
    print("Loading tokenizer...")
    tokenizer_loaded = load_tokenizer(args.tokenizer_path)

    print(f"\n📍 Configuration:")
    print(f"  API endpoint: {BASE_URL}")
    print(f"  Image directory: {IMAGE_DIR}")
    print(f"  Log file: {LOG_FILE}")
    print(f"  Web port: {args.port}")
    print(f"  Tokenizer: {'✓ Loaded' if tokenizer_loaded else '⚠ Approximate counting'}")
    print("=" * 50)

    # Test API connection
    api_ok = test_api_connection()
    if not api_ok:
        print(f"\n⚠ WARNING: API server may not be available!")
        print(f"   The web interface will start, but inference may fail.")
        print(f"   Press Ctrl+C to exit, or Enter to continue anyway...")
        try:
            input()
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            return

    # Check if image directory exists
    if not os.path.exists(IMAGE_DIR):
        print(f"\n⚠ Warning: Image directory '{IMAGE_DIR}' does not exist!")
        print(f"  Creating directory...")
        os.makedirs(IMAGE_DIR, exist_ok=True)

    # Check if port is available
    if is_port_in_use(args.port):
        print(f"\n✗ Error: Port {args.port} is already in use.")
        print(f"  Please use a different port with -p option.")
        return

    # Start browser in background
    if not args.no_browser:
        threading.Thread(target=open_browser, args=(args.port,), daemon=True).start()

    # Start Flask app
    print(f"\n🚀 Starting web server on http://localhost:{args.port}")
    print(f"   Press Ctrl+C to stop\n")
    
    try:
        app.run(host='localhost', port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")


if __name__ == "__main__":
    main()
