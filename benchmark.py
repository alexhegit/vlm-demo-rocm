import os
import random
import base64
import requests
import time
import argparse
import json
import re
from datetime import datetime

# Import tokenizer libraries
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Configuration
IMAGE_DIR = "./images"  # Directory path of images
BASE_URL = "http://localhost:8080"  # Your API endpoint
INSTRUCTION = "Describe the images in 80 words within one sentence"  # Prompt for inference
LOG_FILE = "inference-result.json"  # Log file name

# Global tokenizer variable
tokenizer = None

def load_tokenizer(tokenizer_path=None):
    """
    Load tokenizer from local directory
    
    Args:
        tokenizer_path: Path to local tokenizer directory
    """
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
    """
    Approximate token counting method
    
    Note: This is still word-based counting, not true tokenization.
    For accurate token counting, you would need:
    1. tiktoken library: for OpenAI-compatible models
    2. transformers library: model.tokenizer.encode(text)
    3. API response: many APIs return actual token counts
    
    Current method: Enhanced word counting as proxy
    """
    if not text:
        return 0
    
    # Enhanced word-based approximation
    # This counts words + punctuation as separate units
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text.strip())
    
    # Apply rough correction factor for subword tokenization
    # Most modern tokenizers split words into subwords
    word_count = len([t for t in tokens if re.match(r'\w+', t)])
    punct_count = len(tokens) - word_count
    
    # Rough estimation: average 1.0 tokens per word for subword tokenization
    estimated_tokens = int(word_count * 1.0) + punct_count
    
    return estimated_tokens

def get_random_images(num_images):
    """Select a random set of .jpg images and return base64 data list"""
    if not os.path.exists(IMAGE_DIR):
        print(f"Directory {IMAGE_DIR} does not exist")
        return []

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.jpg')]
    if not image_files:
        print(f"No .jpg files found in directory {IMAGE_DIR}")
        return []

    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    image_paths = [os.path.join(IMAGE_DIR, img) for img in selected_images]
    base64_images = []

    for img_path in image_paths:
        try:
            with open(img_path, "rb") as img_file:
                img_bytes = img_file.read()
            img_str = base64.b64encode(img_bytes).decode('utf-8')
            base64_images.append(f"data:image/jpeg;base64,{img_str}")
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            continue

    print(f"Using {len(base64_images)} images")
    return base64_images

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
            response.raise_for_status()  # Raise exception for HTTP errors
            
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
                print(f"  → Using API token count: {completion_tokens} completion tokens, {prompt_tokens} prompt tokens")
            else:
                tokens = count_tokens_with_tokenizer(full_response)
                token_source = "tokenizer" if tokenizer is not None else "estimated"
                print(f"  → Using {token_source} token count: {tokens} (API didn't provide usage info)")
            
            return first_token_time, latency, tokens, full_response.strip(), {
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens
            }
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
        return None, None, 0, "", {}
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None, None, 0, "", {}

def write_log(inference_no, ttft, latency, tokens, content, token_usage=None):
    """Write the inference result to the JSON file"""
    result = {
        "inference_no": inference_no,
        "ttft": round(ttft, 4) if ttft is not None else None,
        "latency": round(latency, 4) if latency is not None else None,
        "tokens": tokens,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add detailed token usage if available
    if token_usage and any(token_usage.values()):
        result["token_usage"] = token_usage
        result["token_source"] = "api" if token_usage.get("completion_tokens", 0) > 0 else "tokenizer"
    else:
        result["token_source"] = "tokenizer" if tokenizer is not None else "estimated"

    # If the file does not exist, create and write the first line
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([result], f, ensure_ascii=False, indent=2)
    else:
        # Read existing data, append new result, and write back
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Remove stats if they exist (they'll be added at the end)
            if data and isinstance(data[-1], dict) and "total_latency" in data[-1]:
                data.pop()
            
            data.append(result)
            
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, FileNotFoundError):
            # If file is corrupted or doesn't exist, create new
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                json.dump([result], f, ensure_ascii=False, indent=2)

    # Print to terminal
    ttft_str = f"{ttft:.4f}s" if ttft is not None else "N/A"
    latency_str = f"{latency:.4f}s" if latency is not None else "N/A"
    token_source = result.get("token_source", "estimated")
    
    # Calculate both token counts for comparison display
    tokenizer_tokens = count_tokens_with_tokenizer(content) if tokenizer is not None else 0
    approx_tokens = count_tokens_approximately(content)
    
    # Show token counts based on tokenizer availability
    if tokenizer is not None:
        print(f"[Inference {inference_no}] TTFT: {ttft_str} | Latency: {latency_str} | Tokens: {tokens} ({token_source}) | Approx: {approx_tokens} | Content: {content}")
    else:
        print(f"[Inference {inference_no}] TTFT: {ttft_str} | Latency: {latency_str} | Tokens: {tokens} ({token_source}) | Content: {content}")

def finalize_log(stats):
    """Add final statistics to the log file"""
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Remove existing stats if present
        if data and isinstance(data[-1], dict) and "total_latency" in data[-1]:
            data.pop()
        
        # Add new stats
        data.append(stats)
        
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error finalizing log: {e}")

def main():
    parser = argparse.ArgumentParser(description="Image inference test script")
    parser.add_argument("-n", type=int, default=1, help="Number of inferences (default: 1)")
    parser.add_argument("-i", "--imgs", type=int, default=9, help="Number of images per inference (default: 9)")
    parser.add_argument("-t", "--tokenizer-path", type=str, help="Path to local tokenizer directory")
    args = parser.parse_args()

    n = args.n
    img_num = args.imgs

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer_loaded = load_tokenizer(args.tokenizer_path)

    print(f"Starting {n} inference(s) with {img_num} images each")
    print(f"API endpoint: {BASE_URL}")
    print(f"Log file: {LOG_FILE}")
    print(f"Image directory: {IMAGE_DIR}")
    if tokenizer_loaded:
        print(f"Tokenizer: Loaded from {args.tokenizer_path}")
    else:
        print("Tokenizer: Using approximate counting")
    print("=" * 50)

    successful_inferences = 0
    total_latency = 0
    total_tokens = 0
    ttfts = []
    latencies = []
    inference_times = []  # For pure inference time (excluding network overhead)

    for i in range(1, n + 1):
        print(f"\n=== Starting Inference {i}/{n} ===")
        image_base64_list = get_random_images(img_num)
        if not image_base64_list:
            print(f"Inference {i} failed: No images available")
            write_log(i, None, None, 0, "No images available")
            continue

        ttft, latency, tokens, content, token_usage = send_request(INSTRUCTION, image_base64_list)
        if ttft is None or latency is None:
            print(f"Inference {i} failed")
            write_log(i, None, None, 0, "Request failed")
            continue

        successful_inferences += 1
        total_latency += latency
        total_tokens += tokens
        ttfts.append(ttft)
        latencies.append(latency)
        
        # Calculate pure inference time (latency - ttft gives generation time)
        if ttft is not None:
            inference_time = latency - ttft
            inference_times.append(inference_time)

        write_log(i, ttft, latency, tokens, content, token_usage)

    # Calculate statistics
    if successful_inferences > 0:
        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        
        # Calculate tokens per second using both counting methods
        # Use current test results only (not historical data)
        total_tokens_tokenizer = 0
        total_tokens_approx = 0
        
        # Calculate total tokens using both methods for current test only
        for i in range(1, successful_inferences + 1):
            # Find the current test inference result
            try:
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    log_data = json.load(f)
                # Look for the most recent entry with this inference number
                for entry in reversed(log_data):
                    if isinstance(entry, dict) and entry.get("inference_no") == i and "content" in entry:
                        content = entry["content"]
                        if tokenizer is not None:
                            total_tokens_tokenizer += count_tokens_with_tokenizer(content)
                        total_tokens_approx += count_tokens_approximately(content)
                        break
            except:
                pass
        
        # Calculate tokens per second for both methods
        if sum(inference_times) > 0:
            tokens_per_second_inference_tokenizer = total_tokens_tokenizer / sum(inference_times)
            tokens_per_second_inference_approx = total_tokens_approx / sum(inference_times)
        else:
            tokens_per_second_inference_tokenizer = 0
            tokens_per_second_inference_approx = 0
            
        if total_latency > 0:
            tokens_per_second_total_tokenizer = total_tokens_tokenizer / total_latency
            tokens_per_second_total_approx = total_tokens_approx / total_latency
        else:
            tokens_per_second_total_tokenizer = 0
            tokens_per_second_total_approx = 0
        
        # Keep original calculations for compatibility
        tokens_per_second_inference = total_tokens / sum(inference_times) if sum(inference_times) > 0 else 0
        tokens_per_second_total = total_tokens / total_latency if total_latency > 0 else 0
        
        # Calculate min/max for better analysis
        min_ttft = min(ttfts) if ttfts else 0
        max_ttft = max(ttfts) if ttfts else 0
        min_latency = min(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0

        stats = {
            "test_summary": {
                "total_inferences": n,
                "successful_inferences": successful_inferences,
                "success_rate": successful_inferences / n * 100
            },
            "timing_stats": {
                "total_latency": round(total_latency, 4),
                "avg_ttft": round(avg_ttft, 4),
                "min_ttft": round(min_ttft, 4),
                "max_ttft": round(max_ttft, 4),
                "avg_latency": round(avg_latency, 4),
                "min_latency": round(min_latency, 4),
                "max_latency": round(max_latency, 4),
                "avg_inference_time": round(avg_inference_time, 4)
            },
            "token_stats": {
                "total_tokens": total_tokens,
                "avg_tokens_per_inference": round(total_tokens / successful_inferences, 2) if successful_inferences > 0 else 0,
                "tokens_per_second_total": round(tokens_per_second_total, 2),
                "tokens_per_second_inference": round(tokens_per_second_inference, 2),
                # Add tokenizer-based stats
                "total_tokens_tokenizer": total_tokens_tokenizer,
                "total_tokens_approx": total_tokens_approx,
                "tokens_per_second_total_tokenizer": round(tokens_per_second_total_tokenizer, 2),
                "tokens_per_second_total_approx": round(tokens_per_second_total_approx, 2),
                "tokens_per_second_inference_tokenizer": round(tokens_per_second_inference_tokenizer, 2),
                "tokens_per_second_inference_approx": round(tokens_per_second_inference_approx, 2)
            },
            "test_config": {
                "images_per_inference": img_num,
                "instruction": INSTRUCTION,
                "timestamp": datetime.now().isoformat()
            }
        }

        finalize_log(stats)

        print("\n" + "=" * 50)
        print("=== TEST COMPLETED ===")
        print(f"Successful inferences: {successful_inferences}/{n} ({successful_inferences/n*100:.1f}%)")
        print(f"Total latency: {total_latency:.4f}s")
        print(f"Average TTFT: {avg_ttft:.4f}s (min: {min_ttft:.4f}s, max: {max_ttft:.4f}s)")
        print(f"Average E2EL: {avg_latency:.4f}s (min: {min_latency:.4f}s, max: {max_latency:.4f}s)")
        print(f"Average decoding time: {avg_inference_time:.4f}s")
        print(f"Total tokens: {total_tokens}")
        print(f"Average tokens per inference: {total_tokens/successful_inferences:.2f}" if successful_inferences > 0 else "N/A")
        
        # Show token counting comparison
        print("\n--- Decoding Statics ---")
        print(f"Total tokens (tokenizer): {total_tokens_tokenizer}")
        print(f"Total tokens (approximate): {total_tokens_approx}")
        print(f"TPS (decoding only) - tokenizer: {tokens_per_second_inference_tokenizer:.2f}")
        print(f"TPS (decoding only) - approximate: {tokens_per_second_inference_approx:.2f}")
        print("=" * 50)
    else:
        print("\n=== TEST FAILED ===")
        print("No successful inferences completed")
        stats = {
            "test_summary": {
                "total_inferences": n,
                "successful_inferences": 0,
                "success_rate": 0
            },
            "error": "No successful inferences completed",
            "timestamp": datetime.now().isoformat()
        }
        finalize_log(stats)

if __name__ == "__main__":
    main()
