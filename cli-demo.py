import cv2
import base64
import requests
import time
import threading

# Configuration
BASE_URL = "http://localhost:8080"  # Your API endpoint
INSTRUCTION = "What do you see?"     # Default instruction
INTERVAL = 0.5                       # Request interval (seconds)
STOP_FLAG = False

def capture_frame():
    """Capture camera frame and return Base64 encoded image"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera")
        return None

    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame")
        return None

    _, buffer = cv2.imencode('.jpg', frame)
    img_str = base64.b64encode(buffer).decode('utf-8')
    cap.release()
    return f"data:image/jpeg;base64,{img_str}"

def send_request(instruction, image_base64):
    """Send request and return response"""
    payload = {
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": image_base64}}
                ]
            }
        ]
    }

    try:
        response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
        if response.status_code == 200:
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "No response")
        else:
            return f"Server error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Request failed: {str(e)}"

def process_loop():
    global STOP_FLAG
    while not STOP_FLAG:
        image = capture_frame()
        if image:
            result = send_request(INSTRUCTION, image)
            print(f"AI Response: {result}")
        else:
            print("Cannot capture image")
        time.sleep(INTERVAL)

def main():
    print("Camera started, press Ctrl+C to stop")
    thread = threading.Thread(target=process_loop)
    thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        global STOP_FLAG
        STOP_FLAG = True
        thread.join()
        print("Program stopped")

if __name__ == "__main__":
    main()
