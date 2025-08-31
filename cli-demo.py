import cv2
import base64
import requests
import time
import threading

# 配置
BASE_URL = "http://localhost:8080"  # 你的 API 地址
INSTRUCTION = "What do you see?"     # 默认指令
INTERVAL = 0.5                       # 请求间隔（秒）
STOP_FLAG = False

def capture_frame():
    """捕获摄像头画面并返回 Base64 编码图像"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法访问摄像头")
        return None

    ret, frame = cap.read()
    if not ret:
        print("无法读取帧")
        return None

    _, buffer = cv2.imencode('.jpg', frame)
    img_str = base64.b64encode(buffer).decode('utf-8')
    cap.release()
    return f"data:image/jpeg;base64,{img_str}"

def send_request(instruction, image_base64):
    """发送请求并返回响应"""
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
            return data.get("choices", [{}])[0].get("message", {}).get("content", "无响应")
        else:
            return f"服务器错误: {response.status_code} - {response.text}"
    except Exception as e:
        return f"请求失败: {str(e)}"

def process_loop():
    global STOP_FLAG
    while not STOP_FLAG:
        image = capture_frame()
        if image:
            result = send_request(INSTRUCTION, image)
            print(f"AI 响应: {result}")
        else:
            print("无法捕获图像")
        time.sleep(INTERVAL)

def main():
    print("摄像头已启动，按 Ctrl+C 停止")
    thread = threading.Thread(target=process_loop)
    thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        global STOP_FLAG
        STOP_FLAG = True
        thread.join()
        print("程序已停止")

if __name__ == "__main__":
    main()

