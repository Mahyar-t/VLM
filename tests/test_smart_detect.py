import base64
import requests
import json
import time
import os

def test_smart_detect(image_path, query):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
    payload = {
        "image_base64": img_b64,
        "user_query": query,
        "threshold": 0.3
    }
    
    start_time = time.time()
    print(f"Sending request for query: '{query}'...")
    response = requests.post("http://127.0.0.1:8001/api/smart-detect", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print("\nSUCCESS!")
        print("Queries used:", data.get("queries_used"))
        detections = data.get("detections", [])
        print(f"Found {len(detections)} detections.")
        for i, det in enumerate(detections):
            print(f"[{i}] Label: {det['label']} | Score: {det['score']:.2f} | Bbox: {det['box']}")
    else:
        print("\nFAILED!")
        print("Status code:", response.status_code)
        print("Error:", response.text)
        
    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    img_path = "/home/mahyart/Desktop/github_repos/VLM/test.jpg"
    test_smart_detect(img_path, "find all interesting objects")
