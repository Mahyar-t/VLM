# VisionBox Web App

The VisionBox web app provides a beautiful user interface for interacting with various Computer Vision models including Image Captioning, Visual Question Answering, and core Image Classification.

The backend uses a **Java Spring Boot** application that acts as a bridge, automatically starting and serving the **FastAPI** Python application in the background!

---

## 1. Prerequisites
- **Python** 3.9+ and **Java JDK** 17+
- Activated environment (e.g. `conda activate vlm` or `.venv/bin/activate`)
- Python packages installed (`pip install -e .`)
- **Maven** installed

---

## 2. Run the Web Application

The Java application automatically starts the Python Uvicorn server in the background for you. To start it:

```bash
cd /home/mahyart/Desktop/github_repos/VLM/webapp

# run the spring boot app
mvn spring-boot:run
```

---

## 3. Accessing the UI

Once the server says "Started Application", open your web browser and navigate to:

👉 **http://localhost:8080**

From the dashboard, you can access all features through the sidebar:
- **Image Captioning (with Multimodal LLM Tab)**
- **Visual Question Answering (VQA)**
- **Inference & Fine-Tuning**

---

## 4. Troubleshooting Models / VRAM

The app loads deep learning models directly into your GPU VRAM (`cuda`):
- If you face `Out of Memory` (OOM) errors, click the red **`Reset the cache models`** button in the web app UI to unconditionally clear the loaded models.
- Qwen2.5-VL-3B is loaded locally in 4-bit precision to save memory.
- You can monitor your active GPU connection via the "GPU READY" indicator on the captioning page.


