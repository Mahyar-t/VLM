# NiluLab Web App

The NiluLab web app provides a beautiful user interface for interacting with various Computer Vision models including Image Captioning, Visual Question Answering, and core Image Classification.

The backend uses a **Java Spring Boot** application that acts as a bridge, automatically starting and serving the **FastAPI** Python application in the background!

---

## 1. Prerequisites
- **Python** 3.9+ and **Java JDK** 17+
- Activated environment (e.g. `conda activate vlm` or `.venv/bin/activate`)
- Python packages installed (`pip install -e .`)
- **Maven** installed

---

## 2. Using Qwen 2.5-VL

The Image Captioning multimodal LLM feature requires the **Qwen2.5-VL-3B-Instruct** model. 
Since the backend uses the Hugging Face Hub directly, the application will automatically download and cache it (`~/.cache/huggingface/hub/`) the first time it is requested. 

If you prefer to download it manually before starting the app to avoid long loading times, you have two options:

**Option A: Hugging Face CLI (Recommended)**
1. Install the CLI:
   ```bash
   pip install -U "huggingface_hub[cli]"
   ```
2. Download the model to your local cache:
   ```bash
   huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct
   ```

**Option B: Manual Repository Visit**
You can also visit the official repository directly to view the model, read its documentation, or manually pull the files:
👉 [https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/main](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/main)

If you use this method, you must download all the files and place them inside a folder named `Qwen/Qwen2.5-VL-3B-Instruct` exactly in the **root of the repository**, next to the `webapp` directory. The application will automatically detect this folder and load from it instead of downloading.

Your folder structure should look like this:
```text
VLM/
├── webapp/
├── Qwen/
│   ├── Qwen2.5-VL-3B-Instruct/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── ... (other model files)
```

---

## 3. Run the Web Application

The Java application automatically starts the Python Uvicorn server in the background for you. To start it:

```bash
cd /home/mahyart/Desktop/github_repos/VLM/webapp

# run the spring boot app
kill -9 $(lsof -t -i:8000)
mvn spring-boot:run
```

---

## 4. Accessing the UI

Once the server says "Started Application", open your web browser and navigate to:

👉 **http://localhost:8080**

From the dashboard, you can access all features through the sidebar:
- **Image Captioning (with Multimodal LLM Tab)**
- **Visual Question Answering (VQA)**
- **Inference & Fine-Tuning**

---

## 5. Troubleshooting Models / VRAM

The app loads deep learning models directly into your GPU VRAM (`cuda`):
- If you face `Out of Memory` (OOM) errors, click the red **`Reset the cache models`** button in the web app UI to unconditionally clear the loaded models.
- Qwen2.5-VL-3B is loaded locally in 4-bit precision to save memory.
- You can monitor your active GPU connection via the "GPU READY" indicator on the captioning page.


