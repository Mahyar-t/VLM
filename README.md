# NiluLab 🪷

NiluLab is a Computer Vision platform that provides a practical web interface and a Python API for interacting with state-of-the-art models. Features include **Image Captioning (with Multimodal LLMs)**, **Visual Question Answering (VQA)**, **Smart Object Detection**, and **Image Classification**.

---

## 🐳 Quick Start with Docker (Recommended)

The easiest way to get NiluLab up and running is using Docker. This handles all dependencies (Java, Python, CUDA) in a single container.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
- For GPU support (NVIDIA): [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Mahyar-t/VLM.git
   cd VLM
   ```
2. (Optional but Recommended) Download the Qwen model manually to [root]/Qwen/Qwen2.5-VL-3B-Instruct to avoid slow downloads inside the container.
3. Start the application:
   ```bash
   docker compose up -d
   ```
4. Access the UI at: 👉 **http://localhost:8080**

---

## Manual Installation

### 1. Prerequisites

You will need the following installed on your system:

- **Python 3.9+**
- **Java JDK 17+**
- **Maven** (for building the Java backend)
- **Git**

### 2. Linux / WSL2 Setup

```bash
# Clone the repository
git clone https://github.com/Mahyar-t/VLM.git
cd VLM

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -e .
pip install ultralytics qwen-vl-utils bitsandbytes accelerate

# Build and run the Web App
cd webapp
mvn clean package -DskipTests
java -jar target/visionbox-api-0.1.0.jar
```

### 3. Windows Native Setup

1. **Python**: Install from [python.org](https://www.python.org/). Ensure "Add Python to PATH" is checked.
2. **Java**: Install [OpenJDK 17](https://adoptium.net/temurin/releases/?version=17).
3. **Setup**:

   ```powershell
   # Clone and enter repo
   git clone https://github.com/Mahyar-t/VLM.git
   cd VLM

   # Create environment
   python -m venv .venv
   .venv\Scripts\activate

   # Install requirements
   pip install -e .
   pip install ultralytics qwen-vl-utils bitsandbytes accelerate

   # Build and run
   cd webapp
   mvn clean package -DskipTests
   java -jar target/visionbox-api-0.1.0.jar
   ```

---

## Model Configuration: Qwen 2.5-VL

The Image Captioning feature uses **Qwen2.5-VL-3B-Instruct**. The app will automatically download it if not found, but we recommend manual placement for a smoother experience.

1. Create a folder named `Qwen` in the root of the repo.
2. Download all files from [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/main) and place them in `Qwen/Qwen2.5-VL-3B-Instruct`.

Your structure should look like this:

```text
VLM/
├── Qwen/
│   └── Qwen2.5-VL-3B-Instruct/
│       ├── config.json
│       ├── model.safetensors
│       └── ...
├── webapp/
└── ...
```

---

## CLI Tools

NiluLab also provides CLI tools for specific tasks:

- **Captioning**: `visionbox-caption --image path/to/img.jpg`
- **VQA**: `visionbox-vqa --image path/to/img.jpg --question "What is in the image?"`
- **Training**: `visionbox-train --data-dir dataset/`

---

## Troubleshooting & VRAM

- **Out of Memory (OOM)**: Large models stay cached in VRAM for speed. If you run out of memory, click the red **`Reset the cache models`** button in the UI.
- **VRAM Monitor**: Check the real-time VRAM usage pill at the top of the Image Captioning page.
- **Java Port**: The app uses port `8080` (Web) and `8000` (Python). Ensure these are free.

---

## 🏛️ Design Heritage

The logo and name are inspired by the ancient lotus flower (Niloofar in Farsi) one of the most sacred and ubiquitous symbols of the Achaemenid and Sassanid Empires and a symbol of peace and love. This motif can be seen in Persepolis and other historical sites in Iran, with roots dating back roughly 2,500 years.
