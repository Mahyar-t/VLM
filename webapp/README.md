# imgcls-ft Web API (Java / Spring Boot)

A REST API built with **Spring Boot 3** that wraps the `imgcls-ft` Python image classification package. The Java server invokes the Python CLI commands (`imgcls-predict`, `imgcls-train`) via `ProcessBuilder`.

---

## Prerequisites

| Tool | Version |
|------|---------|
| **Java JDK** | 17+ |
| **Maven** | 3.8+ |
| **Python** | 3.9+ |

---

## 1 — Install the Python package

```bash
# from the project root (where pyproject.toml lives)
cd /home/mahyart/Desktop/github_repos/VLM

# create & activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# install in editable mode
pip install -e .
```

Verify the CLI is available:

```bash
imgcls-predict --help
imgcls-train --help
```

---

## 2 — Build & run the API

```bash
cd webapp

# build
mvn clean package -DskipTests

# run
mvn spring-boot:run
```

The server starts on **http://localhost:8080**. Open this URL in your browser to see the landing page with interactive documentation.

---

## 3 — API Endpoints

### Health check

```bash
curl http://localhost:8080/api/health
# {"status":"ok"}
```

### List supported models

```bash
curl http://localhost:8080/api/models
# {"models":["mobilenet_v3_small","mobilenet_v3_large","resnet18","resnet50","densenet121","efficientnet_b0"]}
```

### Predict (classify an image)

```bash
curl -X POST http://localhost:8080/api/predict \
  -F "image=@/path/to/photo.jpg" \
  -F "weights=/path/to/best.pt" \
  -F "class_map=/path/to/class_to_idx.json" \
  -F "model=mobilenet_v3_small" \
  -F "device=cpu" \
  -F "topk=5"
```

Response:

```json
{
  "status": "ok",
  "predictions": [
    {"class": "cat", "probability": 0.9234},
    {"class": "dog", "probability": 0.0412}
  ]
}
```

### Train (fine-tune a model)

```bash
curl -X POST http://localhost:8080/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "data_dir": "/path/to/dataset",
    "model": "mobilenet_v3_small",
    "epochs": 10,
    "batch_size": 32,
    "lr": 0.0001,
    "device": "cpu",
    "checkpoint": "best.pt",
    "save_class_map": "class_to_idx.json"
  }'
```

---

## 4 — Configuration

Edit `src/main/resources/application.properties`:

| Property | Default | Description |
|----------|---------|-------------|
| `server.port` | `8080` | HTTP port |
| `spring.servlet.multipart.max-file-size` | `50MB` | Max upload size |
| `python.executable` | `python3` | Path to your Python binary |

---

## Project structure

```
webapp/
├── pom.xml
├── README.md
└── src/main/
    ├── java/com/imgclsft/api/
    │   ├── Application.java          # Spring Boot entry point
    │   ├── ApiController.java        # REST endpoints
    │   └── PythonBridge.java         # Calls Python CLI via ProcessBuilder
    └── resources/
        ├── application.properties
        └── static/
            └── index.html            # Landing page with predict form
```
