# Use NVIDIA CUDA base for GPU support
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# - Python 3.9, pip, venv
# - OpenJDK 17
# - Maven for building the Java app
# - libgl1-mesa-glx for OpenCV/YOLO
# - lsof for Java's process management
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    openjdk-17-jdk-headless \
    maven \
    libgl1-mesa-glx \
    libglib2.0-0 \
    lsof \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set python3.9 as default python3 and python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --set python3 /usr/bin/python3.9 \
    && ln -s /usr/bin/python3 /usr/bin/python

# Set the working directory
WORKDIR /app

# Copy python requirements first for caching
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Install the visionbox package in editable mode (or standard mode)
RUN pip3 install -e .

# Build the Java Spring Boot application
WORKDIR /app/webapp
RUN mvn clean package -DskipTests

# Expose ports
# 8080: Java Web App
# 8000: Python FastAPI (internal use but can be exposed)
EXPOSE 8080 8000

# Set the environment variable for Python executable needed by Java app
ENV PYTHON_EXECUTABLE=python3

# Start the application
# We run the JAR directly. The Java code will start the Python server.
CMD ["java", "-jar", "target/visionbox-api-0.1.0.jar"]
