import os
import sys
import subprocess
import argparse

def run_cmd(cmd, cwd=None):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=cwd)

def setup_colab(token, port):
    # 1. Install pyngrok
    run_cmd("pip install pyngrok --quiet")
    from pyngrok import ngrok
    
    if token:
        ngrok.set_auth_token(token)
        print("✅ ngrok token set.")
    else:
        print("⚠️ No ngrok token provided. You might be limited.")

    # 2. System Dependencies
    print("📦 Installing system dependencies...")
    run_cmd("apt-get update -y && apt-get install -y openjdk-17-jdk-headless maven libgl1-mesa-glx python3-pip git")

    # 3. Python Dependencies
    print("🐍 Installing Python dependencies...")
    run_cmd("pip install -e . --quiet")

    # 4. Build Java Frontend
    print("☕ Building Java frontend (Maven)...")
    webapp_dir = os.path.join(os.getcwd(), "webapp")
    run_cmd("mvn clean package -DskipTests", cwd=webapp_dir)

    # 5. Start Tunnel
    print(f"🌐 Starting ngrok tunnel on port {port}...")
    public_url = ngrok.connect(port).public_url
    print("\n" + "="*50)
    print(f"🚀 YOUR WEBSITE IS LIVE AT: {public_url}")
    print("="*50 + "\n")

    # 6. Start Application
    print("🎬 Starting NiluLab...")
    os.environ["PYTHON_EXECUTABLE"] = "python3"
    jar_path = os.path.join(webapp_dir, "target", "visionbox-api-0.1.0.jar")
    run_cmd(f"java -jar {jar_path} --server.port={port}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NiluLab Colab Setup")
    parser.add_argument("--token", help="ngrok authtoken")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the webapp on (default: 8080)")
    args = parser.parse_args()
    
    setup_colab(args.token, args.port)
