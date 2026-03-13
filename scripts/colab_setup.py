import os
import sys
import subprocess
import argparse

def run_cmd(cmd, cwd=None):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=cwd)

def setup_colab(token, port, persistent=False):
    # 1. Install pyngrok
    try:
        from pyngrok import ngrok
    except ImportError:
        run_cmd("pip install pyngrok --quiet")
        from pyngrok import ngrok
    
    if token:
        ngrok.set_auth_token(token)
        print("✅ ngrok token set.")
    else:
        print("⚠️ No ngrok token provided. You might be limited.")

    # 2. System Dependencies
    print("📦 Checking system dependencies...")
    # Quick check for java and mvn
    if subprocess.call("command -v java >/dev/null 2>&1", shell=True) != 0 or \
       subprocess.call("command -v mvn >/dev/null 2>&1", shell=True) != 0:
        run_cmd("apt-get update -y && apt-get install -y openjdk-17-jdk-headless maven libgl1-mesa-glx python3-pip git")
    else:
        print("✅ System dependencies already installed.")

    # 3. Python Dependencies
    print("🐍 Checking Python dependencies...")
    # Check if we need to install the package
    try:
        import visionbox
        print("✅ Project package already installed.")
    except ImportError:
        run_cmd("pip install -e . --quiet")

    # 4. Build Java Frontend
    print("☕ Checking Java frontend build...")
    webapp_dir = os.path.join(os.getcwd(), "webapp")
    jar_path = os.path.join(webapp_dir, "target", "visionbox-api-0.1.0.jar")
    
    if not os.path.exists(jar_path):
        print("Building Maven project...")
        run_cmd("mvn clean package -DskipTests", cwd=webapp_dir)
    else:
        print("✅ Java build found, skipping Maven.")

    # 5. Start Tunnel
    print(f"🌐 Starting ngrok tunnel on port {port}...")
    try:
        public_url = ngrok.connect(port).public_url
        print("\n" + "="*50)
        print(f"🚀 YOUR WEBSITE IS LIVE AT: {public_url}")
        print("="*50 + "\n")
    except Exception as e:
        print(f"❌ Failed to start ngrok: {e}")
        # Sometimes a tunnel is already open
        tunnels = ngrok.get_tunnels()
        if tunnels:
            print(f"Existing tunnel found: {tunnels[0].public_url}")
        else:
            raise

    # 6. Start Application
    print("🎬 Starting NiluLab...")
    os.environ["PYTHON_EXECUTABLE"] = "python3"
    run_cmd(f"java -jar {jar_path} --server.port={port}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NiluLab Colab Setup")
    parser.add_argument("--token", help="ngrok authtoken")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the webapp on (default: 8080)")
    parser.add_argument("--persistent", action="store_true", help="Enable persistence mode (optimized for GDrive)")
    args = parser.parse_args()
    
    setup_colab(args.token, args.port, args.persistent)
