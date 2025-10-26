#!/bin/bash
set -e

echo "Starting Voice AI deployment..."

if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Installing..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "Docker installed. Please log out and log back in, then run this script again."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Installing..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "Docker Compose installed"
fi

# Check for NVIDIA GPU support
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU support not detected. Installing nvidia-docker2..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker
    echo "NVIDIA Docker support installed"
fi

# Check for .env file
if [ ! -f .env ]; then
    echo "Creating .env file template..."
    cat > .env << EOF
HF_ACCESS_TOKEN=your_huggingface_token_here
ASR_MOCK_MODE=false
TTS_MOCK_MODE=false
LLM_ECHO_MODE=false
OLLAMA_MODEL=llama3.2:1b
EOF
    echo "Please edit .env file with your Hugging Face token"
    echo "Press Enter to continue or Ctrl+C to exit..."
    read
fi

# Build and start services
echo "Building containers..."
docker-compose build

echo "Starting services..."
docker-compose up -d

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
sleep 15

# Pull LLM model
echo "Pulling LLM model (this may take a few minutes)..."
docker-compose exec ollama ollama pull llama3.2:1b || echo "Model pull failed, will try again later"

echo ""
echo "Deployment complete!"
echo ""
echo "Services:"
echo "  - FastAPI: http://localhost:8000"
echo "  - Ollama: http://localhost:11434"
echo ""
echo "Health check:"
echo "  curl http://localhost:8000/health"
echo ""
echo "View logs:"
echo "  docker-compose logs -f fastapi"
echo ""
echo "To access from outside, open port 8000 in your VM firewall"
