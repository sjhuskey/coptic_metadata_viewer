#!/bin/bash

# Change to the directory where this script is
cd "$(dirname "$0")"

# Check if Ollama is reachable from the host (not the container)
if ! curl -s http://localhost:11434 > /dev/null; then
    echo "❌ Ollama is not running on the host. Please start it before running this app."
    exit 1
fi

# Build the image
docker build -t coptic-viewer .

# Run the container
docker run --rm -it -p 8501:8501 \
    coptic-viewer
