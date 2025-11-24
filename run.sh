#!/bin/bash
set -e

echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# ensure folders exist
mkdir -p uploaded_files
mkdir -p models

PORT=${PORT:-8000}
echo "Starting server on port $PORT"

exec uvicorn api:app --host 0.0.0.0 --port $PORT
