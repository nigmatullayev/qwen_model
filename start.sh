#!/bin/bash

echo "Starting Qwen-Image RunPod Serverless Handler..."

# Environment variables tekshirish
if [ -z "$RUNPOD_API_KEY" ]; then
    echo "Warning: RUNPOD_API_KEY not set"
fi

# Python handler ni ishga tushirish
python3 -u /app/handler.py