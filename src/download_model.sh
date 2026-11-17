#!/bin/bash
mkdir -p models
if [ ! -f "models/best_efficientnet.pth" ]; then
    echo "Downloading model..."
    wget https://github.com/radhikagarg004/medical-analysis-final/releases/download/v1.0-model/best_efficientnet.pth -O models/best_efficientnet.pth
fi