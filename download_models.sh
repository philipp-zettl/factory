#!/bin/bash
#
#
set -e

echo "Downloading encoder model..."
mkdir -p image_encoder
wget -O image_encoder/config.json https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/config.json?download=true
wget -O image_encoder/model.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors?download=true

echo "Downloading ip-adapter models..."
wget -O ip-adapter_sd15.bin https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin?download=true
wget -O ip-adapter-faceid-portrait_sd15.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait_sd15.bin?download=true
wget -O ip-adapter-faceid-plusv2_sd15.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin?download=true
