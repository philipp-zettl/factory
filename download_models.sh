#!/bin/bash
#
#
set -e

(

  cd models/

  echo "Downloading encoder model..."
  mkdir -p image_encoder
  wget -O image_encoder/config.json https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/config.json?download=true
  wget -O image_encoder/model.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors?download=true

  echo "Downloading ip-adapter models..."
  mkdir -p ip-adapter
  wget -O ip-adapter/ip-adapter_sd15.bin https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin?download=true
  wget -O ip-adapter/ip-adapter-plus_sd15.bin https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.bin?download=true
  wget -O ip-adapter/ip-adapter-faceid-portrait_sd15.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait_sd15.bin?download=true
  wget -O ip-adapter/ip-adapter-faceid-plusv2_sd15.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin?download=true


  echo "Downloading tts voice samples..."
  mkdir -p tts
  wget -O tts/cmu_us_awb_arctic-wav-arctic_a0002.npy https://huggingface.co/spaces/Matthijs/speecht5-tts-demo/resolve/main/spkemb/cmu_us_awb_arctic-wav-arctic_a0002.npy?download=true
  wget -O tts/cmu_us_bdl_arctic-wav-arctic_a0009.npy https://huggingface.co/spaces/Matthijs/speecht5-tts-demo/resolve/main/spkemb/cmu_us_bdl_arctic-wav-arctic_a0009.npy?download=true
  wget -O tts/cmu_us_clb_arctic-wav-arctic_a0144.npy https://huggingface.co/spaces/Matthijs/speecht5-tts-demo/resolve/main/spkemb/cmu_us_clb_arctic-wav-arctic_a0144.npy?download=true 
  wget -O tts/cmu_us_rms_arctic-wav-arctic_b0353.npy https://huggingface.co/spaces/Matthijs/speecht5-tts-demo/resolve/main/spkemb/cmu_us_rms_arctic-wav-arctic_b0353.npy?download=true
  wget -O tts/cmu_us_slt_arctic-wav-arctic_a0508.npy https://huggingface.co/spaces/Matthijs/speecht5-tts-demo/resolve/main/spkemb/cmu_us_slt_arctic-wav-arctic_a0508.npy?download=true

)
