#!/bin/bash
#
#
set -e

(

  cd models/


  echo "Downloading phi-3 t2t models..."
  mkdir -p phi-3-mini-128k-chat

  file_names=(
    "added_tokens.json"
    "config.json"
    "configuration_phi3.py"
    "genai_config.json"
    "phi3-mini-128k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx"
    "phi3-mini-128k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx.data"
    "special_tokens_map.json"
    "tokenizer.json"
    "tokenizer.model"
    "tokenizer_config.json"
  )

  for file_name in "${file_names[@]}"
  do
    # check if file exists
    if [ -f "phi-3-mini-128k-chat/$file_name" ]; then
      echo "phi-3-mini-128k-chat/$file_name exists, skipping..."
      continue
    fi
    wget -O phi-3-mini-128k-chat/$file_name https://huggingface.co/microsoft/Phi-3-mini-128k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/$file_name
  done
)
