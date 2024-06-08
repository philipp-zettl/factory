#!/usr/bin/env python3

import argparse
import os
import requests
import json
import base64

from huggingface_hub import InferenceClient


def parse_args():
    parser = argparse.ArgumentParser(description='Generate request for the API')
    parser.add_argument('--input', type=str, required=False, help='Path to the input image')
    parser.add_argument('--output', type=str, required=True, help='Path to the output image')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt for the model')
    parser.add_argument('--options', type=str, default='{}', help='Options for the request')
    parser.add_argument('--parameters', type=str, default='{}', help='Parameters for the model')
    parser.add_argument('--model', type=str, default='tiny_diffusion', help='Model name')
    return parser.parse_args()


def main():
    args = parse_args()

    parameters = {
        "negative_prompt": "cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck", 
        "num_inference_steps": 35,
        "guidance_scale": 7,
        "width": 512,
        "height": 512
    }
    options = {}

    url = f'http://localhost:7777/models/{args.model}/image-to-image/'
    client = InferenceClient(url)
    with open(args.input or './assets/gabbie-carter-2.jpg', 'rb') as f:
        img = f.read()

    parameters['images'] = [base64.b64encode(img).decode('ascii')]
    response = client.image_to_image(img, args.prompt, **parameters, **options)
    response.save(args.output)

if __name__ == '__main__':
    main()

