# Factory
Project/Service for generative AI

Provides API endpoints for general purpose GAI.

## Features
- Text2Image:
  - [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- Image2Image:
  - [IP-Adapter](https://huggingface.co/stabilityai/image-prompt-adapter)

## Usage
Before we get started, make sure you have the following installed:
- `poetry`
- `python>=3.10`

To install the dependencies, run:
```bash
poetry install
```
Apart from the python dependencies, you'll need to download the models from the IP-Adapter repositories.
You can do this by running:
```bash
poetry run bash download_models.sh
```

## API
- `/models/`: List of available models.
- `/gen/`: Text2Image API for generating images based on SDXL, sends a prompt to generate a new image.
- `/ip-faces/`: Image2Image API for portraits based on IP-Adapter, sends a single image and a prompt to generate a new image.
- `/ip-faces-multi/`: Image2Image API for portraits based on IP-Adapter, sends multiple images and a prompt to generate a new portrait.
- `/ip-person/`: Image2Image API for images of people based on IP-Adapter, sends a single image and a prompt to generate a new image.

**Note**: The API is not public, you need to run the service locally. Apart from this, to send images to the API, you'll need to encode the images in base64.
The API will return a JSON response with the generated image encoded in base64, too.

## Examples:
Using the `requests` library in Python:
```python
import requests
import base64
images = [
  # YOUR IMAGES
]
encoded_imgs = []
for img in images:
    with open(img, "rb") as image_file:
        encoded_imgs.append(base64.b64encode(image_file.read()).decode('ascii'))


prompt = "YOUR PROMPT"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
res = requests.post('http://localhost:8889/ip-faces-multi/', json={
    'task': {
        'images': encoded_imgs,
        'prompt': prompt,
        'options': {
            'num_inference_steps': 30,
            'negative_prompt': negative_prompt,
            #'guidance_scale': 7.5,
            'seed': 420,
            #'negative_prompt': 'monochrome, lowres, bad anatomy, worst quality, low quality, blurry',
            #'s_scale': 0.05
        }
    }
})
```

## Acknowledgements
Special thanks to the developers and authors of the models used in this project.
This project could not have been possible without the following:
- [IP-Adapter](https://huggingface.co/h94/IP-Adapter/tree/main) for several image2image models
- [huggingface 🤗](https://huggingface.co/) for diffusers and the model-hub
- [pytorch](https://pytorch.org/) for the amazing deep learning framework
