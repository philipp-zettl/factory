# Factory

> [!IMPORTANT]  
> This service is in a early phase of development and it's API as well as the supported models are changing frequently.
> Use the current state as a reference for future development or as a starting point for your own project.

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
Run the service and see the API documentation at `http://localhost:8888/docs`.

## Examples:

> [!IMPORTANT] Please note that albeit the API provides an interface to generate multiple images at once,
> when using the HF InferenceClient, only one image can be generated at a time.
> This is due to limitations in the HF InferenceClient implementation.


### HuggingFace Inferece API Client
Using the [huggingface.co](https://huggingface.co/) Inference API Client:

### Text2Image
```python
from huggingface_hub import InferenceClient

client = InferenceClient("http://localhost:8888/models/tiny_diffusion/text-to-image")
response = client.text_to_image("A cat in a hat")
response.save('cat_in_hat.png')
```

### Image2Image
#### IP-Adapter (Image Prompt Adapter) - Single Image
```python
from huggingface_hub import InferenceClient

client = InferenceClient("http://localhost:8888/models/ip/image-to-image")
with open("cat.jpg", "rb") as image_file:
    response = client.image_to_image(image_file.read(), "A cat in a hat")
    response.save('cat_in_hat.png')
```
#### IP-Adapter Portrait (single reference image)
```python
from huggingface_hub import InferenceClient

client = InferenceClient("http://localhost:8888/models/ip-faces/image-to-image")

with open('portrait.jpg', 'rb') as image_file:
    response = client.image_to_image(image_file.read(), "A portrait of a young man")
    response.save('portrait_you.jpg')
```

#### IP-Adapter (Image Prompt Adapter) - Multiple Images
```python
from huggingface_hub import InferenceClient

client = InferenceClient("http://localhost:8888/models/ip-faces-portrait/image-to-image")
images = [
  # YOUR IMAGES
]

payload_images = []
for img in images:
  with open(img, "rb") as image_file:
    payload_images.append(image_file.read())

response = client.image_to_image(..., "A portrait", images=payload_images)
response.save('portrait_you.jpg')
```

## Using the API directly
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
