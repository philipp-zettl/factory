# Factory
A service built on top of the [HuggingFace](https://huggingface.co/) Inference API to provide a general purpose generative AI service.
To allow an easy integration into existing projects, the service provides a RESTful API to interact with the models.

The main intention is to run models that can be used in [hf.easybits.tech](https://hf.easybits.tech/).

> [!IMPORTANT]  
> This service is in a early phase of development and it's API as well as the supported models are changing frequently.
> Use the current state as a reference for future development or as a starting point for your own project.

Project/Service for generative AI

Provides API endpoints for general purpose GAI.

## Features
- Pipeline implementations for different generation tasks
  - text2image
    - Diffusion
    - ONNX Diffusion models
    - incl. LoRA loading
  - image2image
    - IP-Adapter, Portrait and non-portrait (incl. PLUS)
    - ControlNet
    - QR-Code-Monster
    - Impainting with Diffusion models
  - speech2text
    - Whisper
    - DistillWhisper
  - text2speech with different speaker voices
    - SunoAI's BARK
    - microsoft's T5Speech
  - text2text
    -  chat-completion via ANY LLM on HF
    - seq2seq via ANY LM on HF
      - QA
      - summarization
      - translation
      - content generation
      - ...
- API for easy access to fetch available models, can be launched as a standalone service via environment variable `LOAD_MODELS=False`
- Simple configuration via YAML/JSON files in the `./models/configs/` directory


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

## Configuration
To configure models available in the API, you can use the `./models/model_manager.yaml` file.

An example for tested and supported models can be found in the `./models/model_manager_full.yaml` file.

The service supports pytorch models, ONNX models and HuggingFace (i.e. pickle or safetensors) models.

### Considerations
It is important to note that native pytorch models are generally slower than ONNX models when running on CPU devices.

To overcome this limitation, we can convert any model on HuggingFace to an ONNX model using the `optimum-cli` tool
```shell
poetry run optimum-cli export onnx --model_name "model_name" --output_dir "output_dir"
```

Supported pipelines are:
- ONNXDiffusionPipeline
- ONNXChatPipeline

> [!IMPORTANT]
> For transformer based models we can also add a quantization step to further reduce the model size and increase inference speed.

To do so, we can use the `optimum-cli` tool:
```shell
poetry run optimum-cli export onnx --model_name "model_name" --optimize O1 --output_dir "output_dir"
```
check the available optimization levels with `poetry run optimum-cli export onnx --help` for more information.

## API
Launch the API via:
```bash
poetry run uvicorn api_v2:app --port 8000
```

You can find the API documentation at `http://localhost:8000/docs`.

Using the environment variable `LOAD_MODELS` you can spin off a second worker instance to accept `GET` requests for model information.
```bash
LOAD_MODELS=False poetry run uvicorn api_v2:app --port 8001
```

To bundle both services onto a single port on your machine, you can use the provided `docker-compose.yaml` file.
You can change the port mappings in the `docker-compose.yaml` file to your liking.

This basically spins off an instance of NGINX to route the requests to the respective services.
You can find the underlying configuration in the `nginx.conf` file.

## Rotating Models
> [!NOTE]
> Please note that the docker setup currently only supports CPU devices, GPU support is not yet implemented.
> Due to that fact most Text2Image models are not usable in the docker setup.
> To overcome this limitation, we provide an option to use the ONNX-runtime for Text2Image and Text2Text models.
> This also accelerates the inference process.

To rotate models, you can change the configuration provided in the `./models/model_manager.yaml` file.
You can find example configurations in the `./models/model_manager_full.yaml` file.

### Altering base models
In the beginning of the `model_manager.yaml` file, you can find the `base_models` section.
This section defines the base models that are used in modular pipelines.

This is extremely useful when you want to run a single base model with different attachments.
For instance, for stable diffusion based models we can attach LoRA models, ControlNet variants or the IP-Adapter to the base models.
Instead of running multiple instances of the same base model, we can attach the respective models to it and perform inference with the smallest memory footprint possible.

For instance, we can attach the IP-Adapter and IP-Adapter-PLUS to the Realistic Vision model:
```yaml
base_models:
  sd15:
    constructor: DiffusionModel
    args:
      - SG161222/Realistic_Vision_V4.0_noVAE

models:
  ip:
    constructor: IPPipeline
    base_model: sd15
    kwargs:
      plus: False
  ip-plus:
    constructor: IPPipeline
    base_model: sd15
    kwargs:
      plus: True
```

### Model configurations
You can find pre-created configurations in the `./models/configs/` directory.
In essence, the configurations are simple YAML files that define the model, the pipeline and the respective parameters passed to the pipeline.

## Examples:

> [!IMPORTANT]
> Please note that albeit the API provides an interface to generate multiple images at once,
> when using the HF InferenceClient, only one image can be generated at a time.
> This is due to limitations in the HF InferenceClient implementation.


### HuggingFace Inferece API Client
Using the [huggingface.co](https://huggingface.co/) Inference API Client:

### Text2Image
```python
from huggingface_hub import InferenceClient

client = InferenceClient("http://localhost:8000/models/tiny_diffusion")
response = client.text_to_image("A cat in a hat")
response.save('cat_in_hat.png')
```

### Image2Image
#### IP-Adapter (Image Prompt Adapter) - Single Image
```python
from huggingface_hub import InferenceClient

client = InferenceClient("http://localhost:8000/models/ip")
with open("cat.jpg", "rb") as image_file:
    response = client.image_to_image(image_file.read(), "A cat in a hat")
    response.save('cat_in_hat.png')
```
#### IP-Adapter Portrait (single reference image)
```python
from huggingface_hub import InferenceClient

client = InferenceClient("http://localhost:8000/models/ip-faces")

with open('portrait.jpg', 'rb') as image_file:
    response = client.image_to_image(image_file.read(), "A portrait of a young man")
    response.save('portrait_you.jpg')
```

#### IP-Adapter (Image Prompt Adapter) - Multiple Images
```python
from huggingface_hub import InferenceClient

client = InferenceClient("http://localhost:8000/models/ip-faces-portrait")
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

#### QRCode Generator
Currently only via direct HTTP API available
```python
import requests

requests.post('http://localhost:7777/models/qr', json={
    'inputs': 'https://blog.godesteem.de/notes/controlnet/',
    'parameters': {
        'prompt': 'A scawy monsta',
        'num_inference_steps': 15,
        'guidance_scale': 8.0,
        #'seed': 420,
        'negative_prompt': "scawy",
        'controlnet_conditioning_scale': 0.8,
        #'s_scale': 0.7
        'seed': 420,
    }
})

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
res = requests.post('http://localhost:8889/models/ip-faces-multi', json={
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
- [monster-labs](https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster) for the QRCode model
- [huggingface 🤗](https://huggingface.co/) for diffusers and the model-hub
- [pytorch](https://pytorch.org/) for the amazing deep learning framework
- [NGINX](https://www.nginx.com/) for the reverse proxy
