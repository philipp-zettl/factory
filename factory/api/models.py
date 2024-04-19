from pydantic import BaseModel
import enum


class Tasks(str, enum.Enum):
    text_to_image = 'text-to-image'
    image_to_image = 'image-to-image'
    image_to_image_multi = 'image-to-image-multi'


class GenerationRequest(BaseModel):
    task: Tasks
    inputs: str | list
    options: dict = {}
    parameters: dict = {}


class TextToImageParameters(BaseModel):
    negative_prompt: str = None
    num_inference_steps: int = 45
    guidance_scale: int = 7
    width: int = 512
    height: int = 512
    s_scale: float = 1.0
    num_samples: int = 1
    seed: int = 2024


class TextToImageRequest(GenerationRequest):
    task: Tasks = Tasks.text_to_image
    parameters: TextToImageParameters


class ImageToImageParameters(BaseModel):
    prompt: str = None
    negative_prompt: str = None
    images: list = []
    num_inference_steps: int = 45
    guidance_scale: int = 7
    width: int = 512
    height: int = 512
    s_scale: float = 1.0
    num_samples: int = 1
    seed: int = 2024

class ImageToImageRequest(GenerationRequest):
    task: Tasks
    inputs: str
    parameters: ImageToImageParameters
