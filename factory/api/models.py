from pydantic import BaseModel, Field, ConfigDict
import enum
import numpy as np


class Tasks(str, enum.Enum):
    text_to_image = 'text-to-image'
    image_to_image = 'image-to-image'
    image_to_image_multi = 'image-to-image-multi'
    text_to_speech = 'text-to-speech'
    text_to_text = 'text-to-text'
    chat_completion = 'chat-completion'
    automatic_speech_recognition = 'automatic-speech-recognition'


class GenerationRequest(BaseModel):
    task: Tasks
    inputs: str | list
    options: dict = {}
    parameters: dict = {}
    is_multi: bool = False


class TextToImageParameters(BaseModel):
    model_config = ConfigDict(extra='allow')

    negative_prompt: str = None
    num_inference_steps: int = 45
    guidance_scale: float = 7
    width: int = 512
    height: int = 512
    s_scale: float = 2.0
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
    guidance_scale: float = 7
    width: int = 512
    height: int = 512
    s_scale: float = 1.0
    scale: float = 1.0
    num_samples: int = 1
    seed: int = 2024
    mask_image: str = None
    controlnet_conditioning_scale: float = 1.0


class ImageToImageRequest(GenerationRequest):
    task: Tasks
    inputs: str
    parameters: ImageToImageParameters


class TextToSpeechRequest(GenerationRequest):
    task: Tasks = Tasks.text_to_speech
    inputs: str
    parameters: dict = {}


class TextGenerationRequest(GenerationRequest):
    task: Tasks = Tasks.text_to_image
    inputs: str
    parameters: dict = {}


class ChatCompletionRequest(GenerationRequest):
    task: Tasks = Tasks.chat_completion
    inputs: list[dict] = []
    parameters: dict = {}


class SpeechToTextRequest(GenerationRequest):
    task: Tasks = Tasks.automatic_speech_recognition
    parameters: dict = {}
    inputs: bytes = None
