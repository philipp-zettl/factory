import base64
from typing import Union
import cv2
import enum

from fastapi import FastAPI, Response
from pydantic import BaseModel
from io import BytesIO
from factory.ml import models
from factory.api.models import GenerationRequest, TextToImageRequest, ImageToImageRequest
from tempfile import NamedTemporaryFile
from PIL import Image


app = FastAPI()


def send_image_response_base64(results: bytes) -> Response:
    response = []
    for result in results:
        buffer = BytesIO()
        result.save(buffer, format="JPEG")
        response.append(buffer.getvalue())
    print(f'Generated {len(response)} images')
    return Response(content=response[0] if len(response) == 1 else response, media_type='image/jpeg')


response_type_map = {
    'image': send_image_response_base64,
}


@app.post('/models/{model_name}/{task}/')
async def generate(
        model_name: str,
        q: GenerationRequest | dict,
        task: str,
        response: Response
    ):
    task = {
        'text-to-image': TextToImageRequest,
        'image-to-image': ImageToImageRequest,
        'image-to-image-multi': ImageToImageRequest,
    }.get(task)(**q, task=task)
    model = models.get(model_name)
    if model is None:
        response.status_code = 400
        return {'error': 'Model not found'}
    try:
        results = model.run_task(task)
    except ValueError as e:
        response.status_code = 400
        return {'error': str(e)}
    return send_image_response_base64(results)

