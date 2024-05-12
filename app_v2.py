import base64
from typing import Union
import cv2
import enum
import json
import numpy as np
import torch

from fastapi import FastAPI, Response
from pydantic import BaseModel
from io import BytesIO
from factory.ml import models
from factory.api.models import GenerationRequest, TextToImageRequest, ImageToImageRequest, TextToSpeechRequest, TextGenerationRequest
from tempfile import NamedTemporaryFile
from PIL import Image


app = FastAPI()


def send_image_response_base64(results: bytes, format: str = "image/jpeg") -> Response:
    response = []
    if format == 'image/jpeg':
        for result in results:
            buffer = BytesIO()
            result.save(buffer, format=format.split('/')[-1])
            print('converted to buffer')
            response.append(buffer.getvalue())
    elif 'audio' in format:
        response = [results]
    else:
        if isinstance(results, list):
            if isinstance(results[0], dict):
                for result in results:
                    response.append({
                        key: value.tolist() if isinstance(value, (np.ndarray, torch.Tensor)) else value
                        for key, value in result.items()
                    })
                response = json.dumps(response)
            else:
                response = json.dumps(results)
        else:
            response = json.dumps([results.tolist()])
    return Response(content=response[0] if len(response) == 1 else response, media_type=format)


response_type_map = {
    'image': 'image/jpeg',
    'audio': 'audio/flac',
    'text': 'application/json',
}


@app.post('/models/{model_name}')
async def generate(
        model_name: str,
        q: GenerationRequest | dict,
        response: Response
    ):
    is_multi = 'multi' in model_name
    model_name = model_name.replace('-multi', '').replace('_', '/', 1)
    model = models.get(model_name)
    task = model.get_task(is_multi)
    task = {
        'text-to-image': TextToImageRequest,
        'image-to-image': ImageToImageRequest,
        'image-to-image-multi': ImageToImageRequest,
        'text-to-speech': TextToSpeechRequest,
        'text-to-text': TextGenerationRequest,
    }.get(task)(**q, task=task, is_multi=is_multi)
    if model is None:
        response.status_code = 400
        return {'error': 'Model not found'}
    try:
        results = model.run_task(task)
    except ValueError as e:
        response.status_code = 400
        return {'error': str(e)}
    return send_image_response_base64(results, response_type_map.get(model.output_type))


@app.get('/models')
async def get_models():
    return {
        model_name: model.get_options()
        for model_name, model in models.items()
    }


@app.get('/models/{model_name}')
async def get_model(model_name: str):
    return models.get(model_name).get_options()
