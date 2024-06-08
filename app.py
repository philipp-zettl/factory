import base64
from typing import Union
import cv2

from fastapi import FastAPI, Response
from io import BytesIO
from factory.ml import models
from tempfile import NamedTemporaryFile
from PIL import Image


app = FastAPI()


def send_image_response(results):
    result = results[0]
    buffer = BytesIO()
    result.save(buffer, format="JPEG")
    result.save('foo.jpeg', format='JPEG')
    img_str = base64.b64encode(buffer.getvalue())
    response = img_str.decode('ascii')
    return {'image': response}


@app.post('/gen/')
async def generate(payload: dict):
    task = payload.get('task')
    model_name = task.get('model', 'ssd_1B')
    print(task)
    print(models, model_name)
    print(models.get(model_name))
    model = models.get(model_name)
    if model is None:
        return {'error': 'Model not found'}
    results = model.text_to_image(task.get('prompt'), task.get('options', {}))
    return send_image_response(results)


@app.get('/models/')
async def get_models():
    return {'models': list(models.keys())}


@app.post('/ip-faces/')
async def get_ip(payload: dict, response: Response):
    task = payload.get('task')
    model = models.get('ip-faces')
    if model is None:
        return {'error': 'Model not found'}

    with open('________input.png', 'wb') as f:
        f.write(base64.decodebytes(task.get('ip').encode('ascii')))
        f.seek(0)

    try:
        model.register_ip(f.name)
    except ValueError:
        response.status_code = 400
        return {'error': 'Invalid IP'}

    try:
        results = model.text_to_image(task.get('prompt'), task.get('options', {}))
    except Exception as e:
        response.status_code = 400
        raise e
        return {'error': str(e)}
    return send_image_response(results)


@app.post('/ip-faces-multi/')
async def get_ip_multi(payload: dict, response: Response):
    task = payload.get('task')
    model = models.get('ip-faces-portrait')
    if model is None:
        return {'error': 'Model not found'}

    images = []
    for idx, img in enumerate(task.get('images')):
        with NamedTemporaryFile(delete=False, suffix='.jpg') as f:
            f.write(base64.decodebytes(img.encode('ascii')))
            f.seek(0)
            im = cv2.imread(f.name)
            images.append(im)

    try:
        results = model.images_to_image(images, task.get('prompt'), task.get('options', {}))
    except Exception as e:
        response.status_code = 400
        raise e
        return {'error': str(e)}
    return send_image_response(results)


@app.post('/ip-person/')
async def get_ip_person(payload: dict, response: Response):
    task = payload.get('task')
    model = models.get('ip')
    if model is None:
        return {'error': 'Model not found'}

    with open('________input.png', 'wb') as f:
        f.write(base64.decodebytes(task.get('ip').encode('ascii')))
        f.seek(0)


    try:
        results = model.image_to_image(Image.open(f.name), task.get('options', {}))
    except Exception as e:
        response.status_code = 400
        raise e
        return {'error': str(e)}
    return send_image_response(results)
