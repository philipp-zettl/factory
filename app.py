import base64
from typing import Union

from fastapi import FastAPI
from io import BytesIO
from factory.models import models


app = FastAPI()


@app.post('/gen/')
async def generate(payload: dict):
    task = payload.get('task')
    model_name = task.get('model', 'ssd_1B')
    print(task)
    model = models.get(model_name)
    results = model.predict(task.get('prompt'), task.get('options', {}))
    result = results[0]
    buffer = BytesIO()
    result.save(buffer, format="JPEG")
    result.save('foo.jpeg', format='JPEG')
    img_str = base64.b64encode(buffer.getvalue())
    response = img_str.decode('ascii')
    return {'image': response}
