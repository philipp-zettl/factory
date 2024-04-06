import base64
from typing import Union

from fastapi import FastAPI
from io import BytesIO
from factory.models import stable_diffusion


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post('/gen/')
async def generate(payload: dict):
    task = payload.get('task')
    print(task)
    results = stable_diffusion.predict(task.get('prompt'), task.get('options', {}))
    result = results[0]
    buffer = BytesIO()
    result.save(buffer, format="JPEG")
    result.save('foo.jpeg', format='JPEG')
    img_str = base64.b64encode(buffer.getvalue())
    response = img_str.decode('ascii')
    return {'image': response}
