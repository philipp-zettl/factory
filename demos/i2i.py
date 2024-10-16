# Description: A simple script to interact with the I2I API.
import argparse
from requests import Session
from urllib.parse import urljoin
from base64 import b64encode
import os
import numpy as np

import matplotlib.pyplot as plt


class I2I:
    def __init__(self, base_url):
        self.session = Session()
        self.base_url = base_url

    def build_url(self, path):
        return urljoin(self.base_url, path)

    def get(self, path):
        return self.session.get(self.build_url(path))

    def post(self, path, data):
        return self.session.post(self.build_url(path), json=data)


if __name__ == '__main__':
    i2i = I2I('http://localhost:8001')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=False, default='ip')
    parser.add_argument('--ip', type=str, required=False, default='assets/gabbie-carter-2.jpg')
    parser.add_argument('--prompt', type=str, required=False, default='')
    parser.add_argument('--negative_prompt', type=str, required=False, default='')
    parser.add_argument('--num_inference_steps', type=int, required=False, default=45)
    parser.add_argument('--guidance_scale', type=int, required=False, default=7)
    parser.add_argument('--width', type=int, required=False, default=512)
    parser.add_argument('--height', type=int, required=False, default=768)
    parser.add_argument('--s_scale', type=float, required=False, default=1.0)
    parser.add_argument('--scale', type=float, required=False, default=1.0)
    parser.add_argument('--num_samples', type=int, required=False, default=1)
    parser.add_argument('--seed', type=int, required=False, default=2024)
    parser.add_argument('--output', type=str, required=False, default='ip.jpg')
    parser.add_argument('--single_generation', action='store_true', default=False)

    args = parser.parse_args()

    with open(args.ip, 'rb') as f:
        image_content = f.read()

    image_content = b64encode(image_content).decode()

    payload = {
        'parameters': {
            'prompt': args.prompt or input('Prompt: '),
            'negative_prompt': args.negative_prompt,
            'num_inference_steps': args.num_inference_steps,
            'guidance_scale': args.guidance_scale,
            'width': args.width,
            'height': args.height,
            's_scale': args.s_scale,
            'scale': args.scale,
            'num_samples': args.num_samples,
            'seed': args.seed,
        },
        'inputs': image_content
    }
    if 'ip-faces' in args.model:
        payload['parameters']['images'] = [image_content]
        payload['inputs'] = ''

    os.makedirs(f'results/{args.model}', exist_ok=True)

    prompt = payload['parameters']['prompt'].lower().replace(' ', '_')
    filenames = []

    if args.single_generation:
        res = i2i.post(f'/models/{args.model}', payload)
        filename = f'results/{args.model}/{prompt}.jpg'
        filenames.append(filename)
        with open(filename, 'wb') as f:
            f.write(res.content)
        exit(0)

    for param in np.linspace(0.1, 0.5, 10):
        payload['parameters']['scale'] = param
        res = i2i.post(f'/models/{args.model}', payload)
        filename = f'results/{args.model}/{prompt}-scale={param}.jpg'
        filenames.append(filename)
        with open(filename, 'wb') as f:
            f.write(res.content)

    print('Done!\nGenerating overview...')

    fig, axs = plt.subplots(1, len(filenames), figsize=(5 * len(filenames), 10))
    for i, name in enumerate(filenames):
        axs[i].imshow(plt.imread(name))
        axs[i].axis('off')
        param = name.split('=')[-1].replace('.jpg', '')
        axs[i].set_title(f'Scale: {param}')

    plt.tight_layout()
    plt.savefig(f'results/{args.model}/{prompt}-overview.jpg')
    plt.show()
