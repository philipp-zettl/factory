# Load model directly
from transformers import AutoProcessor, AutoModelForSeq2SeqLM, BlipForConditionalGeneration
from datasets import IterableDataset, Dataset
from PIL import Image
from tqdm import tqdm
import argparse
import numpy as np
import  os


class ImageCaptioner:
    def __init__(self, use_gpu):
        device = 'cuda' if use_gpu else 'cpu'
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model.to(device)
        self.device = device

    def caption_image(self, image):
        # unconditional image captioning
        inputs = self.processor(image, return_tensors="pt").to(self.device)

        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)


def sample_generator(directory, prefix='SUPER_PROMPT'):
    for root, dirs, files in os.walk(directory, topdown=False):
        img_files = list(filter(lambda x: '.jpg' in x or '.png' in x, files))
        for name in tqdm(img_files, total=len(img_files), desc='Captioning images...'):
            fn = os.path.join(root, name)
            img = Image.open(fn)

            caption = captioner.caption_image(img)
            yield {"image": img, "text": f'{prefix} {caption}'}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running on GPU requires ~2.5GB VRAM.')

    parser.add_argument('directory')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--dataset-name', dest='dataset_name', help='Name of dataset on huggingface.')

    args = parser.parse_args()

    use_gpu = args.gpu
    captioner = ImageCaptioner(use_gpu)
    dataset = [s for s in sample_generator(args.directory)]

    ds = Dataset.from_dict({"image": [d['image'] for d in dataset], "text": [d['text'] for d in dataset]})

    ds.push_to_hub(f'philipp-zettl/{args.dataset_name or args.directory.split("/")[-1]}', private=True)

