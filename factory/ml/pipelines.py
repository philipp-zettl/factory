import os
import base64
import json
import numpy as np
import cv2
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from tempfile import NamedTemporaryFile
from io import BytesIO

from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL

from factory.ml.models import DiffusionPipelineConfig
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlusXL, IPAdapterFaceIDPlus
from ip_adapter.ip_adapter import IPAdapter, IPAdapterPlus
from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID
import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import torch

from dataclasses import dataclass
from PIL import Image
from typing import Optional


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PipelineMixin:
    def run_task(self, task):
        raise NotImplementedError


class IPAdapterMixin:
    def __init__(self, pipeline=None, base_model_path=None, vae_model_path=None):
        vae_model_path = vae_model_path or  "stabilityai/sd-vae-ft-mse"
        base_model_path = base_model_path or "SG161222/Realistic_Vision_V4.0_noVAE"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if pipeline is None:
            pipeline = DiffusionModel(base_model_path).base

        self.pipeline = pipeline


class IPFaceIDPipeline(IPAdapterMixin, PipelineMixin):
    def __init__(self, pipeline, portrait=False):
        super().__init__(pipeline, "SG161222/Realistic_Vision_V4.0_noVAE")
        self.portrait = portrait
        ip_ckpt = "ip-adapter-faceid-portrait_sd15.bin" if portrait else "ip-adapter-faceid-plusv2_sd15.bin"
        image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        device = "cuda"

        # load ip-adapter
        if portrait:
            self.adapter = IPAdapterFaceID(self.pipeline, ip_ckpt, device, num_tokens=16, n_cond=5, torch_dtype=torch.float16)
        else:
            self.adapter = IPAdapterFaceIDPlus(self.pipeline, image_encoder_path, ip_ckpt, device, torch_dtype=torch.float16)
        self.face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        self.use_shortcut = ip_ckpt == "ip-adapter-faceid-plusv2_sd15.bin"
        self.model_params = {
            "num_inference_steps": 35,
            "negative_prompt": "monochrome, lowres, bad anatomy, worst quality, low quality, blurry",
            "s_scale": 1.0,
            "width": 512,
            "height": 768,
            "num_samples": 1,
            "seed": 2024
        }
        self.faceid_embeds = None
        self.face_image = None

    def run_task(self, task):
        if task.task == "image-to-image":
            for img in task.parameters.images:
                with NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                    f.write(base64.decodebytes(img.encode('ascii')))
                    f.seek(0)
                    self.register_ip(f.name)
            params = task.parameters.dict().copy()
            params.pop("images", None)
            prompt = params.pop("prompt", None)
            return self.text_to_image(prompt, params)
        elif self.portrait and task.task == "image-to-image-multi":
            images = []
            for img in task.parameters.images:
                with NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                    f.write(base64.decodebytes(img.encode('ascii')))
                    f.seek(0)
                    im = cv2.imread(f.name)
                    images.append(im)

            params = task.parameters.dict().copy()
            params.pop("images", None)
            prompt = params.pop("prompt", None)
            print(f'Collecting {len(images)} images, has None? {[i is None for i in images]}')
            return self.images_to_image(images, prompt, params)
        raise ValueError("Invalid task")

    def register_ip(self, ip):
        image = cv2.imread(ip)
        faces = self.face_app.get(image)
        if len(faces) == 0:
            raise ValueError("No face detected")
        self.faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        self.face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224)

    def text_to_image(self, prompt, options):
        if self.faceid_embeds is None or self.face_image is None:
            raise ValueError("No IP registered")

        return self.adapter.generate(
            prompt=prompt,
            shortcut=self.use_shortcut,
            face_image=self.face_image,
            faceid_embeds=self.faceid_embeds,
            **{**self.model_params, **options}
        )

    def images_to_image(self, images, prompt, options):
        face_embeds = []
        for image in images:
            faces = self.face_app.get(image)
            if len(faces) == 0:
                raise ValueError("No face detected")

            face_embeds.append(torch.from_numpy(faces[0].normed_embedding).unsqueeze(0).unsqueeze(0))

        faceid_embeds = torch.cat(face_embeds, dim=1)
        return self.adapter.generate(
            prompt=prompt,
            shortcut=self.use_shortcut,
            faceid_embeds=faceid_embeds,
            **{**self.model_params, **options}
        )


class IPPipeline(IPAdapterMixin, PipelineMixin):
    def __init__(self, pipeline):
        super().__init__(pipeline)
        image_encoder_path = "./image_encoder/"
        ip_ckpt = "ip-adapter_sd15.bin"
        device = "cuda"

        self.adapter = IPAdapter(self.pipeline, image_encoder_path, ip_ckpt, device)

    def run_task(self, task):
        if task.task == "image-to-image":
            params = task.parameters.dict().copy()
            prompt = params.pop("prompt", None)
            img_ip = None
            img_ip = Image.open(BytesIO(base64.decodebytes(task.inputs.encode('ascii'))))
            return self.image_to_image(img_ip, prompt, params)
        raise ValueError("Invalid task")

    def image_to_image(self, image: Image, prompt: Optional[str] = None, options: Optional[dict] = None):
        return self.adapter.generate(
            pil_image=image,
            prompt=prompt,
            **options
        )


class DiffusionModel(PipelineMixin):
    def __init__(self, model_name):
        model_config = {}
        # parse model configs from directory
        for file in os.listdir('./models'):
            if file.endswith(".json") and model_name.replace('/', '_') in file:
                with open(os.path.join('./models', file)) as f:
                    model_config = DiffusionPipelineConfig(**json.load(f))

        model_args = model_config.base.dict()
        vae = model_args.pop('vae', None)
        scheduler = model_args.pop('scheduler', None)
        if scheduler:
            model_args['scheduler'] = DDIMScheduler(**scheduler['args'])
        if vae:
            model_args['vae'] = AutoencoderKL.from_pretrained(vae).to(dtype=torch.float16)

        self.base = DiffusionPipeline.from_pretrained(**model_args)#.to(device)
        
        if model_config.use_scheduler:
            config = self.base.scheduler.config
            extra_config = {}
            if "algorithm_type" in config and config.get("algorithm_type") == "deis":
                extra_config["algorithm_type"] = "sigma_min"
            config = dict(**{**config, **extra_config})
            self.base.scheduler = DPMSolverMultistepScheduler.from_config(config)

        if model_config.loras:
            self.base.load_lora_weights(model_config.loras[0])

        if model_config.enable_model_offload:
            self.base.enable_model_cpu_offload()
        #self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)

        self.model_params = {
            "num_inference_steps": 1,
        }

    def run_task(self, task):
        if task.task == "text-to-image":
            return self.text_to_image(task.inputs, task.parameters.dict() or self.model_params)
        raise ValueError("Invalid task")

    def text_to_image(self, prompt, options):
        return self.base(
            prompt=prompt,
            **{**self.model_params, **options}
        ).images