import os
import base64
import json
import numpy as np
import cv2
import torch
import qrcode
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, LCMScheduler, AutoPipelineForInpainting, ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from tempfile import NamedTemporaryFile
from io import BytesIO

from factory.ml.models import DiffusionPipelineConfig, ControlNetConfig
from factory.ml.pipelines.general import PipelineMixin
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



class IPAdapterMixin:
    def __init__(self, pipeline=None, base_model_path=None, vae_model_path=None):
        vae_model_path = vae_model_path or  "stabilityai/sd-vae-ft-mse"
        base_model_path = base_model_path or "SG161222/Realistic_Vision_V4.0_noVAE"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = pipeline

    def _load_pipeline(self):
        if self.pipeline is None:
            self.pipeline = DiffusionModel(base_model_path).base


class IPFaceIDPipeline(IPAdapterMixin, PipelineMixin):
    output_type = "image"

    def __init__(self, pipeline, portrait=False):
        super().__init__(pipeline, "SG161222/Realistic_Vision_V4.0_noVAE")
        self.portrait = portrait
        self.ip_ckpt = "./models/ip-adapter/ip-adapter-faceid-portrait_sd15.bin" if portrait else "./models/ip-adapter/ip-adapter-faceid-plusv2_sd15.bin"
        self.device = "cuda"
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

    def _load_pipeline(self):
        super()._load_pipeline()
        # load ip-adapter
        if portrait:
            self.adapter = IPAdapterFaceID(self.pipeline, self.ip_ckpt, self.device, num_tokens=16, n_cond=5)#, torch_dtype=torch.float16)
        else:
            image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
            self.adapter = IPAdapterFaceIDPlus(self.pipeline, image_encoder_path, self.ip_ckpt, self.device, torch_dtype=torch.float16)
        self.face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        self.use_shortcut = self.ip_ckpt == "./models/ip-adapter/ip-adapter-faceid-plusv2_sd15.bin"

    def get_options(self):
        return {
            'task': 'image-to-image',
            'output_type': self.output_type,
            'parameters': {
                **self.model_params,
                'images': ['base64'],
            }
        }

    def get_task(self, is_multi):
        return "image-to-image-multi" if is_multi else "image-to-image"

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
                with NamedTemporaryFile(suffix='.jpg') as f:
                    f.write(base64.decodebytes(img.encode('ascii')))
                    im = cv2.imread(f.name)
                images.append(im)

            params = task.parameters.dict().copy()
            params.pop("images", None)
            prompt = params.pop("prompt", None)
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
        faceid_embeds = []
        for image in images:
            faces = self.face_app.get(image)
            if len(faces) == 0:
                raise ValueError("No face detected")

            faceid_embeds.append(torch.from_numpy(faces[0].normed_embedding).unsqueeze(0).unsqueeze(0))

        faceid_embeds = torch.cat(faceid_embeds, dim=1)
        return self.adapter.generate(
            prompt=prompt,
            shortcut=self.use_shortcut,
            faceid_embeds=faceid_embeds,
            **{**self.model_params, **options}
        )


class IPPipeline(IPAdapterMixin, PipelineMixin):
    output_type = "image"

    def __init__(self, pipeline, plus=False):
        super().__init__(pipeline)
        self.image_encoder_path = "./models/image_encoder/"
        self.ip_ckpt = "./models/ip-adapter/ip-adapter-plus_sd15.bin" if plus else "./models/ip-adapter/ip-adapter_sd15.bin"
        self.device = "cuda"
        self.plus = plus

        self.model_params = {
            'negative_prompt': 'monochrome, lowres, bad anatomy, worst quality, low quality, blurry',
            'num_inference_steps': 45,
            'width': 512,
            'height': 512,
            'scale': 1.0,
        }

    def _load_pipeline(self):
        super()._load_pipeline()
        if self.plus:
            self.adapter = IPAdapterPlus(self.pipeline, self.image_encoder_path, self.ip_ckpt, self.device, num_tokens=16)
        else:
            self.adapter = IPAdapter(self.pipeline, self.image_encoder_path, self.ip_ckpt, self.device)

    def get_task(self, is_multi):
        if is_multi:
            raise ValueError("Invalid task")
        return "image-to-image"

    def get_options(self):
        return {
            'task': 'image-to-image',
            'output_type': self.output_type,
            'parameters': {
                'inputs': 'binary',
                'prompt': 'An image of a cat',
                **self.model_params,
            }
        }

    def run_task(self, task):
        if task.task == "image-to-image":
            params = task.parameters.dict().copy()
            prompt = params.pop("prompt", None)
            img_ip = None
            img_ip = Image.open(BytesIO(base64.decodebytes(task.inputs.encode('ascii'))))
            return self.image_to_image(img_ip, prompt, params)
        raise ValueError("Invalid task")

    def image_to_image(self, image: Image, prompt: Optional[str] = None, options: Optional[dict] = None):
        print(f'prompt: {prompt}, options: {options}')
        return self.adapter.generate(
            pil_image=image,
            prompt=prompt,
            **options
        )


class ImpaintPipeline(PipelineMixin):
    output_type = "image"

    def __init__(self):
        super().__init__()
        self.pipe = None
        self.model_params = {
            'negative_prompt': 'monochrome, lowres, bad anatomy, worst quality, low quality, blurry',
            'num_inference_steps': 45,
            'width': 512,
            'height': 512,
            'scale': 1.0,
        }

    def _load_pipeline(self):
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")

        # set scheduler
        self.pipe.scheduler = LCMScheduler.from_config(self.ipe.scheduler.config)

        # load LCM-LoRA
        self.pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        self.pipe.fuse_lora()


    def get_task(self, is_multi):
        if is_multi:
            raise ValueError("Invalid task")
        return "image-to-image"

    def get_options(self):
        return {
            'task': 'image-to-image',
            'output_type': self.output_type,
            'parameters': {
                'inputs': 'binary',
                'prompt': 'An image of a cat',
                **self.model_params,
            }
        }

    def run_task(self, task):
        if task.task == "image-to-image":
            params = task.parameters.dict().copy()
            prompt = params.pop("prompt", None)
            img_ip = Image.open(BytesIO(base64.decodebytes(task.inputs.encode('ascii'))))
            mask_ip = Image.open(BytesIO(base64.decodebytes(task.mask_image.encode('ascii'))))
            return self.image_to_image(img_ip, mask_ip, prompt, params)
        raise ValueError("Invalid task")

    def image_to_image(self, image: Image, mask_ip: Image, prompt: Optional[str] = None, options: Optional[dict] = None):
        return self.pipe(
            image=image,
            mask_image=mask_ip,
            prompt=prompt,
            **options
        )


class ControlNetPipeline(PipelineMixin):
    output_type = "image"

    def __init__(self, controlnet_pipeline=None, controlnet_config=None):
        super().__init__()
        if controlnet_config is None:
            controlnet_config = "monster-labs/control_v1p_sd15_qrcode_monster"
        for file in os.listdir('./models/configs/controlnet'):
            if file.endswith(".json") and controlnet_config.replace('/', '_') in file:
                with open(os.path.join('./models/configs/controlnet', file)) as f:
                    model_config = ControlNetConfig(**json.load(f))

        self.model_config = model_config
        if not self.model_config:
            raise ValueError("Invalid controlnet config")

        self.controlnet_pipeline = controlnet_pipeline
        self.model_params = {
            **self.model_config.model_params,
            'negative_prompt': 'monochrome, lowres, bad anatomy, worst quality, low quality, blurry',
            'num_inference_steps': 4,
            'controlnet_conditioning_scale': 1.0,
            'guidance_scale': 1.0
        }

    def _load_pipeline(self):
        if self.controlnet_pipeline is None:
            controlnet = ControlNetModel.from_pretrained(self.model_config.base, **self.model_config.config)
            self.controlnet_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                "SimianLuo/LCM_Dreamshaper_v7",
                controlnet=controlnet,
                safety_checker=None,
                torch_dtype=torch.float16,
            ).to("cuda")

        self.pipe = self.controlnet_pipeline
        # set scheduler
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        # load LCM-LoRA
        """
        self.pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        self.pipe.fuse_lora()
        """
        self.pipe.enable_model_cpu_offload()


    def get_task(self, is_multi):
        if is_multi:
            raise ValueError("Invalid task")
        return "image-to-image"

    def get_options(self):
        return {
            'task': 'image-to-image',
            'output_type': 'image',
            'parameters': {
                'inputs': 'binary',
                'prompt': 'An image of a cat',
                **self.model_params,
            }
        }

    def run_task(self, task, img_ip=None):
        if task.task == "image-to-image":
            if img_ip is None:
                img_ip = Image.open(BytesIO(base64.decodebytes(task.inputs.encode('ascii'))))
            params = {**self.model_params, **task.parameters.dict()}
            prompt = params.pop('prompt', None)
            for key in ['width', 'height', 'scale']:
                params.pop(key, None)
            print(f'prompt: {prompt}, params: {params}')
            return self.image_to_image(
                img_ip,
                prompt,
                params
            )
        raise ValueError("Invalid task")

    def image_to_image(self, qrcode_image: Image, prompt: Optional[str] = None, options: Optional[dict] = None):
        print(f'prompt: {prompt}')
        seed = options.pop('seed', 420)
        generator = torch.manual_seed(seed) if seed != -1 else torch.Generator()
         
        return self.pipe(
            prompt=prompt,
            image=qrcode_image,
            width=qrcode_image.width,
            height=qrcode_image.height,
            generator=generator,
            **options
        ).images


class QRCodePipeline(ControlNetPipeline):
    def run_task(self, task, img_ip=None):
        if task.task == "image-to-image":
            if img_ip is None:
                qr_code_content = task.inputs
                img_ip = self.generate_qr_code(qr_code_content)

            return super().run_task(task, img_ip)
        raise ValueError("Invalid task")

    def generate_qr_code(self, content: str, fill_color="black", back_color="white"):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=16,
            border=1,
        )
        qr.add_data(content)
        qr.make(fit=True)
        img = qr.make_image(fill_color=fill_color, back_color=back_color)

        # find smallest image size multiple of 256 that can fit qr
        offset_min = 8 * 16
        w, h = img.size
        w = (w + 255 + offset_min) // 256 * 256
        h = (h + 255 + offset_min) // 256 * 256
        if w > 1024:
            raise Exception("QR code is too large, please use a shorter content")
        bg = Image.new('L', (w, h), 128)

        # align on 16px grid
        coords = ((w - img.size[0]) // 2 // 16 * 16,
                  (h - img.size[1]) // 2 // 16 * 16)
        bg.paste(img, coords)
        return bg
