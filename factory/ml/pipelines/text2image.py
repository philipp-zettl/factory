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
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, AutoTokenizer, AutoModel, AutoProcessor

from factory.ml.models import DiffusionPipelineConfig
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class DiffusionModel(PipelineMixin):
    output_type = "image"

    def __init__(self, model_name):
        model_config = {}
        # parse model configs from directory
        for file in os.listdir('./models/configs'):
            if file.endswith(".json") and model_name.replace('/', '_') in file:
                with open(os.path.join('./models/configs/', file)) as f:
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

    def get_options(self):
        return {
            'task': 'text-to-image',
            'output_type': self.output_type,
            'parameters': {
                'inputs': 'An image of a cat',
                **self.model_params,
            }
        }

    def get_task(self, is_multi):
        if is_multi:
            raise ValueError("Invalid task")
        return "text-to-image"

    def run_task(self, task):
        if task.task == "text-to-image":
            return self.text_to_image(task.inputs, task.parameters.dict() or self.model_params)
        raise ValueError("Invalid task")

    def text_to_image(self, prompt, options):
        return self.base(
            prompt=prompt,
            **{**self.model_params, **options}
        ).images


