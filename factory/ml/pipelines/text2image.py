import os
import json
import torch
import torch._dynamo
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from optimum.onnxruntime import ORTStableDiffusionPipeline, ORTStableDiffusionXLPipeline

from factory.ml.models import DiffusionPipelineConfig, ONNXDiffusionPipelineConfig
from factory.ml.pipelines.general import PipelineMixin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiffusionModel(PipelineMixin):
    output_type = "image"

    def __init__(self, model_name):
        model_config = {}
        # parse model configs from directory
        for file in os.listdir('./models/configs'):
            if file.endswith(".json") and model_name.replace('/', '_') in file and not 'onnx' in file:
                with open(os.path.join('./models/configs/', file)) as f:
                    model_config = DiffusionPipelineConfig(**json.load(f))

        self.model_config = model_config

        self.model_params = {
            "num_inference_steps": 1,
        }

    def _load_pipeline(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_args = self.model_config.base.dict()
        vae = model_args.pop('vae', None)
        scheduler = model_args.pop('scheduler', None)
        if scheduler:
            model_args['scheduler'] = DDIMScheduler(**scheduler['args'])
        if vae:
            model_args['vae'] = AutoencoderKL.from_pretrained(vae).to(dtype=torch.float16)

        self.base = DiffusionPipeline.from_pretrained(**model_args).to(device)
        
        
        if self.model_config.use_scheduler:
            config = self.base.scheduler.config
            extra_config = {}
            if "algorithm_type" in config and config.get("algorithm_type") == "deis":
                extra_config["algorithm_type"] = "sigma_min"
            config = dict(**{**config, **extra_config})
            self.base.scheduler = DPMSolverMultistepScheduler.from_config(config)

        if self.model_config.loras:
            self.base.load_lora_weights(self.model_config.loras[0])

        self.base.unet.set_default_attn_processor()
        self.base.vae.set_default_attn_processor()

        if self.model_config.enable_model_offload:
            torch.cuda.empty_cache()
            self.base.enable_sequential_cpu_offload()

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

    def build_embeddings(self, enhanced_prompt, negative_prompt=None):
        max_length = self.base.tokenizer.model_max_length

        input_ids = self.base.tokenizer(enhanced_prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to("cuda")

        negative_ids = self.base.tokenizer(
            negative_prompt or "",
            truncation=False,
            padding="max_length",
            max_length=input_ids.shape[-1],
            return_tensors="pt"
        ).input_ids
        negative_ids = negative_ids.to("cuda")

        concat_embeds = []
        neg_embeds = []
        for i in range(0, input_ids.shape[-1], max_length):
            concat_embeds.append(self.base.text_encoder(input_ids[:, i: i + max_length])[0])
            neg_embeds.append(self.base.text_encoder(negative_ids[:, i: i + max_length])[0])

        prompt_embeds = torch.cat(concat_embeds, dim=1)
        negative_prompt_embeds = torch.cat(neg_embeds, dim=1)
        return prompt_embeds, negative_prompt_embeds

    def text_to_image(self, prompt, options):
        negative_prompt = options.pop('negative_prompt', None)
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.base.encode_prompt(
            prompt,
            negative_prompt=negative_prompt
        )

        return self.base(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            **{**self.model_params, **options}
        ).images


class ONNXDiffusionModel(DiffusionModel):
    def __init__(self, model_name):
        model_config = {}
        # parse model configs from directory
        for file in os.listdir('./models/configs'):
            if file.endswith(".json") and model_name.replace('/', '_') in file and 'onnx' in file:
                with open(os.path.join('./models/configs/', file)) as f:
                    model_config = ONNXDiffusionPipelineConfig(**json.load(f))

        self.model_config = model_config


        self.model_params = {
            "num_inference_steps": 1,
        }

    def _load_pipeline(self):
        model_args = self.model_config.base.dict()
        vae = model_args.pop('vae', None)
        scheduler = model_args.pop('scheduler', None)
        if scheduler:
            model_args['scheduler'] = DDIMScheduler(**scheduler['args'])
        if vae:
            model_args['vae'] = AutoencoderKL.from_pretrained(vae).to(dtype=torch.float16)

        if self.model_config.is_sdxl:
            pipeline = ORTStableDiffusionPipeline
        else:
            pipeline = ORTStableDiffusionXLPipeline
            
        self.base = pipeline.from_pretrained(**model_args, export=True)#.to(device)

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
