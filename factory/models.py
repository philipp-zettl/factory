from diffusers import DiffusionPipeline
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiffusionModel:
    def __init__(self, model_name):
        model_config = {
            'stable-diffusion': {
                'base': {
                    'pretrained_model_name_or_path': "stabilityai/stable-diffusion-xl-base-1.0",
                    'torch_dtype': torch.float16,
                    'use_safetensors': True,
                    'variant': 'fp16',
                }

            },
            'small-sd': {
                'base': {
                    'pretrained_model_name_or_path': 'segmind/small-sd',
                    'torch_dtype': torch.float16,
                }
            }
        }.get(model_name)
        self.base = DiffusionPipeline.from_pretrained(**model_config['base'])

        #self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
        self.base.enable_model_cpu_offload()

        self.model_params = {
            'num_inference_steps': 1,
        }

    def predict(self, prompt, options):
        return self.base(
            prompt=prompt,
            **{**self.model_params, **options}
        ).images


models = {
    'stable_diffusion': DiffusionModel('stable-diffusion'),
    'small_diffusion': DiffusionModel('small-sd'),
}

