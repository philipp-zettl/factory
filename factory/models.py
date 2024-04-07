from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiffusionModel:
    def __init__(self, model_name):
        model_config = {
            'ssd-1b': {
                'base': {
                    'pretrained_model_name_or_path': 'segmind/SSD-1B',
                    'torch_dtype': torch.float16,
                    'use_safetensors': True,
                    'variant': 'fp16',
                    'safety_checker': None,
                    'requires_safety_checker': False
                },
                'use_scheduler': True,
            },
            'stable-diffusion': {
                'base': {
                    'pretrained_model_name_or_path': "stabilityai/stable-diffusion-xl-base-1.0",
                    'torch_dtype': torch.float16,
                    'use_safetensors': True,
                    'variant': 'fp16',
                    'safety_checker': None,
                    'requires_safety_checker': False
                },
                'use_scheduler': True,
            },
            'small-sd': {
                'base': {
                    'pretrained_model_name_or_path': 'segmind/small-sd',
                    'safety_checker': None,
                    'requires_safety_checker': False,
                    'torch_dtype': torch.float16,
                },
                'use_scheduler': False,
            },
            'tiny-sd': {
                'base': {
                    'pretrained_model_name_or_path': 'segmind/tiny-sd',
                    'torch_dtype': torch.float16,
                    'safety_checker': None,
                    'requires_safety_checker': False,
                },
                'use_scheduler': False,
            },
            'philipp-zettl/margot_robbie-lora': {
                'base': {
                    'pretrained_model_name_or_path': 'segmind/tiny-sd',
                    'torch_dtype': torch.float16,
                    'safety_checker': None,
                    'requires_safety_checker': False,
                },
                'loras': [
                    'philipp-zettl/margot_robbie-lora',
                ],
                'use_scheduler': False,
            },
            'philipp-zettl/ssd-margot_robbie-lora': {
                'base': {
                    'pretrained_model_name_or_path': 'segmind/ssd-1B',
                    'torch_dtype': torch.float16,
                    'use_safetensors': True,
                    'variant': 'fp16',
                    'safety_checker': None,
                    'requires_safety_checker': False,
                },
                'loras': [
                    'philipp-zettl/ssd-margot_robbie-lora',
                ],
                'use_scheduler': True,
            },
            'philipp-zettl/ssd-jon_juarez-lora': {
                'base': {
                    'pretrained_model_name_or_path': 'segmind/ssd-1B',
                    'torch_dtype': torch.float16,
                    'use_safetensors': True,
                    'variant': 'fp16',
                    'safety_checker': None,
                    'requires_safety_checker': False,
                },
                'loras': [
                    'philipp-zettl/ssd-jon_juarez-lora',
                ],
                'use_scheduler': True,
            },
        }.get(model_name)
        self.base = DiffusionPipeline.from_pretrained(**model_config['base'])
        
        if model_config['use_scheduler']:
            config = self.base.scheduler.config
            extra_config = {}
            if 'algorithm_type' in config and config.get('algorithm_type') == 'deis':
                extra_config['algorithm_type'] = 'sigma_min'
            config = dict(**{**config, **extra_config})
            self.base.scheduler = DPMSolverMultistepScheduler.from_config(config)

        if 'loras' in model_config:
            self.base.load_lora_weights(model_config['loras'][0])

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
    'ssd_1B': DiffusionModel('ssd-1b'),
    'stable_diffusion': DiffusionModel('stable-diffusion'),
    'small_diffusion': DiffusionModel('small-sd'),
    'tiny_diffusion': DiffusionModel('tiny-sd'),
    'philipp-zettl/margot_robbie-lora': DiffusionModel('philipp-zettl/margot_robbie-lora'),
    'philipp-zettl/ssd-margot_robbie-lora': DiffusionModel('philipp-zettl/ssd-margot_robbie-lora'),
}

