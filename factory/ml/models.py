from typing import Optional
from factory.utils import BaseModel
from dataclasses import dataclass
import torch


@dataclass
class ModelConfig(BaseModel):
    pretrained_model_name_or_path: str
    torch_dtype: torch.dtype
    use_safetensors: bool = False
    variant: str = None
    safety_checker: str = None
    requires_safety_checker: bool = False
    vae: str = None
    scheduler: Optional[dict] = None


@dataclass
class DiffusionPipelineConfig(BaseModel):
    base: ModelConfig
    use_scheduler: bool = False
    loras: list[str] = None
    enable_model_offload: bool = True

    def __post_init__(self):
        if isinstance(self.base, dict):
            data = self.base
            if 'torch_dtype' in data:
                data['torch_dtype'] = torch.float16 if data['torch_dtype'] == 'float16' else torch.float32

            self.base = ModelConfig(**data)


