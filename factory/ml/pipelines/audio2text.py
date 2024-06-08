import torch
from tempfile import NamedTemporaryFile

from .general import PipelineMixin
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer


class SpeechToTextPipeline(PipelineMixin):
    output_type = 'text'

    def __init__(self, model_name="distil-whisper/distil-large-v3"):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _load_pipeline(self):
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(self.device)

        processor = AutoProcessor.from_pretrained(model_name)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=25,
            batch_size=16,
            torch_dtype=torch_dtype,
            device=self.device,
        )

    def get_task(self, is_multi):
        if is_multi:
            raise ValueError("Invalid task")
        return "automatic-speech-recognition"

    def run_task(self, task):
        if task.task == "automatic-speech-recognition":
            return self.speech_to_text(task.inputs, task.parameters)
        raise ValueError("Invalid task")

    def get_options(self):
        return {
            'task': 'automatic-speech-recognition',
            'output_type': 'text',
            'parameters': {
                'send via multiform data': 'An audio file',
            }
        }

    def speech_to_text(self, audio, parameters):
        return self.pipe(audio, **parameters)
