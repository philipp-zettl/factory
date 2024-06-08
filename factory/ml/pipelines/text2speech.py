from factory.ml.pipelines.general import PipelineMixin
from factory.ml.pipelines.utils.bark_generation import generate_text_semantic, preload_models
from factory.ml.pipelines.utils.bark_api import semantic_to_waveform
import nltk
import random
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, AutoProcessor, AutoModel
from tempfile import NamedTemporaryFile
from pydub import AudioSegment
from scipy.io.wavfile import write
from io import BytesIO
import torch
import numpy as np


class TTSMixin:
    sample_rate = None
    def speech_to_binary(self, speech):
        assert self.sample_rate is not None, "sample_rate is not set"

        buffer = BytesIO()
        with NamedTemporaryFile(suffix=".wav", delete=False) as f:
            write(f.name, self.sample_rate, np.int16(speech / np.max(np.abs(speech)) * 32767))
            f.seek(0)
            speech = f.read()
            sound = AudioSegment.from_file(f.name, format="wav")
            sound.export(buffer, format="flac")

        return buffer.getvalue()


class Speech5TTSPipeline(PipelineMixin, TTSMixin):
    output_type = "audio"
    sample_rate = 16000

    def __init__(self, speaker_name='tts'):
        self.checkpoint = "microsoft/speecht5_tts"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.current_speaker = None
        self.speaker_embeddings = None
        self.speaker_name = speaker_name

        self.model_params = {}
        self.speakers_map = {
            "tts-1": "./models/tts/cmu_us_bdl_arctic-wav-arctic_a0009.npy",
            "tts-2": "./models/tts/cmu_us_clb_arctic-wav-arctic_a0144.npy",
            "tts-3": "./models/tts/cmu_us_rms_arctic-wav-arctic_b0353.npy",
            "tts-4": "./models/tts/cmu_us_slt_arctic-wav-arctic_a0508.npy",
            "tts-5": "./models/tts/cmu_us_slt_arctic-wav-arctic_a0002.npy",
        }

    def _load_pipeline(self):
        self.processor = SpeechT5Processor.from_pretrained(self.checkpoint)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(self.checkpoint).to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
        self.load_speaker(self.speaker_name)

    def get_options(self):
        return {
            'task': 'text-to-speech',
            'output_type': self.output_type,
            'parameters': {
                'inputs': 'An image of a cat',
                'speaker': 'tts-1',
                **self.model_params,
            },
            'speakers': list(self.speakers_map.keys())
        }

    def load_speaker(self, speaker_name):
        print(f'Switching to speaker: {speaker_name}')

        self.speaker_embeddings = {
        }.get(speaker_name, "cmu_us_bdl_arctic-wav-arctic_a0009.npy")

        self.current_speaker = self.speaker_embeddings
        self.speaker_embeddings = np.load(self.speaker_embeddings)
        self.speaker_embeddings = torch.from_numpy(self.speaker_embeddings).to(self.device).unsqueeze(0)

    def get_task(self, is_multi):
        if is_multi:
            raise ValueError("Invalid task")
        return "text-to-speech"

    def run_task(self, task):
        if task.task == "text-to-speech":
            return self.text_to_speech(task.inputs, task.parameters)
        raise ValueError("Invalid task")

    def text_to_speech(self, text, parameters):
        if len(text.strip()) == 0:
            return self.speech_to_binary(np.zeros(0).astype(np.int16))

        if parameters.get("speaker") and parameters["speaker"] != self.current_speaker:
            self.load_speaker(parameters["speaker"])

        inputs = self.processor(text=text, return_tensors="pt").to(self.device)

        speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder).cpu().numpy().squeeze()

        return self.speech_to_binary(speech)


class BarkTTSPipeline(PipelineMixin, TTSMixin):
    output_type = "audio"
    sample_rate = 24000

    def __init__(self, speaker_name='bark'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_params = {}

        self.speakers = {
            'tts-1': 'v2/en_speaker_6',
            'tts-2': 'v2/en_speaker_9',
        }

    def _load_pipeline(self):
        self.models = preload_models()

    def get_options(self):
        return {
            'task': 'text-to-speech',
            'output_type': self.output_type,
            'info': {
                'description': 'Bark is a text-to-speech model that can generate high-quality speech from text.',
                'sample_rate': self.sample_rate,
            },
            'parameters': {
                'inputs': 'An image of a cat',
                'speaker': 'tts-1',
                **self.model_params,
            },
            'speakers': list(self.speakers.keys())
        }

    def get_task(self, is_multi):
        if is_multi:
            raise ValueError("Invalid task")
        return "text-to-speech"

    def run_task(self, task):
        if task.task == "text-to-speech":
            return self.text_to_speech(task.inputs, {**self.model_params, **task.parameters})
        raise ValueError("Invalid task")

    def text_to_speech(self, text, parameters):
        speaker = self.speakers.get(parameters.get('speaker'), './models/tts/en_speaker_3.npz')
        GEN_TEMP = 0.6

        sentences = nltk.sent_tokenize(text)
        pieces = []
        for sentence in sentences:
            semantic_tokens = generate_text_semantic(
                sentence,
                history_prompt=speaker,
                temp=GEN_TEMP,
                min_eos_p=0.05,  # this controls how likely the generation is to end
            )

            audio_array = semantic_to_waveform(semantic_tokens, history_prompt=speaker,)
            silence = np.zeros(int(random.randint(1, 10)/100 * self.sample_rate))  # between 0.01 and quarter second of silence
            pieces += [audio_array, silence.copy()]


        return self.speech_to_binary(np.concatenate(pieces))

