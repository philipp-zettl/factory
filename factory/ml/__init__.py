from .pipelines import IPFaceIDPipeline, IPPipeline, DiffusionModel, ONNXDiffusionModel, Speech5TTSPipeline, BarkTTSPipeline, SummarizationPipeline, QRCodePipeline, ChatPipeline, SpeechToTextPipeline


models = {
    #"ssd_1B": DiffusionModel("ssd-1b"),
    #"stable-diffusion": (sd:=ONNXDiffusionModel("stable-diffusion-onnx")),
    #"SG161222/Realistic_Vision_V4.0_noVAE": (sd15 := DiffusionModel("SG161222/Realistic_Vision_V4.0_noVAE")),
    #"small_diffusion": DiffusionModel("small-sd"),
    #"tiny_diffusion": DiffusionModel("tiny-sd"),

    # image concepts
    #"ip-faces": IPFaceIDPipeline(sd15.base),
    #"ip-faces-portrait": IPFaceIDPipeline(sd15.base, portrait=True),
    #"ip": IPPipeline(sd15.base),
    #"ip-plus": IPPipeline(sd15.base, plus=True),
    #"qr": QRCodePipeline(),

    # loras
    #"philipp-zettl/margot_robbie-lora": DiffusionModel("philipp-zettl/margot_robbie-lora"),
    #"philipp-zettl/ssd-margot_robbie-lora": DiffusionModel("philipp-zettl/ssd-margot_robbie-lora"),

    "distil-large-v3": SpeechToTextPipeline("distil-whisper/distil-large-v3"),

    #"tts": Speech5TTSPipeline("tts-1"),
    "bark-tts": BarkTTSPipeline("tts-2"),
    #'pszemraj/led-large-book-summary': SummarizationPipeline("pszemraj/led-large-book-summary"),
    
    'phi-3-mini-128k-chat': ChatPipeline('./models/cpu_and_mobile/phi-3-128k'),
    #'phi-3-mini-4k-chat': ChatPipeline('./models/cpu_and_mobile/phi-3-4k'),
}

