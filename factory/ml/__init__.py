from .pipelines import IPFaceIDPipeline, IPPipeline, DiffusionModel, Speech5TTSPipeline, BarkTTSPipeline, SummarizationPipeline, QRCodePipeline


models = {
    #"ssd_1B": DiffusionModel("ssd-1b"),
    #"stable_diffusion": (sd:=DiffusionModel("stable-diffusion")),
    #"SG161222/Realistic_Vision_V4.0_noVAE": (sd15 := DiffusionModel("SG161222/Realistic_Vision_V4.0_noVAE")),
    #"small_diffusion": DiffusionModel("small-sd"),
    #"tiny_diffusion": DiffusionModel("tiny-sd"),

    # image concepts
    #"ip-faces": IPFaceIDPipeline(sd15.base),
    #"ip-faces-portrait": IPFaceIDPipeline(sd15.base, portrait=True),
    #"ip": IPPipeline(sd15.base),
    #"ip-plus": IPPipeline(sd15.base, plus=True),
    "qr": QRCodePipeline(),

    # loras
    #"philipp-zettl/margot_robbie-lora": DiffusionModel("philipp-zettl/margot_robbie-lora"),
    #"philipp-zettl/ssd-margot_robbie-lora": DiffusionModel("philipp-zettl/ssd-margot_robbie-lora"),


    #"tts": Speech5TTSPipeline("tts-1"),
    "bark-tts": BarkTTSPipeline("bark-tts-1"),
    'pszemraj/led-large-book-summary': SummarizationPipeline("pszemraj/led-large-book-summary"),
    
}

