from .pipelines import IPFaceIDPipeline, IPPipeline, DiffusionModel


models = {
    #"ssd_1B": DiffusionModel("ssd-1b"),
    #"stable_diffusion": (sd:=DiffusionModel("stable-diffusion")),
    "SG161222/Realistic_Vision_V4.0_noVAE": (sd15 := DiffusionModel("SG161222/Realistic_Vision_V4.0_noVAE")),
    "ip-faces": IPFaceIDPipeline(sd15.base),
    "ip-faces-portrait": IPFaceIDPipeline(sd15.base, portrait=True),
    "ip": IPPipeline(sd15.base),
    #"small_diffusion": DiffusionModel("small-sd"),
    "tiny_diffusion": DiffusionModel("tiny-sd"),
    #"philipp-zettl/margot_robbie-lora": DiffusionModel("philipp-zettl/margot_robbie-lora"),
    #"philipp-zettl/ssd-margot_robbie-lora": DiffusionModel("philipp-zettl/ssd-margot_robbie-lora"),
}

