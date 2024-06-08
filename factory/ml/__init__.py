from factory.ml.manager import ModelManager
from factory import config

print(f'Loading models... Please wait.. {config.LOAD_MODELS}..')
manager = ModelManager(load_models=config.LOAD_MODELS)
