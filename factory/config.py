import os
from dotenv import load_dotenv

load_dotenv()

LOAD_MODELS = (os.environ.get('LOAD_MODELS', 'True') in ['True', 'true', '1'])
