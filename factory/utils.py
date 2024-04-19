
from dataclasses import dataclass


@dataclass
class BaseModel:
    def dict(self):
        return self.__dict__



