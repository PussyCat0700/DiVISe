import os
import shutil


SPEAKER_ENCODER_PATH = "./CorentinJ/encoder.pt"  # Available at https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Pretrained-models


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))
