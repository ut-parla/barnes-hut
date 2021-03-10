from configparser import ConfigParser
from pathlib import Path

here = Path(__file__).parent
configs_dir = here / ".." / ".." / "configs"

#it's a start, but it's so ugly...
class Config:

    def __init__(self, ini_path=None):
        default_ini = configs_dir / "default.ini"
        self.parser = ConfigParser()
        self.parser.read(ini_path if ini_path is not None else default_ini)

    @staticmethod
    def read_file(ini_path):
        global config_instance
        config_instance = Config(ini_path)
    
    @staticmethod
    def get(*args, **kwargs):
        return config_instance.parser.get(*args, **kwargs)

    @staticmethod
    def getint(*args, **kwargs):
        return config_instance.parser.getint(*args, **kwargs)


config_instance = None
