from configparser import ConfigParser
from pathlib import Path

here = Path(__file__).parent
configs_dir = here / ".." / ".." / "configs"

#it's a start, but it's so ugly...

class Config:

    def __init__(self, ini_path=None):
        default_ini = configs_dir / "default.ini"
        ini_path = configs_dir / ini_path 
        self.parser = ConfigParser()
        in_files = [default_ini] + ([ini_path] if ini_path is not None else [])
        self.parser.read(in_files)

    @staticmethod
    def read_file(ini_path):
        global config_instance
        config_instance = Config(ini_path)
    
    @staticmethod
    def get(section, option):
        return config_instance.parser[section][option]

config_instance = None
