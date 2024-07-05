import os
import yaml


def load_config():
    """YAML 파일을 읽어서 Python 사전으로 반환합니다."""
    with open(f"./config/settings.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config


config = load_config()
