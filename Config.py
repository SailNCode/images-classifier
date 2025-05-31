import yaml

config_path = "configuration.yaml"

def get_config(config_path=config_path) -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
