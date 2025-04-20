import yaml

def load_config(config_path="config.yml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)