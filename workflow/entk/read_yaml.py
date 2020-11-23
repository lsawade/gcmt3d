import yaml


def read_yaml_file(filename):
    with open(filename, "rb") as fh:
        return yaml.load(fh, Loader=yaml.FullLoader)
