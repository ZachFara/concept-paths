import yaml
from types import SimpleNamespace

class Config:

    def __init__(self, config_yaml_path, ingest_yaml = True):
        self.config_yaml_path = config_yaml_path
        self.raw = {}

        if ingest_yaml:
            self.ingest_config()

    def ingest_config(self):
        with open(self.config_yaml_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError("Config YAML must be a mapping at the top level")
        self.raw = data
        converted = self._to_namespace(data)
        for key, value in converted.items():
            setattr(self, key, value)
        return data

    def _to_namespace(self, value):
        if isinstance(value, dict):
            return {
                key: SimpleNamespace(**self._to_namespace(val))
                if isinstance(val, dict)
                else self._to_namespace(val)
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [self._to_namespace(item) for item in value]
        return value

    def get(self, key, default=None):
        return getattr(self, key, default)

def main():
    cfg = Config("config/test.yaml")
    print(cfg.raw)

if __name__ == "__main__":
    main()
