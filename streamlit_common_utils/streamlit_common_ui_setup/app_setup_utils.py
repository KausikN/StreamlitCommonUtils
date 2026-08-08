# Imports
import json
import os

# Main Functions
# UI Utils
## Classes
class UICommon(dict):
    """
    Shared helpers for UI config, data, and cache objects.
    """

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path.replace("\\", "/")
        self.load()

    def load(self):
        """
        Load data from disk.
        """
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf8") as file:
                data = json.load(file)
            if isinstance(data, dict):
                self.clear()
                self.update(data)
        else:
            self.clear()
            self.save()

        return self

    def save(self):
        """
        Save data to disk.
        """
        data_dir = os.path.dirname(self.file_path)
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
        with open(self.file_path, "w", encoding="utf8") as file:
            json.dump(dict(self), file, indent=4)

    def get(self, key, default=None):
        """
        Fetch a value from a dot-separated key path.
        """
        value = self
        for part in key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default

        return value

    def set(self, key, value):
        """
        Set a value at a dot-separated key path.
        """
        parts = key.split(".")
        current = self
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value


class UIConfig(UICommon):
    """
    UI configuration data loaded from a JSON file.
    """

    def __init__(self, file_path="StreamLitGUI/UIConfig.json"):
        super().__init__(file_path)


class UIData(UICommon):
    """
    UI data loaded from a JSON file.
    """

    def __init__(self, file_path="StreamLitGUI/UIData.json"):
        super().__init__(file_path)


class UICache(UICommon):
    """
    UI cache data loaded from a JSON file.
    """

    def __init__(self, file_path="StreamLitGUI/CacheData/Cache.json"):
        super().__init__(file_path)
