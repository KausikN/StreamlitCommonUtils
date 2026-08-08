# Imports
import os

# Main Classes
class DictData(dict):
    '''
    Shared helpers for dict data objects.
    '''

    def __init__(self, data=None):
        super().__init__()
        if data is not None:
            self.update(data)

    def get(self, key, default=None):
        '''
        Fetch a value from a dot-separated key path.
        '''
        value = self
        for part in key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default

        return value

    def set(self, key, value):
        '''
        Set a value at a dot-separated key path.
        '''
        parts = key.split(".")
        current = self
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
