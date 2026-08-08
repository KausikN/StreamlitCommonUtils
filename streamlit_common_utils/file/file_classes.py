# Imports
import os
import json

from .._common.data_classes import *

# Main Classes
class JsonData(DictData):
    '''
    Shared helpers for UI config, data, and cache objects.
    '''

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path.replace("\\", "/")
        self.load()

    def load(self, file_path=None):
        '''
        Load data from disk.
        '''
        if file_path is not None:
            self.file_path = file_path.replace("\\", "/")
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
        '''
        Save data to disk.
        '''
        data_dir = os.path.dirname(self.file_path)
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
        with open(self.file_path, "w", encoding="utf8") as file:
            json.dump(dict(self), file, indent=4)
