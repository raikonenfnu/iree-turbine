import os
from typing import Any
from .._support.indexing import IndexExpr
import inspect


class WaveCacheManager(object):
    def __init__(self):
        self.file_cache: set[str] = set()
        self.session_cache: dict[str, "vmfb"] = dict()
        default_cache_path = f"{os.path.expanduser("~")}/.wave"
        self.cache_path = str(os.environ.get("WAVE_CACHE_DIR", default_cache_path))
        self.update_file_cache()

    def get_hash(self, constraints: list[Constraints], kernel_body: function, hyperparams: dict[IndexExpr, Any]):
        kernel_body_str = inspect.getsource(kernel_body)
        constraints_str = str(constraints)
        hyperparams_str = str(hyperparams)
        seed = kernel_body_str + constraints_str + hyperparams_str
        return hash(seed)

    def update_file_cache(self):
        for entry in os.scandir(self.cache_path):
            if entry.name not in self.file_cache:
                self.file_cache.insert(entry.name)

    def load_cache_from_file(self, entry):
        with open(f"{self.cache_path}/{entry}") as f:
            

    def store_kernel(self, mb, constraints: list[Constraints], kernel_body: function):
        kernel_hash = self.get_hash(constraints, kernel_body)
        pass

    def get_kernel(self, constraints: list[Constraints], kernel_body: function):
        kernel_hash = self.get_hash(constraints, kernel_body)
        if kernel_hash in self.session_cache:
            return self.session_cache[kernel_hash]
        elif kernel_hash in self.file_cache:
            return self.load_from_file(kernel_hash)
        return None
