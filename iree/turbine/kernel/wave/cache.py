import os
from typing import Any, Callable
from .._support.indexing import IndexExpr
import inspect
from pathlib import Path
from .constraints import Constraint
from dataclasses import dataclass
from ..compiler.kernel_codegen import KernelBufferUsage

@dataclass
class WaveCache:
    module_op: str = ""
    kernel_sig: tuple[KernelBufferUsage] = None

class WaveCacheManager(object):
    def __init__(self):
        self.file_cache: set[str] = set()
        self.session_cache: dict[str, WaveCache] = dict()
        default_cache_path = f"{str(Path.home())}/.wave"
        self.cache_path = str(os.environ.get("WAVE_CACHE_DIR", default_cache_path))
        self.counter = 0
        self.update_file_cache()

    def get_hash(
        self,
        constraints: list[Constraint],
        kernel_body: Callable,
        hyperparams: dict[IndexExpr, Any],
        dynamic_symbols: list[IndexExpr, Any],
    ):
        kernel_body_str = inspect.getsource(kernel_body)
        constraints_str = str(constraints)
        hyperparams_str = str(hyperparams)
        dynamic_str = str(dynamic_symbols)
        seed = kernel_body_str + constraints_str + hyperparams_str + dynamic_str
        return hash(seed)

    # File Cache related helpers

    def update_file_cache(self):
        # Early exit if no cache directory found.
        if not os.path.exists(self.cache_path):
            return
        for entry in os.scandir(self.cache_path):
            if entry.name not in self.file_cache:
                self.file_cache.insert(entry.name)

    # def load_cache_from_file(self, entry):
    #     with open(f"{self.cache_path}/{entry}") as f:

    # Session cache related helpers

    def store_kernel(
        self,
        mb,
        constraints: list[Constraint],
        kernel_body: Callable,
        hyperparams: dict[IndexExpr, Any],
        dynamic_symbols: list[IndexExpr, Any],
    ):
        kernel_hash = self.get_hash(constraints, kernel_body, hyperparams, dynamic_symbols)
        self.session_cache[kernel_hash] = mb
        self.store_module_to_file(kernel_hash, mb)

    def get_kernel(
        self,
        constraints: list[Constraint],
        kernel_body: Callable,
        hyperparams: dict[IndexExpr, Any],
        dynamic_symbols: list[IndexExpr, Any],
    ):
        kernel_hash = self.get_hash(constraints, kernel_body, hyperparams, dynamic_symbols)
        if kernel_hash in self.session_cache:
            return self.session_cache[kernel_hash]
        elif kernel_hash in self.file_cache:
            cached_kernel = self.load_cache_from_file(kernel_hash)
            self.session_cache[kernel_hash] = cached_kernel
        return None


def get_cache_manager() -> WaveCacheManager:
    global _global_cache_manager
    if not "_global_cache_manager" in globals():
        _global_cache_manager = WaveCacheManager()
    return _global_cache_manager
