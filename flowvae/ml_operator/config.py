'''
Docstring for flowvae.ml_operator.config
'''

from typing import Optional, Dict, Any
import inspect
import os
import json

from torch import nn

SCHEMA_VERSION = 1

class ModelConfig():

    def __init__(self,
        config_path: Optional[str] = None,
        model_name: Optional[str] = None,
        init_args: Optional[list] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a config dict that can recreate a model from models.py."""

        self.registry = ModelConfig._get_model_registry()

        if config_path is not None:
            self.load(config_path)
        
        else:
            if model_name is None or model_name not in self.registry:
                available = ", ".join(sorted(self.registry.keys()))
                raise KeyError(f"Unknown model_name '{model_name}'. Available: {available}")
            
            self.config: Dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "model_name": model_name,
                "init_args": init_args or [],
                "init_kwargs": init_kwargs or {},
            }
            if extra:
                self.config["extra"] = extra

    @staticmethod
    def _get_model_registry() -> Dict[str, Any]:
        """Return a registry of callables defined in flowvae.app.wing.models."""
        from flowvae.app.wing import models

        registry: Dict[str, Any] = {}
        for name, obj in vars(models).items():
            if name.startswith("_"):
                continue
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            if getattr(obj, "__module__", "") != models.__name__:
                continue
            registry[name] = obj
        return registry

    def create(self) -> nn.Module:
        """Instantiate a model using the config produced by build_model_config."""
        model_name = self.config.get("model_name")

        factory = self.registry[model_name]
        init_args = self.config.get("init_args")
        init_kwargs = self.config.get("init_kwargs")
        return factory(*init_args, **init_kwargs)
    
    def load(self, path) -> None:

        if not os.path.exists(path):
            raise IOError(f"model config not exist in {path}")
        with open(path, "r", encoding="utf-8") as handle:
            model_config = json.load(handle)

        self.config = model_config

    def save(self, path, exist_ok=False) -> None:
        if not exist_ok and os.path.exists(path):
            raise IOError(f"model config exist in {path}")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.config, handle, indent=2, sort_keys=True)

