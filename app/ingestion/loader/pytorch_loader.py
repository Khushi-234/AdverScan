"""
PyTorch model loader implementation.
"""

from pathlib import Path
from typing import Any, Optional, Union
import torch
import torch.nn as nn

from app.ingestion.exceptions import ErrorCategory, ModelLoadError
from app.ingestion.loader.base_loader import BaseModelLoader


class PyTorchLoader(BaseModelLoader):
    """
    Loads PyTorch models from file paths (pickled models, state dicts, TorchScript)
    or validates existing PyTorch model instances.
    """

    def load(
        self,
        model_path: Union[str, Path, nn.Module],
        model_class: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any
    ) -> nn.Module:
        """
        Load a PyTorch model.

        Args:
            model_path: File path to model checkpoint (.pt, .pth), TorchScript model,
                        or an existing nn.Module instance.
            model_class: Model class or instantiated model structure when loading state_dict.
            device: Target device to map stored tensors onto during load.
            **kwargs: Extra parameters passed to torch.load.

        Returns:
            nn.Module: Loaded PyTorch model set in evaluation mode.
        """
        map_location = device or "cpu"

        # Case 1: model_path is already a PyTorch nn.Module or ScriptModule instance
        if isinstance(model_path, (nn.Module, torch.jit.ScriptModule)):
            model_path.eval()
            return model_path

        # Validate file existence for path inputs
        path = Path(model_path)
        if not path.is_file():
            raise ModelLoadError(f"Model file not found at path: {path}")

        try:
            # First try loading via TorchScript if file indicates TorchScript format
            try:
                model = torch.jit.load(str(path), map_location=map_location)
                model.eval()
                return model
            except Exception:
                pass  # Fallback to standard torch.load

            # Support safetensors format directly and safely
            if path.suffix.lower() == ".safetensors":
                try:
                    from safetensors.torch import load_file
                    loaded_obj = load_file(str(path), device=str(map_location))
                except ImportError:
                    raise ModelLoadError("safetensors package is required to load .safetensors models.")
            else:
                # Security boundary: prefer weights_only=True
                unsafe_allow_pickle = kwargs.pop("unsafe_allow_pickle", None)
                safe_mode = kwargs.pop("safe_mode", False)

                if "weights_only" in kwargs:
                    loaded_obj = torch.load(str(path), map_location=map_location, **kwargs)
                else:
                    try:
                        loaded_obj = torch.load(str(path), map_location=map_location, weights_only=True, **kwargs)
                    except Exception as e:
                        # If weights_only failed because it contains full pickled classes
                        if safe_mode or unsafe_allow_pickle is False:
                            from app.ingestion.exceptions import UnsafeModelError
                            raise UnsafeModelError(
                                f"Security risk: Refusing unsafe pickle deserialization from '{path}'. "
                                f"Checkpoint contains non-weight objects that cannot be loaded safely. "
                                f"Pass `unsafe_allow_pickle=True` if you trust this source."
                            ) from e
                        # Fallback for full serialized nn.Module checkpoints with warning
                        loaded_obj = torch.load(str(path), map_location=map_location, weights_only=False, **kwargs)

            # Case 2: loaded_obj is an nn.Module instance
            if isinstance(loaded_obj, nn.Module):
                loaded_obj.eval()
                return loaded_obj

            # Case 3: loaded_obj is a state_dict (dictionary of weights)
            if isinstance(loaded_obj, dict):
                # Check for nested state_dict keys (e.g., 'state_dict' or 'model_state_dict')
                state_dict = loaded_obj
                for key in ["state_dict", "model_state_dict", "model"]:
                    if key in state_dict and isinstance(state_dict[key], dict):
                        state_dict = state_dict[key]
                        break

                if model_class is None:
                    model_class = self._resolve_architecture_from_artifacts(path, state_dict)

                if model_class is None:
                    raise ModelLoadError(
                        "State dictionary detected, but the model architecture could not be determined: "
                        "no model_class, Hugging Face config.json, or timm/torchvision architecture matched. "
                        "Please provide the model class or architecture configuration.",
                        category=ErrorCategory.ARCH_UNKNOWN,
                        rejection_reason="R3",
                    )

                # Instantiate model if model_class is a class, or use instance if already instantiated
                model = model_class() if isinstance(model_class, type) else model_class
                if not isinstance(model, nn.Module):
                    raise ModelLoadError("Provided model_class is not a torch.nn.Module instance or factory.")

                # Verify state_dict key shapes match before accepting
                target_sd = model.state_dict()
                matched_keys = [
                    k for k, v in state_dict.items()
                    if k in target_sd and hasattr(v, "shape") and tuple(v.shape) == tuple(target_sd[k].shape)
                ]
                if not matched_keys:
                    raise ModelLoadError(
                        f"Resolved architecture '{type(model).__name__}' does not match state_dict parameter shapes.",
                        category=ErrorCategory.ARCH_UNKNOWN,
                        rejection_reason="R3",
                    )

                model.load_state_dict(state_dict)
                model.eval()
                return model

            raise ModelLoadError(
                f"Unsupported model payload type loaded from {path}: {type(loaded_obj)}"
            )

        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(f"Failed to load PyTorch model from {path}: {str(e)}") from e

    @staticmethod
    def _resolve_architecture_from_artifacts(path: Path, state_dict: dict) -> Optional[Any]:
        """Try resolving a PyTorch architecture from sibling config.json or README.md without guessing."""
        import json
        parent = path.parent
        cfg_path = parent / "config.json"
        if cfg_path.is_file():
            try:
                cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))
                archs = cfg_dict.get("architectures") or []
                if archs:
                    import transformers
                    from transformers import CONFIG_MAPPING
                    m_type = cfg_dict.get("model_type", "")
                    if m_type and m_type in CONFIG_MAPPING:
                        from app.ingestion.loader.huggingface_loader import patch_hf_config
                        cfg_obj = CONFIG_MAPPING[m_type].from_dict(patch_hf_config(cfg_dict))
                        for arch in archs:
                            if hasattr(transformers, arch):
                                cls = getattr(transformers, arch)
                                return lambda cls=cls, cfg_obj=cfg_obj: cls(cfg_obj)
                arch_name = cfg_dict.get("architecture") or cfg_dict.get("model_name")
                if isinstance(arch_name, str) and arch_name.strip():
                    inst = PyTorchLoader._try_timm_or_torchvision(arch_name.strip(), state_dict)
                    if inst is not None:
                        return inst
            except Exception:
                pass

        readme_path = parent / "README.md"
        if readme_path.is_file():
            try:
                text = readme_path.read_text(encoding="utf-8", errors="ignore")[:8192]
                from app.ingestion.utils.model_utils import _parse_yaml_frontmatter
                meta = _parse_yaml_frontmatter(text)
                for key in ("model_name", "architecture", "base_model"):
                    cand = meta.get(key)
                    if isinstance(cand, str) and cand.strip():
                        inst = PyTorchLoader._try_timm_or_torchvision(cand.strip(), state_dict)
                        if inst is not None:
                            return inst
            except Exception:
                pass
        return None

    @staticmethod
    def _try_timm_or_torchvision(name: str, state_dict: dict) -> Optional[nn.Module]:
        """Instantiate a timm or torchvision architecture by explicit name if shapes match."""
        clean = name.split("/")[-1].strip()
        num_classes = None
        for k, v in state_dict.items():
            if any(k.endswith(s) for s in ("fc.weight", "classifier.weight", "head.weight", "head.fc.weight")) and hasattr(v, "shape") and len(v.shape) == 2:
                num_classes = int(v.shape[0])
                break
        try:
            import timm
            if clean in timm.list_models():
                kwargs = {"pretrained": False}
                if num_classes:
                    kwargs["num_classes"] = num_classes
                return timm.create_model(clean, **kwargs)
        except Exception:
            pass
        try:
            import torchvision.models as tvm
            if hasattr(tvm, clean) and callable(getattr(tvm, clean)):
                kwargs = {"weights": None}
                if num_classes:
                    kwargs["num_classes"] = num_classes
                return getattr(tvm, clean)(**kwargs)
        except Exception:
            pass
        return None
