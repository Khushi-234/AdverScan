"""
Hugging Face model loader implementation for vision, text, time-series, and tabular models.
"""

import json
import logging
from typing import Any, Optional, Union
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoConfig,
    CONFIG_MAPPING,
    PretrainedConfig,
    ViTConfig,
)

from app.ingestion.exceptions import ErrorCategory, ModelLoadError, UnsupportedTaskError
from app.ingestion.loader.base_loader import BaseModelLoader

logger = logging.getLogger(__name__)


def load_keras_or_savedmodel(local_path: str) -> Any:
    """Load a Keras (.keras/.h5) file or TensorFlow SavedModel directory (Keras 2 & 3 compatible)."""
    import tensorflow as tf
    from pathlib import Path

    p = Path(local_path)
    if p.is_file():
        return tf.keras.models.load_model(str(p))

    if p.is_dir():
        for ext in ("*.keras", "*.h5", "*.hdf5"):
            matches = sorted(p.glob(ext))
            if matches:
                return tf.keras.models.load_model(str(matches[0]))

    try:
        return tf.keras.models.load_model(str(p))
    except ValueError as exc:
        if not (p.is_dir() and (p / "saved_model.pb").exists()):
            raise exc

        import keras

        saved = tf.saved_model.load(str(p))
        sigs = getattr(saved, "signatures", {}) or {}
        call_endpoint = "serving_default" if "serving_default" in sigs else (next(iter(sigs.keys()), "serving_default"))
        layer = keras.layers.TFSMLayer(str(p), call_endpoint=call_endpoint)

        sig = sigs.get(call_endpoint)
        if sig is not None:
            try:
                _, kw_spec = sig.structured_input_signature
                first_spec = next(iter(kw_spec.values()), None) if isinstance(kw_spec, dict) else None
                if first_spec is not None and hasattr(first_spec, "shape"):
                    dims = [1 if d is None else int(d) for d in first_spec.shape.as_list()]
                    layer._savedmodel_input_shape = tuple(dims)
            except Exception:
                pass
            try:
                out_specs = sig.structured_outputs
                first_out = next(iter(out_specs.values()), None) if isinstance(out_specs, dict) else out_specs
                if first_out is not None and hasattr(first_out, "shape"):
                    dims = [1 if d is None else int(d) for d in first_out.shape.as_list()]
                    layer._savedmodel_output_shape = tuple(dims)
            except Exception:
                pass
        return layer


def patch_hf_config(cfg_dict: dict) -> dict:
    """Normalize and patch id2label/label2id dictionaries in Hugging Face configs."""
    if not isinstance(cfg_dict, dict):
        return cfg_dict

    if "id2label" in cfg_dict and isinstance(cfg_dict["id2label"], dict):
        normalized_id2label = {}
        for k, v in cfg_dict["id2label"].items():
            try:
                int_key = int(k)
            except (TypeError, ValueError):
                int_key = k
            label_str = str(v).strip() if v is not None else ""
            if not label_str or label_str.lower() == "none":
                label_str = f"Class {int_key}"
            if int_key in normalized_id2label and normalized_id2label[int_key] != label_str:
                raise ValueError(
                    f"Conflicting duplicate id2label key after integer conversion: "
                    f"{int_key} ('{normalized_id2label[int_key]}' vs '{label_str}')"
                )
            normalized_id2label[int_key] = label_str

        cfg_dict["id2label"] = normalized_id2label
        cfg_dict["label2id"] = {
            str(label): idx for idx, label in normalized_id2label.items() if isinstance(idx, int)
        }
        cfg_dict["num_labels"] = len(normalized_id2label)
    elif "label2id" in cfg_dict and isinstance(cfg_dict["label2id"], dict):
        normalized_label2id = {}
        for label, idx in cfg_dict["label2id"].items():
            try:
                int_idx = int(idx)
            except (TypeError, ValueError):
                continue
            label_str = str(label).strip() if label is not None else ""
            if not label_str or label_str.lower() == "none":
                label_str = f"Class {int_idx}"
            normalized_label2id[label_str] = int_idx
        cfg_dict["label2id"] = normalized_label2id
        cfg_dict["id2label"] = {idx: label for label, idx in normalized_label2id.items()}
        if normalized_label2id:
            cfg_dict["num_labels"] = len(normalized_label2id)

    return cfg_dict



class HuggingFaceLoader(BaseModelLoader):
    """
    Loads Hugging Face models across supported domains (image, text, and time-series)
    from Hugging Face Model Hub or local model directories.
    """

    SUPPORTED_DOMAINS = {"image", "text", "time-series", "tabular"}

    def load(
        self,
        model_path: Union[str, Any],
        device: Optional[Union[str, torch.device]] = None,
        domain: Optional[str] = None,
        task_type: Optional[str] = "classification",
        trust_remote_code: bool = False,
        **kwargs: Any
    ) -> nn.Module:
        """
        Load a Hugging Face model.

        Args:
            model_path: Hugging Face hub repository ID (e.g. 'bazyl/gtsrb-model')
                        or directory path containing Hugging Face weights and config.json,
                        or an existing nn.Module instance.
            device: Target device to place model onto.
            domain: Domain identifier ('image' or 'text'). If omitted, auto-detected from model.
            **kwargs: Extra parameters passed to AutoModel loader.

        Returns:
            nn.Module: Loaded PyTorch model set in evaluation mode.
        """
        map_location = str(device) if device is not None else "cpu"

        # If already an nn.Module instance, move to device and return
        if isinstance(model_path, nn.Module):
            model_path.to(map_location)
            model_path.eval()
            return model_path

        model_id = str(model_path).strip()
        domain_str = str(domain).lower().strip().replace("_", "-") if domain is not None else ""
        if domain_str == "timeseries":
            domain_str = "time-series"

        if domain_str not in self.SUPPORTED_DOMAINS:
            # Attempt to auto-detect domain
            from app.ingestion.model_registry import detect_model_domain
            detected = detect_model_domain(model_id)
            if detected in self.SUPPORTED_DOMAINS:
                domain_str = detected
            else:
                domain_str = "unknown"

        if domain_str not in self.SUPPORTED_DOMAINS:
            raise ModelLoadError(
                f"Could not automatically determine model domain for Hugging Face model '{model_id}' (domain: '{domain_str}'). "
                f"Please explicitly specify domain='image', domain='text', domain='time-series', or domain='tabular'."
            )

        # Inspect repository structure and route to appropriate framework loader
        from app.ingestion.utils.model_utils import detect_hf_repo_info
        hf_info = detect_hf_repo_info(model_id)

        repo_type = hf_info.get("repo_type")
        if hf_info.get("rejection_reason") == "R1" or repo_type in ("not_found", "private_or_gated") or hf_info.get("error"):
            err_msg = hf_info.get("error") or f"Repository '{model_id}' is not found, private, or gated."
            raise ModelLoadError(
                f"Cannot access Hugging Face repository '{model_id}' ({repo_type}): {err_msg}",
                rejection_reason="R1",
            )
        if repo_type == "dataset":
            raise ModelLoadError(
                f"Hugging Face repository '{model_id}' appears to be a dataset repository, not a model repository. "
                f"Available files: {hf_info.get('files', [])}",
                rejection_reason="R2",
            )
        if repo_type == "source_only":
            raise ModelLoadError(
                f"Hugging Face repository '{model_id}' contains only source code/documentation and no serialized model weights. "
                f"Available files: {hf_info.get('files', [])}",
                rejection_reason="R2",
            )
        if repo_type == "empty":
            raise ModelLoadError(
                f"Hugging Face repository '{model_id}' contains no files.",
                rejection_reason="R2",
            )

        if hf_info.get("framework") in ("xgboost", "lightgbm", "catboost"):
            from app.ingestion.loader.tree_loader import TreeModelLoader
            return TreeModelLoader().load(model_id, framework=hf_info.get("framework"), **kwargs)

        if hf_info.get("framework") == "sklearn":
            from app.ingestion.loader.sklearn_loader import SklearnLoader
            allow_pickle = kwargs.pop("unsafe_allow_pickle", False)
            return SklearnLoader().load(model_id, unsafe_allow_pickle=allow_pickle, **kwargs)

        if hf_info.get("framework") in ("keras", "tensorflow"):
            try:
                from huggingface_hub import snapshot_download
                local_dir = snapshot_download(model_id)
                return load_keras_or_savedmodel(local_dir)
            except ImportError as imp_err:
                raise ModelLoadError(
                    f"Hugging Face repository '{model_id}' contains a TensorFlow/Keras SavedModel, "
                    f"which requires the 'tensorflow' package. TensorFlow is not installed in the current environment. "
                    f"Please install 'tensorflow' to ingest and evaluate TensorFlow/Keras models.",
                    category=ErrorCategory.MISSING_DEPENDENCY,
                    rejection_reason="R6",
                ) from imp_err

        if hf_info.get("framework") == "onnx":
            try:
                import onnxruntime
                onnx_file = hf_info.get("model_file") or next(
                    (f for f in hf_info.get("files", []) if f.endswith(".onnx")), None
                )
                if not onnx_file:
                    raise ModelLoadError(f"No .onnx model file found in Hugging Face repository '{model_id}'.")
                local_onnx = hf_hub_download(model_id, onnx_file)
                return onnxruntime.InferenceSession(local_onnx)
            except ImportError as imp_err:
                raise ModelLoadError(
                    f"Hugging Face repository '{model_id}' contains an ONNX model, which requires 'onnxruntime'. "
                    f"onnxruntime is not installed in the current environment.",
                    category=ErrorCategory.MISSING_DEPENDENCY,
                    rejection_reason="R6",
                ) from imp_err

        if hf_info.get("framework") == "jax":
            raise UnsupportedTaskError(
                f"Hugging Face repository '{model_id}' contains a JAX/Flax model. "
                f"AdverScan currently supports PyTorch, Scikit-learn, and ONNX models."
            )

        if hf_info.get("files") and not hf_info.get("is_model_repo"):
            raise ModelLoadError(
                f"Hugging Face repository '{model_id}' does not contain recognized serialized model weights. "
                f"Available files: {hf_info.get('files', [])}"
            )

        try:
            # Remove loader-specific kwargs unsupported by Hugging Face AutoModel.from_pretrained
            kwargs.pop("unsafe_allow_pickle", None)
            trust_remote_code = bool(trust_remote_code or kwargs.pop("allow_remote_code", False))

            # 1. Download/read config.json to perform config patching and task verification
            model_config = None
            cfg_dict = {}
            try:
                config_path = hf_hub_download(model_id, "config.json")
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg_dict = json.load(f)

                try:
                    cfg_dict = patch_hf_config(cfg_dict)
                except ValueError as ve:
                    raise ModelLoadError(f"Invalid Hugging Face config for '{model_id}': {ve}") from ve

                model_type = cfg_dict.get("model_type", "")
                if model_type and model_type in CONFIG_MAPPING:
                    model_config = CONFIG_MAPPING[model_type].from_dict(cfg_dict)
                elif model_type == "vit":
                    model_config = ViTConfig.from_dict(cfg_dict)
                else:
                    try:
                        model_config = AutoConfig.from_pretrained(model_id)
                    except Exception:
                        model_config = PretrainedConfig.from_dict(cfg_dict)
            except ModelLoadError:
                raise
            except Exception:
                try:
                    model_config = AutoConfig.from_pretrained(model_id)
                except Exception:
                    model_config = None

            # Detect and reject forecasting/regression tasks for time-series models
            clean_task = str(task_type).lower().strip() if task_type else ""
            if clean_task in ("forecasting", "regression", "time-series-forecasting", "ts-forecasting"):
                raise UnsupportedTaskError(
                    "AdverScan currently supports time-series classification models only; forecasting models are not supported."
                )

            pipeline_tag = (
                cfg_dict.get("pipeline_tag")
                or getattr(model_config, "pipeline_tag", None)
                or getattr(model_config, "task", None)
            )
            if pipeline_tag:
                p_tag = str(pipeline_tag).lower().replace("_", "-")
                if any(k in p_tag for k in ("forecasting", "time-series-forecasting", "time-series-regression")):
                    raise UnsupportedTaskError(
                        "AdverScan currently supports time-series classification models only; forecasting models are not supported."
                    )

            # Check architecture names in config for forecasting / regression
            archs_list = []
            if model_config and getattr(model_config, "architectures", None):
                archs_list.extend([str(a) for a in model_config.architectures])
            if "architectures" in cfg_dict and isinstance(cfg_dict["architectures"], list):
                archs_list.extend([str(a) for a in cfg_dict["architectures"]])

            forecasting_arch_patterns = (
                "forprediction",
                "forforecasting",
                "forregression",
                "timeseriesprediction",
                "timeseriesforecasting",
                "timeseriesregression",
                "chronos",
                "timesfm",
            )
            for arch in archs_list:
                arch_clean = arch.lower()
                if any(pat in arch_clean for pat in forecasting_arch_patterns):
                    raise UnsupportedTaskError(
                        "AdverScan currently supports time-series classification models only; forecasting models are not supported."
                    )

            # 2. Domain-based loader selection
            model = None
            last_exc = None
            if domain_str == "text":
                try:
                    if model_config is not None:
                        model = AutoModelForSequenceClassification.from_pretrained(
                            model_id, config=model_config, **kwargs
                        )
                    else:
                        model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
                except Exception as e:
                    last_exc = e
                    try:
                        auto_kwargs = {"trust_remote_code": trust_remote_code, **kwargs}
                        if model_config is not None:
                            auto_kwargs["config"] = model_config
                        model = AutoModel.from_pretrained(model_id, **auto_kwargs)
                    except Exception as e2:
                        last_exc = e2
                        # Fallback for gliner2 AutoExtractor
                        try:
                            from gliner2 import AutoExtractor  # type: ignore[import-not-found,import]
                            extractor = AutoExtractor.from_pretrained(model_id, **kwargs)
                            model = getattr(extractor, "encoder", getattr(extractor, "model", extractor))
                        except Exception as e3:
                            last_exc = e3
                            model = None

            elif domain_str == "image":
                try:
                    if model_config is not None:
                        model = AutoModelForImageClassification.from_pretrained(
                            model_id, config=model_config, **kwargs
                        )
                    else:
                        model = AutoModelForImageClassification.from_pretrained(model_id, **kwargs)
                except Exception as e:
                    last_exc = e
                    try:
                        auto_kwargs = {"trust_remote_code": trust_remote_code, **kwargs}
                        if model_config is not None:
                            auto_kwargs["config"] = model_config
                        model = AutoModel.from_pretrained(model_id, **auto_kwargs)
                    except Exception as e2:
                        last_exc = e2
                        model = None

            elif domain_str == "time-series":
                import transformers

                # 1. Try architecture specified in model config
                archs = getattr(model_config, "architectures", []) or []
                for arch in archs:
                    if hasattr(transformers, arch):
                        cls = getattr(transformers, arch)
                        try:
                            if model_config is not None:
                                model = cls.from_pretrained(model_id, config=model_config, **kwargs)
                            else:
                                model = cls.from_pretrained(model_id, **kwargs)
                            if model is not None:
                                break
                        except Exception as e:
                            last_exc = e

                # 2. Try common time-series classification model classes in transformers
                if model is None:
                    ts_cls_names = [
                        "PatchTSTForClassification",
                        "PatchTSMixerForTimeSeriesClassification",
                        "AutoModelForSequenceClassification",
                    ]
                    for name in ts_cls_names:
                        if hasattr(transformers, name):
                            cls = getattr(transformers, name)
                            try:
                                if model_config is not None:
                                    model = cls.from_pretrained(model_id, config=model_config, **kwargs)
                                else:
                                    model = cls.from_pretrained(model_id, **kwargs)
                                if model is not None:
                                    break
                            except Exception as e:
                                last_exc = e

                # 3. Fallback to AutoModel
                if model is None:
                    try:
                        auto_kwargs = {"trust_remote_code": trust_remote_code, **kwargs}
                        if model_config is not None:
                            auto_kwargs["config"] = model_config
                        model = AutoModel.from_pretrained(model_id, **auto_kwargs)
                    except Exception as e:
                        last_exc = e
                        model = None

            if model is None:
                err_detail = f": {last_exc}" if last_exc else "."
                err_lower = str(last_exc).lower() if last_exc else ""
                if not trust_remote_code and ("trust_remote_code" in err_lower or "custom code" in err_lower):
                    raise ModelLoadError(
                        f"Model '{model_id}' requires executing remote custom code (`trust_remote_code=True`). "
                        f"Explicit user consent is required before loading custom code.",
                        category=ErrorCategory.CONSENT,
                        rejection_reason="R3",
                    ) from last_exc
                raise ModelLoadError(
                    f"Could not load Hugging Face model '{model_id}' for domain '{domain_str}'{err_detail}"
                ) from last_exc

            model.to(map_location)
            model.eval()
            return model

        except (ModelLoadError, UnsupportedTaskError):
            raise
        except Exception as e:
            raise ModelLoadError(f"Failed to load Hugging Face model '{model_id}' for domain '{domain}': {str(e)}") from e