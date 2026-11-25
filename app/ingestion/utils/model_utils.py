"""
Model utility and detection functions for AdverScan ingestion module.
Includes domain detection, framework detection, and environment setup utilities.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from app.ingestion.presets import DATASET_TO_RECOMMENDED_MODEL, MODEL_PRESETS
from app.ingestion.utils.logger import log_ingestion_event


def is_forecasting_or_regression(
    model_or_path: Any,
    config: Optional[Any] = None,
    task_type: Optional[str] = None,
) -> bool:
    """
    Detect whether a time-series model represents a forecasting/regression task
    rather than classification, based on pipeline_tag, architecture, config, or task info.
    """

# Input
#  │
#  ├── 1. Already have a Config object? ──> Inspect pipeline_tag & architectures list
#  │
#  ├── 2. Input is a string/model ID?
#  │        ├── 2a. Check model name for keywords
#  │        ├── 2b. Try local AutoConfig (local_files_only=True)
#  │        └── 2c. Fallback to HF Hub config.json download & parse
#  │
#  └── 3. Input is an in-memory PyTorch instance? ──> Inspect class name

    clean_task = str(task_type).lower().strip() if task_type else ""
    if clean_task in ("forecasting", "regression", "time-series-forecasting", "ts-forecasting"):
        return True

    forecasting_keywords = (
        "forprediction",
        "forforecasting",
        "forregression",
        "timeseriesprediction",
        "timeseriesforecasting",
        "timeseriesregression",
        "chronos",
        "timesfm",
    )

    # Check config object if provided
    if config is not None:
        p_tag = getattr(config, "pipeline_tag", None) or getattr(config, "task", None)
        if p_tag and any(k in str(p_tag).lower() for k in ("forecasting", "regression")):
            return True
        archs = getattr(config, "architectures", []) or []
        for arch in archs:
            if any(k in str(arch).lower() for k in forecasting_keywords):
                return True

    # Check string model_or_path
    if isinstance(model_or_path, str):
        clean_name = model_or_path.lower()
        if any(k in clean_name for k in ("forprediction", "forforecasting", "forregression", "forecaster")):
            return True

        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_or_path, local_files_only=True)
            if cfg is not None:
                return is_forecasting_or_regression(None, config=cfg)
        except Exception:
            pass

        try:
            from huggingface_hub import hf_hub_download
            cfg_path = hf_hub_download(model_or_path, "config.json")
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw_cfg = json.load(f)
            p_tag = raw_cfg.get("pipeline_tag")
            if p_tag and any(k in str(p_tag).lower() for k in ("forecasting", "regression")):
                return True
            archs = raw_cfg.get("architectures", [])
            for arch in archs:
                if any(k in str(arch).lower() for k in forecasting_keywords):
                    return True
        except Exception:
            pass

    # Check in-memory model or class
    if model_or_path is not None:
        cls_name = model_or_path.__class__.__name__.lower()
        if any(k in cls_name for k in ("forprediction", "forforecasting", "forregression", "forecaster", "forecasting")):
            return True

    return False


def is_sklearn_model(model_or_path: Any) -> bool:
    """
    Check if a model object or file path represents a scikit-learn estimator.
    """
    if isinstance(model_or_path, (str, Path)):
        p_str = str(model_or_path).lower()
        return p_str.endswith((".joblib", ".pkl", ".pickle"))

    # Duck typing for sklearn estimators
    if hasattr(model_or_path, "predict") and (
        hasattr(model_or_path, "fit") or hasattr(model_or_path, "classes_") or hasattr(model_or_path, "_estimator_type")
    ):
        # Exclude PyTorch modules that might define predict()
        if isinstance(model_or_path, torch.nn.Module):
            return False
        return True

    return False


def detect_model_source(model_or_path: Any) -> str:
    """
    Determine model source ('local', 'in_memory', 'huggingface', 'url').
    Distinguishes source (where it comes from) from framework (how it executes).
    """
    if not isinstance(model_or_path, (str, Path)):
        return "in_memory"

    path_str = str(model_or_path).strip()
    if path_str.startswith(("http://", "https://")):
        return "url"

    p = Path(path_str)
    if p.exists():
        return "local"

    if "/" in path_str and not path_str.startswith((".", "/")):
        return "huggingface"

    return "local"


_HF_REPO_INFO_CACHE: Dict[str, Dict[str, Any]] = {}


def _parse_yaml_frontmatter(text: str) -> Dict[str, Any]:
    """Extract simple key/value and list fields from README.md YAML frontmatter without requiring PyYAML."""
    if not text or not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    yaml_block = parts[1]
    result: Dict[str, Any] = {}
    current_key: Optional[str] = None
    for raw_line in yaml_block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- ") and current_key:
            val = line[2:].strip().strip("'\"")
            existing = result.get(current_key)
            if isinstance(existing, list):
                existing.append(val)
            else:
                result[current_key] = [val]
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            current_key = k
            if v:
                if v.startswith("[") and v.endswith("]"):
                    result[k] = [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
                else:
                    result[k] = v
            else:
                result[k] = []
    return result


def clear_hf_repo_cache(repo_id: Optional[str] = None) -> None:
    """Clear cached Hugging Face repository inspection metadata."""
    if repo_id is None:
        _HF_REPO_INFO_CACHE.clear()
    else:
        _HF_REPO_INFO_CACHE.pop(str(repo_id).strip(), None)


def detect_hf_repo_info(repo_id: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Inspect a Hugging Face repository's files, README.md/model card YAML metadata,
    tags, and config.json.
    Distinguishes model repositories from dataset, source-only, or incomplete repositories.
    Uses caching to avoid repeated API calls.
    """
    clean_id = repo_id.strip()
    if force_refresh:
        _HF_REPO_INFO_CACHE.pop(clean_id, None)
    if clean_id in _HF_REPO_INFO_CACHE:
        return _HF_REPO_INFO_CACHE[clean_id]

    res: Dict[str, Any] = {
        "framework": "unknown",
        "domain": "unknown",
        "task_type": "classification",
        "library_name": None,
        "pipeline_tag": None,
        "files": [],
        "model_file": None,
        "is_model_repo": False,
        "repo_type": "unknown",
        "error": None,
    }

    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()

        # 1. Fetch file list
        files: List[str] = []
        try:
            files = api.list_repo_files(clean_id)
            res["files"] = files
        except Exception as exc:
            res["error"] = str(exc)
            err_lower = str(exc).lower()
            if "404" in err_lower or "401" in err_lower or "not found" in err_lower:
                res["repo_type"] = "not_found"
            files = []

        # 2. Fetch model metadata / card data
        tags: List[str] = []
        card_data: Dict[str, Any] = {}
        try:
            info = api.model_info(clean_id)
            res["library_name"] = getattr(info, "library_name", None)
            res["pipeline_tag"] = getattr(info, "pipeline_tag", None)
            tags = [t.lower() for t in (getattr(info, "tags", []) or [])]
            raw_card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
            if isinstance(raw_card, dict):
                card_data = raw_card
            elif hasattr(raw_card, "to_dict"):
                card_data = raw_card.to_dict() or {}
        except Exception:
            info = None

        # Inspect README.md YAML frontmatter if metadata is missing
        yaml_meta: Dict[str, Any] = {}
        if any(f.lower() == "readme.md" for f in files):
            try:
                readme_name = next(f for f in files if f.lower() == "readme.md")
                readme_path = hf_hub_download(clean_id, readme_name)
                with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                    yaml_meta = _parse_yaml_frontmatter(f.read(8192))
                if not res["library_name"] and yaml_meta.get("library_name"):
                    res["library_name"] = str(yaml_meta["library_name"])
                if not res["pipeline_tag"] and yaml_meta.get("pipeline_tag"):
                    res["pipeline_tag"] = str(yaml_meta["pipeline_tag"])
                if isinstance(yaml_meta.get("tags"), list):
                    for t in yaml_meta["tags"]:
                        if str(t).lower() not in tags:
                            tags.append(str(t).lower())
            except Exception:
                pass

        # Inspect config.json if present
        cfg_data: Dict[str, Any] = {}
        if "config.json" in files:
            try:
                cfg_path = hf_hub_download(clean_id, "config.json")
                with open(cfg_path, "r", encoding="utf-8", errors="ignore") as f:
                    loaded_cfg = json.load(f)
                if isinstance(loaded_cfg, dict):
                    cfg_data = loaded_cfg
                    if not res["pipeline_tag"] and cfg_data.get("pipeline_tag"):
                        res["pipeline_tag"] = str(cfg_data["pipeline_tag"])
            except Exception:
                pass

        # Inspect metadata.json if present
        metadata_dict: Dict[str, Any] = {}
        if "metadata.json" in files:
            try:
                meta_path = hf_hub_download(clean_id, "metadata.json")
                with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
                    loaded_meta = json.load(f)
                if isinstance(loaded_meta, dict):
                    metadata_dict = loaded_meta
            except Exception:
                pass

        from app.ingestion.artifact_detector import ModelArtifactDetector
        insp = ModelArtifactDetector.inspect_file_list(
            files=files,
            library_name=res["library_name"],
            pipeline_tag=res["pipeline_tag"],
            tags=tags,
            config_dict=cfg_data,
            metadata_dict=metadata_dict,
            repo_id=clean_id,
        )

        res["framework"] = insp.framework
        res["domain"] = insp.domain if insp.domain != "unknown" else res["domain"]
        res["task_type"] = insp.task_type
        res["model_file"] = insp.model_file
        res["is_model_repo"] = insp.is_model_repo
        res["repo_type"] = insp.repo_type
        res["preprocessing_files"] = insp.preprocessing_files
        res["calibration_files"] = insp.calibration_files
        res["metadata_files"] = insp.metadata_files
        res["weight_files"] = insp.weight_files
        res["feature_names"] = insp.feature_names
        res["raw_feature_names"] = insp.raw_feature_names
        res["num_features"] = insp.num_features
        res["architecture"] = insp.architecture
        if res["framework"] == "sklearn":
            res["domain"] = "tabular"

        # Collect repository dataset files and dataset identifier candidates
        ds_exts = (".csv", ".tsv", ".parquet", ".xlsx", ".jsonl")
        res["dataset_files"] = [
            f for f in files
            if f.lower().endswith(ds_exts) and not f.startswith(".")
        ]
        ds_candidates: List[str] = []
        for ds_source in (card_data.get("datasets"), yaml_meta.get("datasets"), cfg_data.get("datasets")):
            if isinstance(ds_source, list):
                for item in ds_source:
                    if item and str(item).strip() not in ds_candidates:
                        ds_candidates.append(str(item).strip())
            elif isinstance(ds_source, str) and ds_source.strip() not in ds_candidates:
                ds_candidates.append(ds_source.strip())
        for key in ("dataset_name", "dataset"):
            val = cfg_data.get(key) or metadata_dict.get(key)
            if isinstance(val, str) and val.strip() and val.strip() not in ds_candidates:
                ds_candidates.append(val.strip())
        meta_data_field = metadata_dict.get("data")
        if isinstance(meta_data_field, dict):
            for key in ("dataset", "dataset_name", "file", "path"):
                val = meta_data_field.get(key)
                if isinstance(val, str) and val.strip() and val.strip() not in ds_candidates:
                    ds_candidates.append(val.strip())
        res["dataset_candidates"] = ds_candidates
        res["target_column"] = (
            metadata_dict.get("target_column")
            or metadata_dict.get("target")
            or cfg_data.get("target_column")
        )

        # 5. Detect Task Type
        p_tag = (res.get("pipeline_tag") or str(card_data.get("pipeline_tag") or "")).lower()
        if any(k in p_tag for k in ("forecasting", "time-series-forecasting", "regression")):
            res["task_type"] = "forecasting" if "forecasting" in p_tag else "regression"
        else:
            res["task_type"] = "classification"

    except Exception as exc:
        res["error"] = str(exc)

    if res.get("error") is None and res.get("files"):
        _HF_REPO_INFO_CACHE[clean_id] = res
    return res


def detect_model_framework(
    model_or_path: Any,
    framework_override: Optional[str] = None,
    model_class: Optional[Any] = None,
) -> str:
    """
    Determine model framework ('sklearn', 'pytorch', 'tensorflow', 'keras', 'onnx', 'jax').
    Never conflates source ('huggingface') with execution framework.
    """
    if framework_override:
        clean = framework_override.lower().strip()
        if clean in (
            "pytorch", "pytorch_local", "sklearn", "tensorflow", "keras", "onnx", "jax",
            "xgboost", "lightgbm", "catboost", "openvino", "tensorrt", "coreml", "tflite", "paddle", "mxnet",
        ):
            return clean

    # 1. In-memory inspection
    if isinstance(model_or_path, (torch.nn.Module, torch.jit.ScriptModule)):
        return "pytorch"

    if model_class is not None and isinstance(model_class, type) and issubclass(model_class, torch.nn.Module):
        return "pytorch"

    # Duck-typing for tree models
    t_name = model_or_path.__class__.__name__.lower() if not isinstance(model_or_path, (str, Path)) else ""
    if "xgb" in t_name or "xgboost" in t_name:
        return "xgboost"
    if "lgbm" in t_name or "lightgbm" in t_name:
        return "lightgbm"
    if "catboost" in t_name:
        return "catboost"

    if is_sklearn_model(model_or_path):
        return "sklearn"

    # 2. File extension and repository inspection
    if isinstance(model_or_path, (str, Path)):
        p_str = str(model_or_path).lower()
        if p_str.endswith((".pt", ".pth", ".safetensors", ".ckpt", ".bin")):
            return "pytorch"
        if p_str.endswith((".joblib", ".pkl", ".pickle")):
            return "sklearn"
        if p_str.endswith((".onnx",)):
            return "onnx"
        if p_str.endswith((".tflite",)):
            return "tflite"
        if p_str.endswith((".cbm",)):
            return "catboost"
        if p_str.endswith((".engine", ".plan")):
            return "tensorrt"
        if p_str.endswith((".mlmodel", ".mlpackage")):
            return "coreml"
        if p_str.endswith((".h5", ".hdf5", ".keras")):
            return "keras"
        if p_str.endswith((".pb",)):
            return "tensorflow"
        if p_str.endswith((".msgpack", ".flax")):
            return "jax"
        if p_str.endswith((".pdmodel", ".pdparams")):
            return "paddle"
        if p_str.endswith((".params",)):
            return "mxnet"
        if p_str.endswith((".xml",)):
            return "openvino"
        if p_str.endswith((".ubj",)):
            return "xgboost"
        if ("xgb" in p_str or "xgboost" in p_str) and p_str.endswith((".json", ".model")):
            return "xgboost"
        if p_str.endswith(".json") and Path(str(model_or_path)).is_file():
            try:
                with open(str(model_or_path), "r", encoding="utf-8", errors="ignore") as f:
                    d = json.load(f)
                if isinstance(d, dict) and ("learner" in d or "version" in d):
                    return "xgboost"
            except Exception:
                pass

        # Check Hugging Face repos dynamically
        if "/" in p_str and not p_str.startswith((".", "/")):
            hf_info = detect_hf_repo_info(str(model_or_path))
            if hf_info.get("framework") and hf_info["framework"] != "unknown":
                return hf_info["framework"]

            if p_str.startswith("scikit-learn/") or "sklearn" in p_str:
                return "sklearn"
            if p_str.startswith("keras-io/") or "keras" in p_str:
                return "keras"

            try:
                from huggingface_hub import hf_hub_download
                cfg_p = hf_hub_download(str(model_or_path), "config.json", local_files_only=True)
                with open(cfg_p, "r", encoding="utf-8") as f:
                    cfg_d = json.load(f)
                if "sklearn" in cfg_d:
                    return "sklearn"
            except Exception:
                pass

    # 3. Default fallback
    if framework_override:
        return framework_override.lower().strip()

    return "pytorch"


def inspect_model_evidence(
    model_or_path: Any,
    framework: Optional[str] = None,
    domain: Optional[str] = None,
    sample_input: Optional[Any] = None,
) -> List[str]:
    """
    Collect structured evidence explaining framework, domain, and capability detection.
    Useful for audit logs and actionable error reporting.
    """
    evidence: List[str] = []
    source = detect_model_source(model_or_path)
    fw = framework or detect_model_framework(model_or_path)

    evidence.append(f"Model source resolved as '{source}'")
    evidence.append(f"Execution framework resolved as '{fw}'")

    if fw == "sklearn":
        cls_name = model_or_path.__class__.__name__ if not isinstance(model_or_path, (str, Path)) else "Serialized"
        evidence.append(f"Detected scikit-learn estimator ({cls_name})")
        if hasattr(model_or_path, "predict"):
            evidence.append("Model exposes predict()")
        if hasattr(model_or_path, "predict_proba"):
            evidence.append("Model exposes predict_proba()")
        if hasattr(model_or_path, "decision_function"):
            evidence.append("Model exposes decision_function()")
        if hasattr(model_or_path, "n_features_in_"):
            evidence.append(f"Model fitted on n_features_in_ = {model_or_path.n_features_in_}")
        if hasattr(model_or_path, "classes_"):
            evidence.append(f"Detected classes count = {len(model_or_path.classes_)}")
    elif fw == "pytorch":
        evidence.append("Detected PyTorch neural network architecture")

    if domain:
        evidence.append(f"Data domain identified as '{domain}'")

    if sample_input is not None:
        if hasattr(sample_input, "shape"):
            evidence.append(f"Sample input provided with shape {tuple(sample_input.shape)}")
        if hasattr(sample_input, "columns"):
            evidence.append(f"Sample tabular DataFrame columns: {list(sample_input.columns)}")

    return evidence


def detect_model_domain(
    model_or_path: Any,
    sample_input: Optional[Any] = None,
    processor_name: Optional[str] = None,
    model_class: Optional[Any] = None,
) -> str:
    """
    Automatically detect the data domain ('image', 'text', 'tabular', or 'time-series') from a model identifier,
    in-memory model instance, processor/tokenizer name, or sample input.
    Returns 'unknown' if domain cannot be determined with certainty, never falling back to 'image'.
    """
    # 0. Check if model is a scikit-learn estimator
    if is_sklearn_model(model_or_path):
        return "tabular"

    # 1. Preset registry check
    if isinstance(model_or_path, str):
        clean_id = model_or_path.strip()
        if clean_id in MODEL_PRESETS:
            return MODEL_PRESETS[clean_id].get("data_domain", "unknown")
        if clean_id in DATASET_TO_RECOMMENDED_MODEL:
            target = DATASET_TO_RECOMMENDED_MODEL[clean_id]
            if target in MODEL_PRESETS:
                return MODEL_PRESETS[target].get("data_domain", "unknown")

    # 2. Check processor / tokenizer name
    if processor_name and isinstance(processor_name, str):
        proc_lower = processor_name.lower().replace("_", "-")
        if any(k in proc_lower for k in ("tabular", "tab-processor", "tabular-processor", "tabpfn")):
            return "tabular"
        if any(k in proc_lower for k in ("timeseries", "time-series", "ts-processor")):
            return "time-series"
        if any(k in proc_lower for k in ("tokenizer", "bert", "roberta", "gpt", "t5", "bart", "text")):
            return "text"
        if any(k in proc_lower for k in ("image", "vision", "vit", "resnet", "feature-extractor")):
            return "image"

    # 3. Check sample input
    if sample_input is not None:
        if "pandas" in getattr(sample_input.__class__, "__module__", ""):
            return "time-series"
        if isinstance(sample_input, (str, list)) and (
            isinstance(sample_input, str) or (len(sample_input) > 0 and isinstance(sample_input[0], str))
        ):
            return "text"
        if isinstance(sample_input, dict):
            if any(k in sample_input for k in ("input_ids", "attention_mask", "token_type_ids")):
                return "text"
            if any(k in sample_input for k in ("pixel_values", "images")):
                return "image"
            if any(k in sample_input for k in ("past_values", "series", "sequence")):
                return "time-series"
        if "PIL" in getattr(sample_input.__class__, "__module__", ""):
            return "image"
        if isinstance(sample_input, torch.Tensor) or (hasattr(sample_input, "shape") and hasattr(sample_input, "dtype")):
            if len(sample_input.shape) == 4:
                return "image"
            if len(sample_input.shape) == 3 and sample_input.shape[0] in (1, 3, 4) and sample_input.shape[1] > 10 and sample_input.shape[2] > 10:
                return "image"
            dtype_str = str(getattr(sample_input, "dtype", ""))
            if len(sample_input.shape) in (1, 2):
                if any(t in dtype_str for t in ("int", "long")):
                    return "text"
            # 3D float tensor that is not a 2D image (e.g. single channel sequence [B, 1, L] or [B, L, 1])
            if len(sample_input.shape) == 3 and any(t in dtype_str for t in ("float",)):
                if sample_input.shape[1] == 1 or sample_input.shape[2] == 1:
                    return "time-series"

    # 4. Check in-memory model or class
    target_model = model_or_path if isinstance(model_or_path, torch.nn.Module) else None
    if target_model is None and callable(model_class) and issubclass(model_class, torch.nn.Module):
        try:
            target_model = model_class()
        except Exception:
            pass

    if target_model is not None:
        cls_name = target_model.__class__.__name__.lower()
        if any(k in cls_name for k in ("tabular", "tabnet", "fttransformer", "tabtransformer", "tabularmlp", "tabularclassifier", "tabpfn")):
            return "tabular"
        ts_model_keywords = (
            "patchtst", "patchtsmixer", "timesnet", "informer", "autoformer",
            "1dcnn", "cnn1d", "conv1d", "lstm", "gru", "timeseries", "time_series", "time-series"
        )
        if any(k in cls_name for k in ts_model_keywords):
            return "time-series"
        if any(k in cls_name for k in ("vit", "resnet", "conv", "vision", "image", "swin", "deit", "cnn", "vgg", "alexnet")):
            return "image"
        if any(k in cls_name for k in ("bert", "roberta", "gpt", "t5", "text", "nlp", "tokenizer", "bart")):
            return "text"

        has_conv = False
        has_conv1d = False
        has_rnn = False
        has_embed = False
        for module in target_model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.ConvTranspose2d, torch.nn.AdaptiveAvgPool2d)):
                has_conv = True
                break
            if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                has_conv1d = True
            if isinstance(module, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN, torch.nn.RNNBase)):
                has_rnn = True
            if isinstance(module, torch.nn.Embedding):
                has_embed = True

        if has_conv:
            return "image"
        if (has_conv1d or has_rnn) and not has_conv and not has_embed:
            return "time-series"
        if has_embed and not has_conv:
            return "text"

    # 5. Check string identifier or Hugging Face config
    if isinstance(model_or_path, str):
        name_lower = model_or_path.lower().replace("_", "-")

        # Check Hugging Face repos dynamically first
        if "/" in model_or_path and not model_or_path.startswith((".", "/")):
            hf_info = detect_hf_repo_info(model_or_path)
            if hf_info.get("domain") and hf_info["domain"] != "unknown":
                return hf_info["domain"]

        text_keywords = (
            "bert", "roberta", "gpt", "llama", "t5", "bart", "electra", "distilbert",
            "albert", "deberta", "sentiment", "imdb", "nlp", "sequenceclassification",
            "causallm", "gliner", "decide", "token_classification", "tokenclassification",
            "question_answering", "summarization", "translation", "sentence_transformers",
            "embedding", "bge", "e5", "minilm", "extractor"
        )
        vision_keywords = (
            "vit", "resnet", "convnext", "swin", "deit", "vision", "cifar", "gtsrb",
            "mnist", "imagenet", "mobilenet", "efficientnet", "vgg", "clip_vision", "beit",
            "yolo", "densenet", "segmentation", "object_detection", "scenery", "intel", "traffic"
        )
        ts_keywords = (
            "patchtst", "patchtsmixer", "timesnet", "informer", "autoformer",
            "timeseries", "time-series", "chronos", "timesfm", "1dcnn", "cnn1d",
            "lstm", "gru", "ecg", "sensor", "telemetry", "temporal"
        )
        tabular_keywords = (
            "tabular", "tabnet", "tabtransformer", "fttransformer", "tabularmlp", "tabpfn",
            "credit", "fraud", "census", "adult", "churn", "housing", "titanic"
        )

        # Fast local name keyword matching first
        if any(k in name_lower for k in tabular_keywords):
            return "tabular"
        if any(k in name_lower for k in ts_keywords):
            return "time-series"
        if any(k in name_lower for k in text_keywords) or re.search(r"(^|[^a-z])(ner|qa|text)([^a-z]|$)", name_lower):
            return "text"
        if any(k in name_lower for k in vision_keywords):
            return "image"

        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_or_path, local_files_only=True)
            if cfg is not None:
                p_tag = getattr(cfg, "pipeline_tag", "") or getattr(cfg, "task", "")
                if p_tag and any(k in str(p_tag).lower() for k in ("tabular", "tabular-classification")):
                    return "tabular"
                if p_tag and any(k in str(p_tag).lower() for k in ("time-series", "forecasting")):
                    return "time-series"
                m_type = getattr(cfg, "model_type", "").lower()
                archs = [str(a).lower() for a in getattr(cfg, "architectures", [])]
                combined = m_type + " " + " ".join(archs)
                if any(k in combined for k in tabular_keywords):
                    return "tabular"
                if any(k in combined for k in ts_keywords):
                    return "time-series"
                if any(k in combined for k in text_keywords) or re.search(r"(^|[^a-z])(ner|qa|text)([^a-z]|$)", combined):
                    return "text"
                if any(k in combined for k in vision_keywords):
                    return "image"
        except Exception:
            pass

        # Check raw config.json or tokenizer / preprocessor existence on HF Hub
        try:
            from huggingface_hub import hf_hub_download
            try:
                cfg_path = hf_hub_download(model_or_path, "config.json")
                with open(cfg_path, "r", encoding="utf-8") as f:
                    raw_cfg = json.load(f)
                p_tag = raw_cfg.get("pipeline_tag")
                if p_tag and any(k in str(p_tag).lower() for k in ("tabular", "tabular-classification")):
                    return "tabular"
                if "sklearn" in raw_cfg or any(k in str(raw_cfg).lower() for k in ("tabular", "tabular-classification")):
                    return "tabular"
                if p_tag and any(k in str(p_tag).lower() for k in ("time-series", "forecasting")):
                    return "time-series"
                combined_cfg = " ".join(str(v).lower() for v in raw_cfg.values() if isinstance(v, (str, list)))
                if any(k in combined_cfg for k in ts_keywords):
                    return "time-series"
                if any(k in combined_cfg for k in text_keywords) or re.search(r"(^|[^a-z])(ner|qa|text)([^a-z]|$)", combined_cfg):
                    return "text"
                if any(k in combined_cfg for k in vision_keywords):
                    return "image"
            except Exception:
                pass

            # If tokenizer exists, it's text
            try:
                hf_hub_download(model_or_path, "tokenizer_config.json")
                return "text"
            except Exception:
                pass

            # If preprocessor exists, it's image
            try:
                hf_hub_download(model_or_path, "preprocessor_config.json")
                return "image"
            except Exception:
                pass
        except Exception:
            pass

    return "unknown"


def setup_model_environment(
    model_id_or_path: str,
    device: Optional[Union[str, torch.device]] = None,
    custom_dataset_name: Optional[str] = None,
    custom_processor_name: Optional[str] = None,
    data_domain: Optional[str] = None,
    log_log_event: bool = True,
    log_filepath: Union[str, Path] = "results/ingestion_log.json",
) -> Dict[str, Any]:
    """
    Configure and verify execution environment parameters for a selected model.
    Distinguishes model_id (what to load), dataset_name (what to evaluate),
    and processor_name (how to preprocess inputs).

    Args:
        model_id_or_path: Hugging Face hub ID, model preset ID, or local file path.
        device: Target execution device ('cuda', 'cpu', etc.).
        custom_dataset_name: Optional override dataset identifier.
        custom_processor_name: Optional override processor / tokenizer identifier.
        data_domain: Optional override data domain ('image', 'text', etc.).
        log_log_event: Whether to log this setup event to disk.
        log_filepath: Location of ingestion log file.

    Returns:
        Environment metadata dictionary.
    """
    from app.ingestion.model_registry import ModelPresetRegistry
    from app.utils.device_manager import DeviceManager

    target_device = DeviceManager.get_device(device)
    model_info = ModelPresetRegistry.resolve_model_info(model_id_or_path, data_domain=data_domain)

    dataset_name = custom_dataset_name or model_info.get("dataset_name", "bazyl/GTSRB")
    processor_name = custom_processor_name or model_info.get("processor_name", model_info["model_id"])

    resolution = model_info.get("input_resolution")
    expected_resolution = list(resolution) if resolution is not None else None

    env_config = {
        "model_id": model_info["model_id"],
        "model_name": model_info["name"],
        "processor_name": processor_name,
        "data_domain": model_info.get("data_domain", "unknown"),
        "model_type": model_info.get("model_type", "Model"),
        "domain": model_info.get("domain", "Unknown Domain"),
        "framework": model_info["framework"],
        "target_device": str(target_device),
        "cuda_available": torch.cuda.is_available(),
        "recommended_dataset": dataset_name,
        "task_type": model_info.get("task_type", "classification"),
        "expected_num_classes": model_info.get("num_classes"),
        "expected_input_resolution": expected_resolution,
        "sequence_length": model_info.get("sequence_length"),
        "num_channels": model_info.get("num_channels"),
        "num_features": model_info.get("num_features"),
        "status": "ENVIRONMENT_CONFIGURED",
    }

    if log_log_event:
        log_ingestion_event(env_config, log_filepath=log_filepath)

    return env_config


__all__ = [
    "is_forecasting_or_regression",
    "is_sklearn_model",
    "detect_model_source",
    "detect_model_framework",
    "inspect_model_evidence",
    "detect_model_domain",
    "setup_model_environment",
]
