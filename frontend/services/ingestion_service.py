"""
Model Ingestion Service for AdverScan Frontend.

Bridges the Streamlit UI to the existing backend ingestion engine (Module 1)
and evaluation dataset loader without duplicating backend logic.
Extracts comprehensive telemetry matching scripts/run_pipeline_for_multimodel.py.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import torch

from app.evaluation.dataset_loader.factory import (
    get_dataset_loader,
    resolve_dataset_file,
)
from app.ingestion.exceptions import (
    ConfigurationError,
    DatasetFormatError,
    ModelIngestionError,
    ModelLoadError,
    ModelValidationError,
    UnsafeModelError,
    UnsupportedDomainError,
    UnsupportedModelError,
    UnsupportedTaskError,
)
from app.ingestion.pipeline import IngestionResult, ingest_model
from app.ingestion.presets import KNOWN_DATASET_SAMPLE_COUNTS, MODEL_PRESETS
from app.ingestion.utils.model_utils import clear_hf_repo_cache


def get_available_presets() -> Dict[str, Dict[str, Any]]:
    """Retrieve existing model presets configured in the backend."""
    return MODEL_PRESETS


def inspect_model_summary(model_identifier: str) -> Dict[str, Any]:
    """
    Fast informational inspection of a model source (Hugging Face ID, local path, or preset)
    for frontend display prior to full ingestion.
    """
    clean_id = str(model_identifier or "").strip()
    if not clean_id:
        return {}

    # 1. Preset check
    if clean_id in MODEL_PRESETS:
        p = MODEL_PRESETS[clean_id]
        return {
            "model_id": clean_id,
            "framework": p.get("framework", "pytorch").upper(),
            "domain": p.get("data_domain", "image"),
            "architecture": p.get("model_type", "Preset Benchmark Model"),
            "task": p.get("task_type", "classification"),
            "format": "Built-in Preset",
            "features_count": p.get("num_features"),
            "num_classes": p.get("num_classes"),
            "artifacts": ["Built-in PyTorch weights", "Built-in Benchmark dataset"],
            "status": "Ready for Ingestion",
        }

    # 2. Local path
    p = Path(clean_id)
    if p.exists():
        from app.ingestion.utils.model_utils import detect_model_domain, detect_model_framework
        fw = detect_model_framework(clean_id)
        dom = detect_model_domain(clean_id)
        return {
            "model_id": clean_id,
            "framework": fw.upper(),
            "domain": dom,
            "architecture": f"{fw.capitalize()} Model Checkpoint",
            "task": "classification",
            "format": p.suffix or "Directory",
            "model_file": p.name,
            "requires_pickle_consent": p.suffix.lower() in (".joblib", ".pkl", ".pickle"),
            "features_count": None,
            "num_classes": None,
            "artifacts": [p.name],
            "status": "Local file resolved",
        }

    # 3. Hugging Face repository
    if "/" in clean_id and not clean_id.startswith((".", "/")):
        try:
            from app.ingestion.utils.model_utils import detect_hf_repo_info
            hf_info = detect_hf_repo_info(clean_id)
            fw = hf_info.get("framework", "unknown")
            dom = hf_info.get("domain", "unknown")
            fmt = hf_info.get("model_file") or "unknown"
            arch = hf_info.get("architecture") or f"{fw.capitalize()} Model"
            task = hf_info.get("task_type", "classification")
            feats = hf_info.get("num_features")
            raw_feats = len(hf_info.get("raw_feature_names") or [])
            arts = []
            if hf_info.get("model_file"):
                arts.append(hf_info["model_file"])
            arts.extend(hf_info.get("preprocessing_files", []))
            arts.extend(hf_info.get("calibration_files", []))
            arts.extend(hf_info.get("metadata_files", []))

            status_str = "Model weights and artifacts identified" if hf_info.get("is_model_repo") else "Inspecting repository..."
            return {
                "model_id": clean_id,
                "framework": fw.upper() if fw != "unknown" else "Auto-Detecting",
                "domain": dom,
                "architecture": arch,
                "task": task,
                "format": Path(fmt).suffix if fmt != "unknown" else "Repository",
                "model_file": hf_info.get("model_file"),
                "requires_pickle_consent": (
                    fw == "sklearn"
                    or any(
                        str(w).lower().endswith((".joblib", ".pkl", ".pickle"))
                        for w in (hf_info.get("weight_files") or ([fmt] if fmt != "unknown" else []))
                    )
                ),
                "features_count": feats or (raw_feats if raw_feats > 0 else None),
                "num_classes": None,
                "artifacts": arts or hf_info.get("files", [])[:5],
                "status": status_str,
            }
        except Exception:
            pass

    return {
        "model_id": clean_id,
        "framework": "Auto-Detecting",
        "domain": "unknown",
        "architecture": "Auto-Detecting",
        "task": "classification",
        "format": "Auto-Detecting",
        "features_count": None,
        "num_classes": None,
        "artifacts": [],
        "status": "Ready",
    }


def get_dataset_total_samples(
    dataset_name: Optional[str] = None,
    data_domain: Optional[str] = "image",
    model_identifier: Optional[str] = None,
) -> int:
    """
    Resolve total sample count available for a given dataset or model benchmark.
    Checks local files (CSV, PyTorch tensors), known benchmark datasets, and presets.
    """
    clean_ds = str(dataset_name or "").strip().lower()

    # 1. Exact match in known dataset sample counts
    if clean_ds and clean_ds in KNOWN_DATASET_SAMPLE_COUNTS:
        return int(KNOWN_DATASET_SAMPLE_COUNTS[clean_ds])

    # 2. Fuzzy or suffix match against known datasets (e.g. "gtsrb", "cifar10")
    if clean_ds:
        for k, v in KNOWN_DATASET_SAMPLE_COUNTS.items():
            if k.lower() == clean_ds or k.lower().endswith("/" + clean_ds) or clean_ds.endswith("/" + k.lower()):
                return int(v)

    # 3. Model preset lookup
    if model_identifier and model_identifier in MODEL_PRESETS:
        preset_ds = str(MODEL_PRESETS[model_identifier].get("dataset_name", "")).strip().lower()
        if preset_ds in KNOWN_DATASET_SAMPLE_COUNTS:
            return int(KNOWN_DATASET_SAMPLE_COUNTS[preset_ds])
        for k, v in KNOWN_DATASET_SAMPLE_COUNTS.items():
            if k.lower() == preset_ds or k.lower().endswith("/" + preset_ds) or preset_ds.endswith("/" + k.lower()):
                return int(v)

    # 4. Check local disk files (CSV, PT, PTH, Parquet)
    try:
        resolved = resolve_dataset_file(
            identifier=dataset_name,
            candidate_extensions=[".csv", ".pt", ".pth", ".parquet", ".npy"],
            model_identifier=model_identifier,
        )
        if resolved and Path(resolved).is_file():
            p = Path(resolved)
            if p.suffix == ".csv":
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    lines = sum(1 for _ in f) - 1
                if lines > 0:
                    return lines
            elif p.suffix in (".pt", ".pth"):
                obj = torch.load(p, weights_only=False, map_location="cpu")
                if isinstance(obj, dict):
                    for key in ("inputs", "data", "x", "samples", "features"):
                        if key in obj and hasattr(obj[key], "shape") and len(obj[key].shape) > 0:
                            return int(obj[key].shape[0])
                    for v in obj.values():
                        if hasattr(v, "shape") and len(v.shape) > 0:
                            return int(v.shape[0])
                elif hasattr(obj, "shape") and len(obj.shape) > 0:
                    return int(obj.shape[0])
                elif hasattr(obj, "__len__"):
                    return len(obj)
            elif p.suffix == ".parquet":
                df = pd.read_parquet(p)
                return len(df)
    except Exception:
        pass

    # 5. Fallback defaults by domain (only for legacy get_dataset_total_samples callers)
    clean_domain = str(data_domain or "").strip().lower()
    if clean_domain in ("tabular", "csv"):
        return 250
    elif clean_domain in ("time-series", "timeseries"):
        return 200
    elif clean_domain in ("text", "nlp"):
        return 872
    return 1000


def get_dataset_population_info(
    dataset_name: Optional[str] = None,
    data_domain: Optional[str] = "image",
    model_identifier: Optional[str] = None,
    uploaded_df: Optional[pd.DataFrame] = None,
    uploaded_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Resolve actual dataset population metadata from genuine sources only.
    Never invents a 250-sample fallback when no dataset is present.
    """
    if uploaded_df is not None:
        n_rows = int(len(uploaded_df))
        return {
            "total_samples": n_rows,
            "resolved": n_rows > 0,
            "source_type": "uploaded",
            "resolved_source": uploaded_name or "Uploaded Dataset",
            "provenance": f"User-uploaded dataset ({uploaded_name or 'DataFrame'}) • {n_rows:,} rows",
        }

    _model_exts = (".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".pkl", ".pickle", ".joblib", ".onnx", ".ubj", ".cbm", ".h5", ".hdf5", ".keras", ".tflite", ".xml", ".engine", ".plan", ".mlmodel")
    clean_ds = str(dataset_name or "").strip()
    clean_ds_lower = clean_ds.lower()
    is_placeholder = (
        not clean_ds
        or clean_ds_lower in ("custom_dataset", "evaluation-dataset", "none")
        or clean_ds_lower.endswith(_model_exts)
        or (model_identifier and clean_ds == str(model_identifier).strip())
    )

    # 1. Explicit dataset in KNOWN_DATASET_SAMPLE_COUNTS
    if not is_placeholder:
        if clean_ds_lower in KNOWN_DATASET_SAMPLE_COUNTS:
            n = int(KNOWN_DATASET_SAMPLE_COUNTS[clean_ds_lower])
            return {
                "total_samples": n,
                "resolved": True,
                "source_type": "preset",
                "resolved_source": clean_ds,
                "provenance": f"Benchmark dataset '{clean_ds}'",
            }
        for k, v in KNOWN_DATASET_SAMPLE_COUNTS.items():
            if k.lower() == clean_ds_lower or k.lower().endswith("/" + clean_ds_lower) or clean_ds_lower.endswith("/" + k.lower()):
                return {
                    "total_samples": int(v),
                    "resolved": True,
                    "source_type": "preset",
                    "resolved_source": clean_ds,
                    "provenance": f"Benchmark dataset '{clean_ds}'",
                }

    # 2. Model preset lookup
    if model_identifier and model_identifier in MODEL_PRESETS:
        preset_ds = str(MODEL_PRESETS[model_identifier].get("dataset_name", "")).strip()
        preset_ds_lower = preset_ds.lower()
        if preset_ds_lower in KNOWN_DATASET_SAMPLE_COUNTS:
            n = int(KNOWN_DATASET_SAMPLE_COUNTS[preset_ds_lower])
            return {
                "total_samples": n,
                "resolved": True,
                "source_type": "preset",
                "resolved_source": preset_ds,
                "provenance": f"Preset benchmark dataset '{preset_ds}'",
            }
        for k, v in KNOWN_DATASET_SAMPLE_COUNTS.items():
            if k.lower() == preset_ds_lower or k.lower().endswith("/" + preset_ds_lower) or preset_ds_lower.endswith("/" + k.lower()):
                return {
                    "total_samples": int(v),
                    "resolved": True,
                    "source_type": "preset",
                    "resolved_source": preset_ds,
                    "provenance": f"Preset benchmark dataset '{preset_ds}'",
                }

    # 3. Check local disk or HF repository bundled dataset
    try:
        resolved = None
        if not is_placeholder:
            resolved = resolve_dataset_file(
                identifier=clean_ds,
                candidate_extensions=[".csv", ".tsv", ".parquet", ".pt", ".pth", ".npy", ".jsonl"],
                model_identifier=None,
            )
        elif str(data_domain or "").strip().lower() in ("tabular", "csv", "table") and model_identifier:
            from app.evaluation.dataset_loader.factory import resolve_tabular_dataset_source
            resolved = resolve_tabular_dataset_source(
                dataset_name=None if is_placeholder else clean_ds,
                model_id=str(model_identifier).strip(),
            )

        if isinstance(resolved, pd.DataFrame):
            n = int(len(resolved))
            return {
                "total_samples": n,
                "resolved": n > 0,
                "source_type": "hf_dataset",
                "resolved_source": clean_ds if not is_placeholder else str(model_identifier),
                "provenance": f"Resolved DataFrame ({n:,} rows)",
            }
        if resolved and Path(str(resolved)).is_file():
            p = Path(str(resolved))
            n = 0
            if p.suffix.lower() in (".csv", ".tsv"):
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    n = max(0, sum(1 for _ in f) - 1)
            elif p.suffix.lower() == ".parquet":
                n = int(len(pd.read_parquet(p)))
            elif p.suffix.lower() in (".pt", ".pth"):
                obj = torch.load(p, weights_only=False, map_location="cpu")
                if isinstance(obj, dict):
                    for key in ("inputs", "data", "x", "samples", "features"):
                        if key in obj and hasattr(obj[key], "shape") and len(obj[key].shape) > 0:
                            n = int(obj[key].shape[0])
                            break
                elif hasattr(obj, "shape") and len(obj.shape) > 0:
                    n = int(obj.shape[0])
            if n > 0:
                return {
                    "total_samples": n,
                    "resolved": True,
                    "source_type": "local_file",
                    "resolved_source": str(p),
                    "provenance": f"Local file '{p.name}' ({n:,} rows)",
                }
    except Exception:
        pass

    return {
        "total_samples": 0,
        "resolved": False,
        "source_type": "none",
        "resolved_source": None,
        "provenance": "No evaluation dataset resolved",
    }


def load_model_and_dataset(
    model_identifier: Any,
    dataset_name: Optional[str] = None,
    data_domain: Optional[str] = None,
    framework: Optional[str] = None,
    device: Optional[str] = None,
    sample_size: int = 10,
    unsafe_allow_pickle: bool = False,
    force_refresh: bool = True,
    dataset_data: Optional[Any] = None,
    allow_synthetic_fallback: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Ingest a model using the existing backend pipeline, extract complete telemetry,
    and prepare an evaluation sample batch.

    Returns:
        Structured dictionary matching the telemetry in run_pipeline_for_multimodel.py:
            - "adapter": The BaseModelAdapter wrapping the loaded model.
            - "raw_model": The underlying model instance.
            - "metadata": ModelMetadata dataclass instance.
            - "telemetry": Comprehensive model & architecture telemetry dictionary.
            - "test_inputs": Sample inputs tensor for attack execution.
            - "test_labels": Sample labels tensor for attack execution.
            - "dataset_name": Resolved dataset name used.
            - "domain": Resolved data domain.
            - "ingestion_time": Exact duration in seconds.
    """
    if model_identifier is None or (
        isinstance(model_identifier, str) and not model_identifier.strip()
    ):
        raise ConfigurationError("Model identifier or path must not be empty.")

    clean_id = (
        model_identifier.strip()
        if isinstance(model_identifier, str)
        else getattr(model_identifier, "__name__", type(model_identifier).__name__)
    )

    # Determine execution device and framework
    exec_device = device if device and device != "Auto-detect" else None
    exec_framework = (
        None
        if (not framework or str(framework).lower().strip() in ("auto", "auto-detect", "auto-detect (recommended)", "none"))
        else framework
    )

    # Clear cached HF repo metadata on fresh ingestion requests so retries/switches never use stale metadata
    if force_refresh and isinstance(model_identifier, str):
        clear_hf_repo_cache(clean_id)

    # Step 1: Ingest model using existing backend ingestion pipeline and measure time
    t0 = time.time()
    try:
        ingestion_res: IngestionResult = ingest_model(
            model_path=model_identifier,
            dataset_name=dataset_name,
            data_domain=data_domain,
            framework=exec_framework,
            device=exec_device,
            unsafe_allow_pickle=bool(unsafe_allow_pickle),
        )
    except (
        ModelLoadError,
        ModelValidationError,
        ConfigurationError,
        DatasetFormatError,
        UnsafeModelError,
        UnsupportedDomainError,
        UnsupportedModelError,
        UnsupportedTaskError,
    ) as known_err:
        raise known_err
    except Exception as exc:
        raise ModelIngestionError(
            f"Failed to ingest model '{clean_id}': {str(exc)}"
        ) from exc

    ingestion_time = time.time() - t0

    adapter = ingestion_res.adapter
    metadata = ingestion_res.metadata
    raw_model = getattr(adapter, "get_model", lambda: adapter)()
    preset_info = MODEL_PRESETS.get(clean_id, {})
    model_name = (
        preset_info.get("name") or getattr(metadata, "model_name", None) or clean_id
    )

    # Step 2: Compute parameter counts & estimated memory size from actual tensors
    total_params = None
    trainable_params = None
    trainable_pct = 100.0
    param_bytes = 0
    buffer_bytes = 0

    target_for_params = raw_model if hasattr(raw_model, "parameters") else adapter
    if hasattr(target_for_params, "parameters") and callable(
        getattr(target_for_params, "parameters")
    ):
        try:
            params_list = list(target_for_params.parameters())
            if params_list:
                total_params = sum(p.numel() for p in params_list)
                trainable_params = sum(
                    p.numel() for p in params_list if getattr(p, "requires_grad", False)
                )
                param_bytes = sum(
                    p.numel() * p.element_size()
                    for p in params_list
                    if hasattr(p, "element_size")
                )
                if total_params > 0:
                    trainable_pct = (trainable_params / total_params) * 100.0
        except Exception:
            pass

    if hasattr(target_for_params, "buffers") and callable(
        getattr(target_for_params, "buffers")
    ):
        try:
            buffers_list = list(target_for_params.buffers())
            if buffers_list:
                buffer_bytes = sum(
                    b.numel() * b.element_size()
                    for b in buffers_list
                    if hasattr(b, "element_size")
                )
        except Exception:
            pass

    if (total_params is None or total_params == 0) and preset_info.get("parameters"):
        total_params = preset_info.get("parameters")
        trainable_params = total_params

    # Calculate actual stored parameter and buffer memory
    if param_bytes + buffer_bytes > 0:
        estimated_size_mb = (param_bytes + buffer_bytes) / (1024.0 * 1024.0)
    elif total_params is not None and total_params > 0:
        estimated_size_mb = (total_params * 4.0) / (1024.0 * 1024.0)
    else:
        estimated_size_mb = None

    # Architecture discovery: check preset, config architectures, and class hierarchy
    arch_name = getattr(metadata, "model_architecture", None) or preset_info.get(
        "model_type"
    )
    if not arch_name and hasattr(raw_model, "config"):
        cfg = getattr(raw_model, "config")
        architectures = getattr(cfg, "architectures", None)
        if architectures and isinstance(architectures, list) and architectures[0]:
            arch_name = str(architectures[0])
        elif getattr(cfg, "model_type", None):
            arch_name = f"{str(cfg.model_type).capitalize()} Architecture"

    if not arch_name and raw_model is not None:
        cls_name = getattr(raw_model, "__class__", type(raw_model)).__name__
        if cls_name not in ("NoneType", "object", "BaseModelAdapter"):
            arch_name = cls_name

    if not arch_name:
        arch_name = "Unknown"

    # Step 3: Determine resolved domain and dataset
    resolved_domain = (
        (metadata.data_domain or data_domain or "image").lower().replace("_", "-")
    )

    # Execution device
    if hasattr(adapter, "device"):
        device_name = str(adapter.device).upper()
    elif exec_device:
        device_name = str(exec_device).upper()
    elif torch.cuda.is_available():
        device_name = "CUDA (GPU)"
    else:
        device_name = "CPU"

    # Step 4: Load a sample evaluation batch using existing dataset loader factory
    # NEVER treat model_identifier / model filename as a dataset source.
    test_inputs: Optional[torch.Tensor] = None
    test_labels: Optional[torch.Tensor] = None
    is_synthetic_dataset = False
    dataset_resolved = False
    fallback_reason: Optional[str] = None
    explicit_dataset_requested = bool(
        dataset_data is not None
        or (isinstance(dataset_name, str) and dataset_name.strip() and dataset_name.strip().lower() not in ("custom_dataset", "evaluation-dataset", "none"))
    )

    meta_ds = getattr(metadata, "dataset_name", None)
    if meta_ds and str(meta_ds).strip().lower() in ("custom_dataset", "evaluation-dataset", "none", ""):
        meta_ds = None
    resolved_dataset = (
        dataset_name.strip()
        if isinstance(dataset_name, str) and dataset_name.strip()
        else None
    ) or meta_ds

    # Extract model feature schema & preprocessing metadata for compatible tabular dataset resolution
    model_feature_names = (
        getattr(adapter, "feature_names", None)
        or getattr(raw_model, "feature_names", None)
        or (getattr(metadata, "extra_info", {}) or {}).get("feature_names")
    )
    model_raw_features = (
        getattr(adapter, "raw_input_features", None)
        or getattr(raw_model, "raw_input_features", None)
        or (getattr(metadata, "extra_info", {}) or {}).get("raw_feature_names")
    )
    model_preprocessing = (
        getattr(adapter, "preprocessing_info", None)
        or getattr(raw_model, "preprocessing_info", None)
    )
    expected_num_features = None
    if model_feature_names:
        expected_num_features = len(model_feature_names)
    elif hasattr(raw_model, "num_features") and callable(raw_model.num_features):
        expected_num_features = int(raw_model.num_features())
    elif hasattr(raw_model, "n_features_in_"):
        expected_num_features = int(getattr(raw_model, "n_features_in_"))
    elif getattr(metadata, "input_shape", None) and len(metadata.input_shape) >= 1:
        expected_num_features = int(metadata.input_shape[-1])

    loader = None
    try:
        if dataset_data is not None or resolved_dataset is not None or resolved_domain in ("tabular", "csv", "table"):
            loader = get_dataset_loader(
                dataset_name=resolved_dataset or "custom_tabular",
                data_domain=resolved_domain,
                processor_name=metadata.processor_name or clean_id,
                split=f"test[:{sample_size}]",
                batch_size=sample_size,
                data=dataset_data,
                model_id=clean_id,
                feature_names=model_feature_names,
                raw_input_features=model_raw_features,
                preprocessing_info=model_preprocessing,
                expected_num_features=expected_num_features,
            )

            for batch in loader.iterate_batches():
                test_inputs = batch[0]
                test_labels = batch[1]
                break

            if test_inputs is not None and test_labels is not None:
                # Verify dataset compatibility with the model's prediction interface
                try:
                    sample_probe = _slice_sample_probe(test_inputs, 1)
                    _ = adapter.predict(sample_probe)
                except Exception as compat_exc:
                    raise DatasetFormatError(
                        f"Loaded dataset is incompatible with model prediction interface: {compat_exc}"
                    ) from compat_exc
                dataset_resolved = True
        else:
            fallback_reason = (
                f"No evaluation dataset is associated with model '{clean_id}'. "
                f"Please select or upload a compatible evaluation dataset."
            )
    except Exception as exc:
        fallback_reason = str(exc)
        # Case D: If user explicitly provided a dataset/file and it is incompatible or missing, fail cleanly (no synthetic fallback)
        if explicit_dataset_requested and allow_synthetic_fallback is False:
            if isinstance(exc, (DatasetFormatError, ValueError, ConfigurationError)):
                raise DatasetFormatError(f"Dataset validation failed: {exc}") from exc
            raise DatasetFormatError(f"Could not load specified dataset '{resolved_dataset}': {exc}") from exc

    if not dataset_resolved:
        if allow_synthetic_fallback is False:
            # Case B: Keep model ingestion successful; do NOT silently generate synthetic evaluation data
            test_inputs = None
            test_labels = None
            is_synthetic_dataset = False
        else:
            # Explicit synthetic fallback (Case E) or legacy caller compatibility
            if not fallback_reason:
                fallback_reason = "No real evaluation dataset resolved"
            test_inputs, test_labels = _generate_domain_sample_batch(
                metadata, sample_size, raw_model=raw_model, preset_info=preset_info
            )
            is_synthetic_dataset = True

    actual_evaluated_samples = (
        _get_input_sample_count(test_inputs, default=sample_size)
        if test_inputs is not None
        else 0
    )

    # Class resolution
    num_classes = getattr(metadata, "num_classes", None) or preset_info.get(
        "num_classes"
    )
    if num_classes is None and hasattr(raw_model, "config"):
        cfg = getattr(raw_model, "config")
        num_labels = getattr(cfg, "num_labels", None)
        if num_labels is not None:
            num_classes = int(num_labels)
        elif getattr(cfg, "id2label", None) and isinstance(cfg.id2label, dict):
            num_classes = len(cfg.id2label)
    if num_classes is None and hasattr(raw_model, "num_classes"):
        num_classes = getattr(raw_model, "num_classes")

    # Capabilities & gradient access
    supports_gradients = getattr(
        getattr(adapter, "capabilities", None), "supports_gradients", True
    )
    gradient_access = (
        "Full Backprop (White-Box)"
        if supports_gradients
        else "Black-Box (Inference Only)"
    )

    if is_synthetic_dataset:
        dataset_display_name = f"Synthetic Fallback Data ({resolved_dataset or clean_id} unavailable)"
    elif dataset_resolved:
        dataset_display_name = resolved_dataset or (getattr(loader, "dataset_name", None) or "Resolved Dataset")
    else:
        dataset_display_name = "No Evaluation Dataset Loaded"

    # Observed input tensor shape
    observed_input_shape, batch_shape_str = _get_input_shape_desc(test_inputs)
    if observed_input_shape is None:
        observed_input_shape = (
            preset_info.get("input_resolution") or metadata.input_shape or None
        )

    if dataset_resolved and loader is not None and hasattr(loader, "__len__"):
        try:
            total_dataset_samples = int(len(loader))
        except Exception:
            total_dataset_samples = actual_evaluated_samples
    elif dataset_resolved:
        pop_info = get_dataset_population_info(
            dataset_name=resolved_dataset,
            data_domain=resolved_domain,
            model_identifier=clean_id,
            uploaded_df=dataset_data if isinstance(dataset_data, pd.DataFrame) else None,
        )
        total_dataset_samples = int(pop_info["total_samples"]) if pop_info["resolved"] else actual_evaluated_samples
    elif is_synthetic_dataset:
        total_dataset_samples = actual_evaluated_samples
    else:
        total_dataset_samples = 0

    # Build rich telemetry contract (aligned with run_pipeline_for_multimodel.py)
    fitted_prep = getattr(loader, "fitted_preprocessor", None) if loader is not None else None
    if fitted_prep is not None and adapter is not None:
        try:
            adapter.tabular_preprocessor = fitted_prep
        except Exception:
            pass
    orig_feat_count = getattr(loader, "original_feature_count", None) if loader is not None else None
    proc_feat_count = getattr(loader, "processed_feature_count", None) if loader is not None else None
    prep_summary = getattr(loader, "preprocessing_summary", None) if loader is not None else None

    telemetry = {
        "model_identifier": clean_id,
        "model_name": model_name,
        "architecture": arch_name,
        "framework": str(metadata.framework or "PyTorch").upper(),
        "domain": str(resolved_domain).capitalize(),
        "task_type": str(
            metadata.task_type or preset_info.get("task_type") or "Classification"
        ).capitalize(),
        "num_classes": num_classes if num_classes is not None else "Not available",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_pct": trainable_pct,
        "estimated_size_mb": estimated_size_mb,
        "ingestion_time": ingestion_time,
        "device": device_name,
        "dataset_name": dataset_display_name,
        "dataset_resolved": dataset_resolved,
        "total_dataset_samples": total_dataset_samples,
        "is_synthetic_dataset": is_synthetic_dataset,
        "dataset_source": (
            "synthetic_fallback" if is_synthetic_dataset
            else ("real_dataset" if dataset_resolved else "none")
        ),
        "dataset_fallback_reason": fallback_reason,
        "input_shape": (
            str(observed_input_shape)
            if observed_input_shape is not None
            else "Not available"
        ),
        "batch_shape": batch_shape_str,
        "sample_count": actual_evaluated_samples,
        "original_feature_count": orig_feat_count,
        "processed_feature_count": proc_feat_count,
        "preprocessing_summary": prep_summary,
        "gradient_access": gradient_access,
        "supports_gradients": supports_gradients,
    }

    metadata_dict = {
        "Model Name": model_name,
        "Architecture": arch_name,
        "Domain": str(resolved_domain).capitalize(),
        "Framework": str(getattr(metadata, "framework", None) or "PyTorch").upper(),
        "Dataset": dataset_display_name,
        "Total Dataset Samples": f"{total_dataset_samples:,}",
        "Dataset Source": "Synthetic Fallback (Generated)" if is_synthetic_dataset else "Real Dataset",
        "Task Type": str(
            getattr(metadata, "task_type", None)
            or preset_info.get("task_type")
            or "Classification"
        ).capitalize(),
        "Input Shape": telemetry["input_shape"],
        "Total Classes": str(telemetry["num_classes"]),
        "Device": device_name,
        "Gradient Access": gradient_access,
    }

    return {
        "model_identifier": clean_id,
        "adapter": adapter,
        "raw_model": raw_model,
        "model": adapter or raw_model,
        "metadata": metadata,
        "metadata_dict": metadata_dict,
        "telemetry": telemetry,
        "is_synthetic_dataset": is_synthetic_dataset,
        "dataset_resolved": dataset_resolved,
        "dataset_source": (
            "synthetic_fallback" if is_synthetic_dataset
            else ("real_dataset" if dataset_resolved else "none")
        ),
        "dataset_fallback_reason": fallback_reason,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_pct": trainable_pct,
        "estimated_size_mb": estimated_size_mb,
        "model_architecture": arch_name,
        "num_classes": metadata_dict["Total Classes"],
        "input_shape": metadata_dict["Input Shape"],
        "batch_tensor_shape": batch_shape_str,
        "original_feature_count": orig_feat_count,
        "processed_feature_count": proc_feat_count,
        "preprocessing_summary": prep_summary,
        "tabular_preprocessor": fitted_prep,
        "sample_size": actual_evaluated_samples,
        "total_dataset_samples": total_dataset_samples,
        "device": device_name,
        "test_inputs": test_inputs,
        "test_labels": test_labels,
        "dataset_name": dataset_display_name,
        "domain": resolved_domain,
        "ingestion_time": ingestion_time,
        "gradient_access": gradient_access,
    }


def _slice_sample_probe(inputs: Any, count: int = 1) -> Any:
    """Safely extract a small probe batch across tensors, lists, dicts, and HF BatchEncoding mappings."""
    if isinstance(inputs, Mapping):
        return {
            k: (
                v[:count]
                if hasattr(v, "__getitem__") and not isinstance(v, (str, bytes))
                else v
            )
            for k, v in inputs.items()
        }
    if isinstance(inputs, (str, bytes)):
        return inputs
    if hasattr(inputs, "__getitem__"):
        return inputs[:count]
    return inputs


def _get_input_sample_count(inputs: Any, default: int = 1) -> int:
    """Safely count number of evaluation samples across tensors, dicts, BatchEncodings, and lists."""
    if inputs is None:
        return default
    if hasattr(inputs, "shape") and len(inputs.shape) > 0:
        return int(inputs.shape[0])
    if isinstance(inputs, Mapping):
        for k in ("input_ids", "inputs", "pixel_values", "values"):
            if k in inputs and hasattr(inputs[k], "shape") and len(inputs[k].shape) > 0:
                return int(inputs[k].shape[0])
        for v in inputs.values():
            if hasattr(v, "__len__"):
                return len(v)
    if hasattr(inputs, "__len__"):
        return len(inputs)
    return default


def _get_input_shape_desc(inputs: Any) -> Tuple[Optional[List[int]], str]:
    """Safely describe input and batch shape across arbitrary tensor, dict, and text structures."""
    if inputs is None:
        return None, "Not available"
    if hasattr(inputs, "shape"):
        try:
            return list(inputs.shape[1:]), str(list(inputs.shape))
        except Exception:
            pass
    if isinstance(inputs, Mapping):
        for k in ("input_ids", "inputs", "pixel_values", "values"):
            if k in inputs and hasattr(inputs[k], "shape"):
                return list(inputs[k].shape[1:]), f"Dict[{k}: {list(inputs[k].shape)}]"
        return None, f"Dict[{', '.join(inputs.keys())}]"
    if hasattr(inputs, "__len__"):
        return None, f"({len(inputs)}, ...)"
    return None, "Not available"


def _generate_domain_sample_batch(
    metadata: Any,
    sample_size: int = 10,
    raw_model: Optional[Any] = None,
    preset_info: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any]:
    """Generate a sample input tensor conforming to the model's expected shape and domain."""
    domain = str(getattr(metadata, "data_domain", "image")).lower().replace("_", "-")
    num_classes = getattr(metadata, "num_classes", 2) or 2

    if domain in ("image", "vision"):
        inputs = torch.rand(sample_size, 3, 224, 224)
        labels = torch.randint(0, num_classes, (sample_size,))
    elif domain in ("tabular", "csv"):
        # Discover num_features dynamically
        num_features = None
        if raw_model is not None:
            if hasattr(raw_model, "n_features_in_"):
                num_features = getattr(raw_model, "n_features_in_")
            elif hasattr(raw_model, "feature_names_in_"):
                num_features = len(getattr(raw_model, "feature_names_in_"))
            elif hasattr(raw_model, "feature_names") and raw_model.feature_names:
                num_features = len(raw_model.feature_names)
            elif hasattr(raw_model, "num_features") and callable(raw_model.num_features):
                num_features = raw_model.num_features()
            elif hasattr(raw_model, "n_features_"):
                num_features = getattr(raw_model, "n_features_")
            elif hasattr(raw_model, "named_parameters"):
                # Inspect first linear layer weight in PyTorch
                for p_name, param in raw_model.named_parameters():
                    if "weight" in p_name and len(param.shape) == 2:
                        num_features = param.shape[1]
                        break

        if num_features is None and metadata is not None:
            num_features = getattr(metadata, "num_features", None)
            if num_features is None and getattr(metadata, "input_shape", None):
                in_shape = metadata.input_shape
                if len(in_shape) >= 1:
                    num_features = in_shape[-1]
            if num_features is None:
                extra = getattr(metadata, "extra_info", {}) or {}
                num_features = extra.get("num_features")
                if num_features is None and extra.get("feature_names"):
                    num_features = len(extra["feature_names"])
                elif num_features is None and extra.get("raw_feature_names"):
                    num_features = len(extra["raw_feature_names"])

        if num_features is None and preset_info:
            num_features = preset_info.get("num_features")

        if num_features is None:
            raise ConfigurationError(
                f"Cannot determine number of tabular features for model '{getattr(metadata, 'model_id', 'unknown')}'. "
                f"Model has no 'n_features_in_' attribute and no 'num_features' configuration was provided. "
                f"Please specify 'num_features' explicitly in configuration."
            )

        num_features = int(num_features)
        if num_classes is None or num_classes <= 0:
            num_classes = 2
        inputs = torch.randn(sample_size, num_features)
        labels = torch.randint(0, num_classes, (sample_size,))
    elif domain in ("time-series", "timeseries", "sequence"):
        # Correctly format time-series tensors as (batch_size, sequence_length, num_channels)
        cfg = getattr(raw_model, "config", None) if raw_model is not None else None
        seq_len = (
            getattr(cfg, "context_length", None)
            or getattr(cfg, "sequence_length", None)
            or getattr(cfg, "seq_length", None)
            or getattr(cfg, "max_position_embeddings", None)
        )
        num_channels = (
            getattr(cfg, "num_input_channels", None)
            or getattr(cfg, "num_channels", None)
            or getattr(cfg, "input_size", None)
        )

        if seq_len is None and preset_info:
            seq_len = preset_info.get("sequence_length") or preset_info.get("context_length")
        if num_channels is None and preset_info:
            num_channels = preset_info.get("num_channels") or preset_info.get("num_input_channels")

        if seq_len is None and metadata is not None:
            extra = getattr(metadata, "extra_info", {}) or {}
            seq_len = getattr(metadata, "sequence_length", None) or extra.get("sequence_length") or extra.get("context_length")
            if seq_len is None and getattr(metadata, "input_shape", None):
                in_shape = metadata.input_shape
                if len(in_shape) >= 2:
                    seq_len = in_shape[0] if in_shape[0] > in_shape[1] else in_shape[1]

        if num_channels is None and metadata is not None:
            extra = getattr(metadata, "extra_info", {}) or {}
            num_channels = getattr(metadata, "num_channels", None) or extra.get("num_channels") or getattr(metadata, "num_features", None)

        # PyTorch model layer inspection if still None
        if num_channels is None and raw_model is not None and hasattr(raw_model, "named_parameters"):
            for p_name, param in raw_model.named_parameters():
                if ("conv" in p_name or "fc" in p_name or "linear" in p_name) and "weight" in p_name:
                    if len(param.shape) >= 2:
                        num_channels = param.shape[1]
                        break

        if seq_len is None or num_channels is None:
            raise ConfigurationError(
                f"Cannot determine time-series input dimensions for model '{getattr(metadata, 'model_id', 'unknown')}'. "
                f"Discovered seq_len={seq_len}, num_channels={num_channels}. "
                f"Please provide 'sequence_length' and 'num_channels' in configuration."
            )

        seq_len = int(seq_len)
        num_channels = int(num_channels)

        inputs = torch.randn(sample_size, seq_len, num_channels)
        labels = torch.randint(0, num_classes, (sample_size,))
    elif domain in ("text", "nlp"):
        # Generate tokenized text representation for text models
        cfg = getattr(raw_model, "config", None) if raw_model is not None else None
        seq_len = getattr(cfg, "max_position_embeddings", None) or getattr(metadata, "max_length", 32) or 32
        seq_len = min(int(seq_len), 128)
        input_ids = torch.randint(100, 2000, (sample_size, seq_len))
        attention_mask = torch.ones((sample_size, seq_len), dtype=torch.long)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        labels = torch.randint(0, num_classes, (sample_size,))
    else:
        inputs = torch.rand(sample_size, 3, 224, 224)
        labels = torch.randint(0, num_classes, (sample_size,))

    return inputs, labels
