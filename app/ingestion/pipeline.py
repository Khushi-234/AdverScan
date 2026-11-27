"""
Ingestion pipeline logic for AdverScan framework.
Standardizes model loading, adaptation, preprocessing, and validation
for predefined and user-provided custom models across image and text domains.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from app.ingestion.adapters.base_adapter import BaseModelAdapter
from app.ingestion.adapters.deployment_adapter import DeploymentRuntimeAdapter
from app.ingestion.adapters.huggingface_adapter import HuggingFaceAdapter
from app.ingestion.adapters.keras_adapter import KerasAdapter
from app.ingestion.adapters.onnx_adapter import ONNXAdapter
from app.ingestion.adapters.pytorch_adapter import PyTorchAdapter
from app.ingestion.adapters.sklearn_adapter import SklearnAdapter
from app.ingestion.adapters.tree_adapter import TreeModelAdapter
from app.ingestion.exceptions import (
    ModelLoadError,
    UnsupportedDomainError,
    UnsupportedModelError,
    UnsupportedTaskError,
)
from app.ingestion.loader.base_loader import BaseModelLoader
from app.ingestion.loader.huggingface_loader import HuggingFaceLoader, load_keras_or_savedmodel
from app.ingestion.loader.pytorch_loader import PyTorchLoader
from app.ingestion.loader.sklearn_loader import SklearnLoader
from app.ingestion.metadata.model_metadata import ModelMetadata
from app.ingestion.metadata.tabular_metadata import TabularMetadata
from app.ingestion.model_registry import ModelPresetRegistry
from app.ingestion.processors.base_processor import BaseProcessor
from app.ingestion.processors.image_processor import ImageProcessor
from app.ingestion.processors.tabular_processor import TabularProcessor
from app.ingestion.processors.text_processor import TextProcessor
from app.ingestion.processors.time_series_processor import TimeSeriesProcessor
from app.ingestion.utils.device_manager import DeviceManager
from app.ingestion.utils.logger import log_ingestion_event
from app.ingestion.validation.validator import ModelValidator


class IngestionResult(tuple):
    """
    Standardized model ingestion result container.
    Inherits from tuple for full backward compatibility:
        adapter, metadata = ingest_model(...)
    Also exposes attributes:
        result.adapter
        result.metadata
        result.processor
    """

    def __new__(
        cls,
        adapter: BaseModelAdapter,
        metadata: ModelMetadata,
        processor: Optional[BaseProcessor] = None,
    ):
        return super().__new__(cls, (adapter, metadata))

    def __init__(
        self,
        adapter: BaseModelAdapter,
        metadata: ModelMetadata,
        processor: Optional[BaseProcessor] = None,
    ):
        self.adapter = adapter
        self.metadata = metadata
        self.processor = processor

    def __repr__(self) -> str:
        return (
            f"IngestionResult(adapter={self.adapter.__class__.__name__}, "
            f"model={self.metadata.model_name}, framework={self.metadata.framework}, "
            f"domain={self.metadata.data_domain})"
        )


def ingest_model(
    model_path: Union[str, Path, Any],
    sample_input: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
    model_class: Optional[Any] = None,
    model_name: Optional[str] = None,
    task_type: Optional[str] = None,
    domain: Optional[str] = None,
    data_domain: Optional[str] = None,
    framework: Optional[str] = None,
    dataset_name: Optional[str] = None,
    processor_name: Optional[str] = None,
    log_ingestion: bool = False,
    log_filepath: Union[str, Path] = "results/ingestion_log.json",
    **kwargs: Any
) -> IngestionResult:
    """
    Standard ingestion pipeline for loading, adapting, preprocessing, validating,
    and generating metadata for predefined and custom models.

    Flow:
        1. User Model Input
        2. Registry lookup if model exists; Otherwise create runtime configuration from user-provided details
        3. Validate configuration
        4. Select loader using framework
        5. Load model dynamically
        6. Select processor using data_domain
        7. Select adapter using data_domain/framework
        8. Return standardized ingestion result

    Args:
        model_path: Preset identifier, HF Hub repo ID, local path, or existing model object.
        sample_input: Optional sample input for validation and shape inference.
        device: Target execution device ('cuda', 'cpu', etc.).
        model_class: Model class factory or structure when loading state_dict.
        model_name: Optional custom model name.
        task_type: Machine learning task type (default 'classification').
        domain: Legacy alias for data_domain ('image' or 'text').
        data_domain: Target data domain ('image' or 'text').
        framework: Model framework ('huggingface' or 'pytorch').
        dataset_name: Recommended or custom dataset identifier.
        processor_name: Custom or preset tokenizer/image processor identifier.
        log_ingestion: Whether to append ingestion event to structured JSON log.
        log_filepath: File path for ingestion log.
        **kwargs: Additional parameters passed to loaders or configuration.

    Returns:
        IngestionResult: Standardized ingestion result containing adapter, metadata, and processor.
    """
    # 1. Device selection
    target_device = DeviceManager.get_device(device)
    resolved_domain = data_domain or domain
    if resolved_domain:
        resolved_domain = resolved_domain.lower().strip().replace("_", "-")
        if resolved_domain == "timeseries":
            resolved_domain = "time-series"
    resolved_task = task_type or "classification"

    if not resolved_domain:
        from app.ingestion.model_registry import detect_model_domain
        resolved_domain = detect_model_domain(
            model_or_path=model_path,
            sample_input=sample_input,
            processor_name=processor_name,
            model_class=model_class,
        )
    resolved_domain = resolved_domain or "unknown"
    if resolved_domain.lower().strip().replace("_", "-") in ("timeseries", "time-series"):
        resolved_domain = "time-series"

    from app.ingestion.model_registry import is_forecasting_or_regression
    if is_forecasting_or_regression(model_path, task_type=resolved_task):
        raise UnsupportedTaskError(
            "AdverScan currently supports time-series classification models only; forecasting models are not supported."
        )

    # 2. Registry lookup & runtime configuration creation
    from app.ingestion.model_registry import (
        detect_model_framework,
        detect_model_source,
        inspect_model_evidence,
        is_sklearn_model,
    )
    detected_source = detect_model_source(model_path)
    detected_fw = framework or detect_model_framework(model_path, model_class=model_class)

    config_dict: Dict[str, Any] = {}
    is_in_memory_model = isinstance(model_path, nn.Module) or not isinstance(model_path, (str, Path))

    if is_in_memory_model:
        # In-memory PyTorch / scikit-learn / custom model instance
        resolved_framework = framework or detected_fw
        resolved_dom = resolved_domain
        derived_name = model_name or model_path.__class__.__name__
        config_dict = {
            "model_id": derived_name,
            "name": derived_name,
            "framework": resolved_framework,
            "data_domain": resolved_dom,
            "domain": resolved_dom,
            "task_type": resolved_task,
            "dataset_name": dataset_name or "custom_dataset",
            "processor_name": processor_name,
            **kwargs,
        }
    else:
        # String or Path input
        model_id_str = str(model_path).strip()
        config_dict = ModelPresetRegistry.build_runtime_config(
            model_id=model_id_str,
            dataset_name=dataset_name,
            data_domain=resolved_domain if resolved_domain != "unknown" else None,
            task_type=resolved_task,
            framework=framework,
            processor_name=processor_name,
            **kwargs,
        )
        if not config_dict.get("framework"):
            config_dict["framework"] = detected_fw
        if config_dict.get("data_domain") is None:
            config_dict["data_domain"] = resolved_domain
            config_dict["domain"] = resolved_domain

    # Normalize both registry preset and custom user configuration into ModelMetadata
    runtime_metadata = ModelMetadata.from_dict(config_dict)
    runtime_metadata.device = str(target_device)
    if model_name:
        runtime_metadata.model_name = model_name

    # 3. Validate configuration prior to loading
    ModelValidator.validate_configuration(runtime_metadata)

    # 4 & 5. Select loader using framework and load model dynamically
    framework_clean = runtime_metadata.framework.lower()
    raw_model: Any = None
    target_model_path = model_path
    if not is_in_memory_model and config_dict.get("model_id"):
        cand_path = str(config_dict.get("model_id"))
        if Path(cand_path).exists() or "/" in cand_path:
            target_model_path = cand_path

    if framework_clean == "sklearn":
        loader: BaseModelLoader = SklearnLoader()
        allow_pickle = kwargs.pop("unsafe_allow_pickle", False)
        raw_model = loader.load(
            target_model_path,
            unsafe_allow_pickle=allow_pickle,
            **kwargs,
        )
    elif framework_clean in ("keras", "tensorflow"):
        try:
            from huggingface_hub import snapshot_download
            p = Path(target_model_path)
            local_dir = str(p) if p.exists() else snapshot_download(target_model_path)
            raw_model = load_keras_or_savedmodel(local_dir)
        except ImportError:
            raise ModelLoadError(
                f"Model '{target_model_path}' is a TensorFlow/Keras model ('{runtime_metadata.framework}'), "
                f"which requires the 'tensorflow' package. TensorFlow is not installed in the current environment. "
                f"Please install 'tensorflow' to ingest and evaluate TensorFlow/Keras models."
            )
    elif framework_clean in ("xgboost", "lightgbm", "catboost"):
        from app.ingestion.loader.tree_loader import TreeModelLoader
        raw_model = TreeModelLoader().load(target_model_path, framework=framework_clean)
    elif framework_clean == "openvino":
        try:
            import openvino.runtime as ov
            core = ov.Core()
            raw_model = core.compile_model(str(target_model_path), "CPU")
        except ImportError:
            raise ModelLoadError(
                f"Model '{target_model_path}' is an OpenVINO model, which requires 'openvino'. "
                f"openvino is not installed in the current environment. Please install 'openvino'."
            )
    elif framework_clean == "tensorrt":
        raise UnsupportedModelError(
            f"Model '{target_model_path}' is a TensorRT engine. "
            f"TensorRT execution requires GPU runtime with NVIDIA TensorRT drivers. Non-GPU CPU environment is unsupported."
        )
    elif framework_clean == "coreml":
        try:
            import coremltools as ct
            raw_model = ct.models.MLModel(str(target_model_path))
        except ImportError:
            raise ModelLoadError(
                f"Model '{target_model_path}' is a Core ML model, which requires 'coremltools'. "
                f"coremltools is not installed in the current environment. Please install 'coremltools'."
            )
    elif framework_clean == "tflite":
        try:
            import tensorflow as tf
            raw_model = tf.lite.Interpreter(model_path=str(target_model_path))
            raw_model.allocate_tensors()
        except ImportError:
            raise ModelLoadError(
                f"Model '{target_model_path}' is a TFLite model, which requires 'tensorflow' or 'tflite-runtime'. "
                f"Please install 'tflite-runtime' or 'tensorflow'."
            )
    elif framework_clean == "paddle":
        try:
            import paddle
            raw_model = paddle.jit.load(str(target_model_path))
        except ImportError:
            raise ModelLoadError(
                f"Model '{target_model_path}' is a PaddlePaddle model, which requires 'paddlepaddle'. "
                f"Please install 'paddlepaddle'."
            )
    elif framework_clean == "mxnet":
        raise UnsupportedModelError(
            f"Model '{target_model_path}' is an Apache MXNet model. "
            f"Apache MXNet reached end-of-life; please convert the model to ONNX."
        )
    elif framework_clean == "jax":
        raise UnsupportedModelError(
            f"Model '{target_model_path}' is a JAX/Flax model. "
            f"AdverScan currently supports PyTorch, Scikit-learn, ONNX, and tree-based models."
        )
    elif framework_clean == "onnx":
        try:
            import onnxruntime
            p = Path(target_model_path)
            if p.is_file():
                raw_model = onnxruntime.InferenceSession(str(p))
            elif "/" in str(target_model_path) and not str(target_model_path).startswith((".", "/")):
                from huggingface_hub import hf_hub_download
                from app.ingestion.utils.model_utils import detect_hf_repo_info
                hf_info = detect_hf_repo_info(str(target_model_path))
                onnx_file = hf_info.get("model_file") or next(
                    (f for f in hf_info.get("files", []) if str(f).endswith(".onnx")), None
                )
                if not onnx_file:
                    raise ModelLoadError(f"No .onnx model file found in Hugging Face repository '{target_model_path}'.")
                local_onnx = hf_hub_download(str(target_model_path), onnx_file)
                raw_model = onnxruntime.InferenceSession(local_onnx)
            else:
                raw_model = onnxruntime.InferenceSession(str(target_model_path))
        except ImportError:
            raise ModelLoadError(
                f"Model '{target_model_path}' is an ONNX model, which requires 'onnxruntime'. "
                f"onnxruntime is not installed in the current environment. Please install 'onnxruntime'."
            )
    elif framework_clean == "huggingface":
        loader = HuggingFaceLoader()
        raw_model = loader.load(
            target_model_path,
            device=target_device,
            domain=runtime_metadata.data_domain,
            task_type=runtime_metadata.task_type,
            **kwargs,
        )
    elif framework_clean in ("pytorch", "pytorch_local"):
        loader = PyTorchLoader()
        raw_model = loader.load(
            target_model_path,
            model_class=model_class,
            device=target_device,
            **kwargs,
        )
    else:
        raise UnsupportedModelError(
            f"Unsupported framework '{runtime_metadata.framework}'. Supported frameworks are: ['huggingface', 'pytorch', 'sklearn', 'keras', 'tensorflow', 'onnx']"
        )

    # 6. Select processor using data_domain
    domain_clean = (runtime_metadata.data_domain or "unknown").lower().replace("_", "-")
    if domain_clean == "timeseries":
        domain_clean = "time-series"
    processor: Optional[BaseProcessor] = None

    if domain_clean == "image":
        resolution = config_dict.get("input_resolution")
        processor = ImageProcessor(
            processor_name_or_path=runtime_metadata.processor_name,
            device=target_device,
            input_resolution=resolution,
        )
    elif domain_clean == "text":
        max_length = config_dict.get("max_length", 128)
        processor = TextProcessor(
            processor_name_or_path=runtime_metadata.processor_name,
            device=target_device,
            max_length=max_length,
        )
    elif domain_clean == "time-series":
        processor = TimeSeriesProcessor(
            processor_name_or_path=runtime_metadata.processor_name,
            device=target_device,
            target_format=config_dict.get("target_format", "batch_channels_length"),
            sequence_length=config_dict.get("sequence_length"),
            num_channels=config_dict.get("num_channels"),
            normalize=config_dict.get("normalize", False),
            standardize=config_dict.get("standardize", False),
            scaling=config_dict.get("scaling"),
        )
    elif domain_clean == "tabular":
        processor = TabularProcessor(
            processor_name_or_path=runtime_metadata.processor_name,
            device=target_device,
            feature_names=config_dict.get("feature_names"),
            feature_types=config_dict.get("feature_types"),
            numerical_features=config_dict.get("numerical_features"),
            categorical_features=config_dict.get("categorical_features"),
            binary_features=config_dict.get("binary_features"),
            ordinal_features=config_dict.get("ordinal_features"),
            target_column=config_dict.get("target_column"),
            num_features=config_dict.get("num_features"),
            num_classes=config_dict.get("num_classes"),
            class_mapping=config_dict.get("class_mapping"),
            preprocessor=config_dict.get("preprocessor"),
            preprocessing_info=config_dict.get("preprocessing_info"),
            categorical_threshold=config_dict.get("categorical_threshold", 10),
            sample_data=sample_input,
            feature_constraints=config_dict.get("feature_constraints"),
            immutable_features=config_dict.get("immutable_features"),
            mutable_features=config_dict.get("mutable_features"),
            representation_space=config_dict.get("representation_space"),
            categorical_encoding_type=config_dict.get("categorical_encoding_type"),
            encoding_info=config_dict.get("encoding_info"),
        )
    elif domain_clean == "unknown":
        processor = None
    else:
        raise UnsupportedDomainError(
            f"Unsupported data domain '{domain_clean}'. Current scope supports only ['image', 'text', 'time-series', 'tabular']."
        )

    # 7. Select adapter using actual loaded model / framework / data_domain
    adapter: BaseModelAdapter
    if framework_clean == "sklearn" or is_sklearn_model(raw_model):
        adapter = SklearnAdapter(
            raw_model,
            device=target_device,
            processor=processor,
            data_domain=domain_clean,
        )
        runtime_metadata.extra_info["preprocessing"] = SklearnLoader.inspect_preprocessing(raw_model)
        bundle_info = getattr(raw_model, "_bundle_info", None)
        if isinstance(bundle_info, dict):
            runtime_metadata.extra_info["bundle_info"] = bundle_info
            if bundle_info.get("feature_names"):
                runtime_metadata.extra_info["feature_names"] = bundle_info["feature_names"]
                if isinstance(processor, TabularProcessor) and not processor.feature_names:
                    processor.feature_names = list(bundle_info["feature_names"])
                    processor.num_features = len(bundle_info["feature_names"])
            if bundle_info.get("classes") and runtime_metadata.num_classes is None:
                runtime_metadata.num_classes = len(bundle_info["classes"])
    elif framework_clean == "onnx" or (hasattr(raw_model, "run") and hasattr(raw_model, "get_inputs")):
        adapter = ONNXAdapter(
            raw_model,
            device=target_device,
            processor=processor,
            data_domain=domain_clean,
        )
    elif framework_clean in ("xgboost", "lightgbm", "catboost") or any(
        k in raw_model.__class__.__name__.lower() for k in ("xgb", "lgb", "catboost", "booster")
    ):
        fw_resolved = "xgboost" if any(k in raw_model.__class__.__name__.lower() for k in ("xgb", "booster")) else framework_clean
        adapter = TreeModelAdapter(
            raw_model,
            device=target_device,
            processor=processor,
            data_domain=domain_clean,
            framework=fw_resolved,
            feature_names=getattr(raw_model, "feature_names", None),
            raw_input_features=getattr(raw_model, "raw_input_features", None),
            preprocessing_info=getattr(raw_model, "preprocessing_info", None),
            calibrator_info=getattr(raw_model, "calibrator_info", None),
            threshold=getattr(raw_model, "threshold", 0.5),
        )
    elif framework_clean in ("keras", "tensorflow") or any(
        k in getattr(raw_model.__class__, "__module__", "").lower() for k in ("keras", "tensorflow")
    ):
        adapter = KerasAdapter(
            raw_model,
            device=target_device,
            processor=processor,
            data_domain=domain_clean,
        )
    elif framework_clean in ("openvino", "tensorrt", "coreml", "tflite"):
        adapter = DeploymentRuntimeAdapter(
            raw_model,
            device=target_device,
            processor=processor,
            data_domain=domain_clean,
            runtime_name=framework_clean,
        )
    elif framework_clean == "huggingface":
        adapter = HuggingFaceAdapter(
            raw_model,
            device=target_device,
            processor=processor,
            data_domain=domain_clean,
        )
    else:
        adapter = PyTorchAdapter(
            raw_model,
            device=target_device,
            processor=processor,
        )

    # Record detection evidence and adapter capabilities
    evidence = inspect_model_evidence(
        raw_model,
        framework=framework_clean,
        domain=domain_clean,
        sample_input=sample_input,
    )
    runtime_metadata.extra_info["evidence"] = evidence
    runtime_metadata.extra_info["source"] = detected_source
    runtime_metadata.extra_info["capabilities"] = {
        "supports_gradients": adapter.capabilities.supports_gradients,
        "supports_predict": adapter.capabilities.supports_predict,
        "supports_batch_predict": adapter.capabilities.supports_batch_predict,
        "supports_logits": adapter.capabilities.supports_logits,
    }

    # 8. Model validation and sanity verification
    if isinstance(processor, TabularProcessor):
        processor.validate_model_input(adapter, sample_input=sample_input)

    # If sample_input was not provided by caller, generate a realistic sample for validation
    validation_sample = sample_input
    if validation_sample is None:
        try:
            if domain_clean in ("tabular", "csv"):
                n_feats = None
                if hasattr(raw_model, "n_features_in_"):
                    n_feats = getattr(raw_model, "n_features_in_")
                elif hasattr(raw_model, "feature_names_in_"):
                    n_feats = len(getattr(raw_model, "feature_names_in_"))
                elif hasattr(raw_model, "feature_names") and raw_model.feature_names:
                    n_feats = len(raw_model.feature_names)
                elif hasattr(raw_model, "num_features") and callable(raw_model.num_features):
                    n_feats = raw_model.num_features()
                elif getattr(adapter, "feature_names", None):
                    n_feats = len(adapter.feature_names)
                elif config_dict.get("num_features"):
                    n_feats = config_dict["num_features"]
                elif runtime_metadata.input_shape and len(runtime_metadata.input_shape) >= 1:
                    n_feats = runtime_metadata.input_shape[-1]
                elif hasattr(raw_model, "named_parameters"):
                    for p_name, param in raw_model.named_parameters():
                        if "weight" in p_name and len(param.shape) == 2:
                            n_feats = int(param.shape[1])
                            break
                if n_feats is not None:
                    validation_sample = np.random.randn(2, int(n_feats)).astype(np.float32)
            elif domain_clean == "text" and processor is not None:
                validation_sample = ["AdverScan validation sample text."]
            elif domain_clean == "time-series":
                saved_shape = getattr(raw_model, "_savedmodel_input_shape", None) or getattr(raw_model, "input_shape", None)
                if isinstance(saved_shape, (tuple, list)) and len(saved_shape) >= 2 and all(isinstance(d, int) and d > 0 for d in saved_shape[1:]):
                    validation_sample = torch.randn(2, *[int(d) for d in saved_shape[1:]])
                else:
                    s_len = config_dict.get("sequence_length")
                    n_ch = config_dict.get("num_channels")
                    if hasattr(raw_model, "config"):
                        cfg = getattr(raw_model, "config")
                        s_len = s_len or getattr(cfg, "context_length", getattr(cfg, "sequence_length", None))
                        n_ch = n_ch or getattr(cfg, "num_input_channels", getattr(cfg, "in_channels", None))
                    n_ch = n_ch or 1
                    if s_len is not None:
                        # Try batch_length_channels [B, L, C] then batch_channels_length [B, C, L]
                        validation_sample = torch.randn(2, int(s_len), int(n_ch))
                        try:
                            adapter.predict(validation_sample)
                        except Exception:
                            validation_sample = torch.randn(2, int(n_ch), int(s_len))
            elif domain_clean == "image":
                res = config_dict.get("input_resolution") or (3, 224, 224)
                if len(res) == 2:
                    validation_sample = torch.randn(2, 3, res[0], res[1])
                elif len(res) == 3:
                    validation_sample = torch.randn(2, *res)
        except Exception:
            validation_sample = None

    ModelValidator.validate(adapter, sample_input=validation_sample)

    # 9. Metadata extraction and shape inference
    input_shape = None
    output_shape = None
    num_classes = config_dict.get("num_classes")

    effective_sample = sample_input if sample_input is not None else validation_sample
    if effective_sample is not None:
        if isinstance(effective_sample, torch.Tensor):
            input_shape = tuple(effective_sample.shape)
        elif hasattr(effective_sample, "shape"):
            input_shape = tuple(effective_sample.shape)
        elif processor is not None:
            try:
                proc_in = processor.process(effective_sample)
                if hasattr(proc_in, "shape"):
                    input_shape = tuple(proc_in.shape)
                elif isinstance(proc_in, dict):
                    for k in ("input_ids", "pixel_values", "inputs", "x"):
                        if k in proc_in and hasattr(proc_in[k], "shape"):
                            input_shape = tuple(proc_in[k].shape)
                            break
            except Exception:
                pass

        try:
            out = adapter.predict(effective_sample)
            if isinstance(out, torch.Tensor):
                output_shape = tuple(out.shape)
                if len(output_shape) > 1:
                    num_classes = output_shape[-1]
            elif isinstance(out, np.ndarray):
                output_shape = tuple(out.shape)
                if len(output_shape) > 1:
                    num_classes = output_shape[-1]
        except Exception:
            pass
    elif config_dict.get("input_resolution") and domain_clean == "image":
        res = config_dict["input_resolution"]
        input_shape = (1, *res) if len(res) == 3 else (1, 3, *res)
    elif domain_clean == "time-series":
        seq_len = config_dict.get("sequence_length")
        num_ch = config_dict.get("num_channels", 1)
        if hasattr(raw_model, "config"):
            cfg = getattr(raw_model, "config")
            seq_len = seq_len or getattr(cfg, "context_length", getattr(cfg, "sequence_length", None))
            num_ch = num_ch or getattr(cfg, "num_input_channels", getattr(cfg, "in_channels", 1))
            if num_classes is None:
                num_cls_val = getattr(cfg, "num_classes", getattr(cfg, "num_targets", getattr(cfg, "num_labels", None)))
                if num_cls_val is not None:
                    num_classes = int(num_cls_val)

        if num_classes is None:
            for layer_name in ["fc", "classifier", "head", "out_proj"]:
                layer = getattr(raw_model, layer_name, None)
                if isinstance(layer, nn.Linear):
                    num_classes = layer.out_features
                    break

        if seq_len is not None:
            input_shape = (1, num_ch, seq_len)

    if input_shape is None and hasattr(raw_model, "_savedmodel_input_shape"):
        input_shape = getattr(raw_model, "_savedmodel_input_shape")
    if output_shape is None and hasattr(raw_model, "_savedmodel_output_shape"):
        output_shape = getattr(raw_model, "_savedmodel_output_shape")
        if num_classes is None and output_shape and len(output_shape) > 1:
            num_classes = output_shape[-1]

    if domain_clean == "tabular" and isinstance(processor, TabularProcessor):
        runtime_metadata.tabular_metadata = processor.tabular_metadata
        runtime_metadata.extra_info["tabular_metadata"] = processor.tabular_metadata.to_dict()
        if input_shape is None and processor.num_features:
            input_shape = (1, processor.num_features)
        if num_classes is None and processor.tabular_metadata.num_classes:
            num_classes = processor.tabular_metadata.num_classes
        if num_classes is None and hasattr(raw_model, "classes_") and raw_model.classes_ is not None:
            num_classes = len(raw_model.classes_)

    # Parameter count and architecture
    total_params = None
    trainable_params = None
    model_arch = raw_model.__class__.__name__ if hasattr(raw_model, "__class__") else None
    if framework_clean == "xgboost":
        model_arch = "XGBoost Gradient Boosted Trees"
    elif framework_clean == "lightgbm":
        model_arch = "LightGBM Gradient Boosted Trees"
    elif framework_clean == "catboost":
        model_arch = "CatBoost Classifier"
    if hasattr(raw_model, "parameters"):
        try:
            params_list = [p for p in raw_model.parameters() if hasattr(p, "numel")]
            if params_list:
                total_params = int(sum(p.numel() for p in params_list))
                trainable_params = int(sum(p.numel() for p in params_list if getattr(p, "requires_grad", False)))
        except Exception:
            pass

    runtime_metadata.input_shape = input_shape
    runtime_metadata.output_shape = output_shape
    runtime_metadata.num_classes = num_classes
    runtime_metadata.total_parameters = total_params
    runtime_metadata.trainable_parameters = trainable_params
    runtime_metadata.model_architecture = model_arch

    # Ingestion logging
    if log_ingestion:
        log_ingestion_event(
            {
                "status": "INGESTION_SUCCESS",
                "model_id": runtime_metadata.model_id,
                "model_name": runtime_metadata.model_name,
                "framework": runtime_metadata.framework,
                "data_domain": runtime_metadata.data_domain,
                "domain": runtime_metadata.domain,
                "task_type": runtime_metadata.task_type,
                "target_device": runtime_metadata.device,
                "num_classes": runtime_metadata.num_classes,
                "input_shape": list(runtime_metadata.input_shape) if runtime_metadata.input_shape else None,
                "processor_name": runtime_metadata.processor_name,
            },
            log_filepath=log_filepath,
        )

    # 10. Return standardized ingestion result
    return IngestionResult(adapter=adapter, metadata=runtime_metadata, processor=processor)
