"""
Centralized model artifact detection and inspection registry for AdverScan.

Identifies model ecosystems, serialization formats, weight files, preprocessing,
calibration, and metadata artifacts across local files and Hugging Face repositories.
"""

from __future__ import annotations

import json
import logging
import pickletools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class RankedCandidate:
    """A ranked (loader, artifact) candidate with confidence score and evidence."""
    loader: str
    artifact: Optional[str]
    confidence: float
    evidence: List[str] = field(default_factory=list)
    framework: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loader": self.loader,
            "artifact": self.artifact,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "framework": self.framework,
        }


@dataclass
class ArtifactInspectionResult:
    """Detailed result of repository or local directory artifact inspection."""
    framework: str = "unknown"
    domain: str = "unknown"
    task_type: str = "classification"
    serialization_format: str = "unknown"
    model_file: Optional[str] = None
    is_model_repo: bool = False
    repo_type: str = "unknown"  # "model", "dataset", "source_only", "empty", "incomplete"
    weight_files: List[str] = field(default_factory=list)
    preprocessing_files: List[str] = field(default_factory=list)
    calibration_files: List[str] = field(default_factory=list)
    metadata_files: List[str] = field(default_factory=list)
    config_files: List[str] = field(default_factory=list)
    tokenizer_files: List[str] = field(default_factory=list)
    feature_names: Optional[List[str]] = None
    raw_feature_names: Optional[List[str]] = None
    num_features: Optional[int] = None
    num_classes: Optional[int] = None
    architecture: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    error_detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "framework": self.framework,
            "domain": self.domain,
            "task_type": self.task_type,
            "serialization_format": self.serialization_format,
            "model_file": self.model_file,
            "is_model_repo": self.is_model_repo,
            "repo_type": self.repo_type,
            "weight_files": self.weight_files,
            "preprocessing_files": self.preprocessing_files,
            "calibration_files": self.calibration_files,
            "metadata_files": self.metadata_files,
            "config_files": self.config_files,
            "tokenizer_files": self.tokenizer_files,
            "feature_names": self.feature_names,
            "raw_feature_names": self.raw_feature_names,
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "architecture": self.architecture,
            "evidence": self.evidence,
            "error_detail": self.error_detail,
        }


class ModelArtifactDetector:
    """
    Format-driven, extensible artifact detection system.
    Inspects file manifests, file signatures, and metadata to deterministically
    classify model ecosystems without relying solely on file extensions.
    """

    # Non-model artifacts to distinguish from weights
    METADATA_FILENAMES = {
        "metadata.json", "metrics.json", "reference_stats.json", "meta.json",
        "model_card.json", "schema.json", "info.json"
    }
    PREPROCESSING_FILENAMES = {
        "preprocess.json", "preprocessor.json", "scaler.json", "pipeline.json",
        "transforms.json", "feature_map.json", "encoders.json", "preprocessor_config.json"
    }
    CALIBRATION_FILENAMES = {
        "calibrator.json", "calibration.json", "platt.json", "isotonic.json"
    }
    DATASET_EXTENSIONS = {
        ".csv", ".tsv", ".parquet", ".arrow", ".feather", ".jsonl", ".sqlite", ".db"
    }
    SOURCE_CODE_EXTENSIONS = {
        ".py", ".sh", ".bash", ".c", ".cpp", ".cu", ".h", ".ipynb", ".r"
    }
    DOC_EXTENSIONS = {
        ".md", ".rst", ".txt", ".pdf", ".png", ".jpg", ".jpeg", ".svg", ".gif"
    }

    @classmethod
    def inspect_file_list(
        cls,
        files: List[str],
        library_name: Optional[str] = None,
        pipeline_tag: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        metadata_dict: Optional[Dict[str, Any]] = None,
        repo_id: Optional[str] = None,
    ) -> ArtifactInspectionResult:
        """
        Analyze a file manifest and metadata to categorize model artifacts.
        """
        result = ArtifactInspectionResult()
        norm_files = [f.strip() for f in files if f.strip() and not f.startswith(".git")]
        result.config_files = [f for f in norm_files if f.lower() in ("config.json", "hyperparameters.json")]
        result.preprocessing_files = [f for f in norm_files if Path(f).name.lower() in cls.PREPROCESSING_FILENAMES]
        result.calibration_files = [f for f in norm_files if Path(f).name.lower() in cls.CALIBRATION_FILENAMES]
        result.metadata_files = [f for f in norm_files if Path(f).name.lower() in cls.METADATA_FILENAMES]
        result.tokenizer_files = [
            f for f in norm_files if any(k in f.lower() for k in ("tokenizer", "vocab", "merges", "spiece"))
        ]

        lib = (library_name or "").lower().strip()
        p_tag = (pipeline_tag or "").lower().strip()
        tag_list = [t.lower().strip() for t in (tags or [])]

        # 1. Collect candidate weight files across ecosystems
        pytorch_weights: List[str] = []
        xgboost_weights: List[str] = []
        lightgbm_weights: List[str] = []
        catboost_weights: List[str] = []
        sklearn_weights: List[str] = []
        onnx_weights: List[str] = []
        keras_weights: List[str] = []
        deployment_weights: List[str] = []
        jax_weights: List[str] = []

        for f in norm_files:
            lower = f.lower()
            fname = Path(lower).name

            # Skip auxiliary metadata / preprocessing / dataset files
            if fname in cls.METADATA_FILENAMES or fname in cls.PREPROCESSING_FILENAMES or fname in cls.CALIBRATION_FILENAMES:
                continue
            if fname in ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"):
                continue

            # Safetensors and PyTorch
            if lower.endswith((".safetensors", ".pt", ".pth", ".ckpt")) or fname in ("pytorch_model.bin", "model.bin"):
                pytorch_weights.append(f)
            elif lower.endswith(".bin") and not any(k in lower for k in ("openvino", "params")):
                pytorch_weights.append(f)

            # XGBoost
            elif (
                fname in ("xgb_model.json", "xgboost.json", "model.json", "booster.json")
                or lower.endswith((".ubj",))
                or ("xgb" in lower and lower.endswith((".json", ".model")))
                or ("xgboost" in lower and lower.endswith((".json", ".model")))
            ):
                xgboost_weights.append(f)

            # LightGBM
            elif (
                fname in ("model.txt", "lightgbm.model", "lgb_model.txt")
                or ("lightgbm" in lower and lower.endswith((".txt", ".model")))
                or ("lgb" in lower and lower.endswith((".txt", ".model")))
            ):
                lightgbm_weights.append(f)

            # CatBoost
            elif lower.endswith((".cbm",)) or ("catboost" in lower and lower.endswith((".cbm", ".bin", ".model"))):
                catboost_weights.append(f)

            # Scikit-learn
            elif lower.endswith((".joblib", ".pkl", ".pickle")):
                sklearn_weights.append(f)

            # ONNX
            elif lower.endswith(".onnx"):
                onnx_weights.append(f)

            # TensorFlow / Keras
            elif fname in ("saved_model.pb", "keras_metadata.pb") or lower.endswith((".keras", ".h5", ".hdf5")):
                keras_weights.append(f)

            # JAX / Flax
            elif lower.endswith((".msgpack", ".flax")):
                jax_weights.append(f)

            # Deployment Formats
            elif lower.endswith((".xml", ".engine", ".plan", ".mlmodel", ".mlpackage", ".tflite", ".pdmodel", ".params")):
                deployment_weights.append(f)

        # 2. Check for dataset-only or source-only repos
        dataset_files = [f for f in norm_files if any(f.lower().endswith(ext) for ext in cls.DATASET_EXTENSIONS)]
        source_files = [f for f in norm_files if any(f.lower().endswith(ext) for ext in cls.SOURCE_CODE_EXTENSIONS)]

        has_any_weights = bool(
            pytorch_weights or xgboost_weights or lightgbm_weights or catboost_weights
            or sklearn_weights or onnx_weights or keras_weights or deployment_weights or jax_weights
        )

        if not norm_files:
            result.repo_type = "empty"
            result.is_model_repo = False
            result.error_detail = "Repository contains no files."
            return result

        if not has_any_weights:
            if dataset_files and len(dataset_files) >= len(norm_files) - 2:
                result.repo_type = "dataset"
                result.is_model_repo = False
                result.error_detail = "Repository appears to be a dataset repository without serialized model weights."
                return result
            if source_files and not any(f.endswith((".json", ".pb", ".bin")) for f in norm_files):
                result.repo_type = "source_only"
                result.is_model_repo = False
                result.error_detail = "Repository contains only source code and documentation without model weights."
                return result

            # Check if library is xgboost or lightgbm and there is an ambiguous json/txt
            if "xgboost" in lib or "xgboost" in tag_list:
                cand_json = [f for f in norm_files if f.endswith(".json") and f not in cls.METADATA_FILENAMES and f not in cls.PREPROCESSING_FILENAMES]
                if cand_json:
                    xgboost_weights.extend(cand_json)
                    has_any_weights = True

        # 3. Determine framework deterministically using weights + metadata
        if xgboost_weights and (
            "xgboost" in lib or "xgboost" in tag_list
            or any("xgb" in w.lower() for w in xgboost_weights)
            or not (pytorch_weights or keras_weights)
        ):
            result.framework = "xgboost"
            result.serialization_format = Path(xgboost_weights[0]).suffix or ".json"
            result.model_file = xgboost_weights[0]
            result.weight_files = xgboost_weights
            result.is_model_repo = True
            result.domain = "tabular"
            result.task_type = "classification"
            result.architecture = "XGBoost Gradient Boosted Trees"
            result.evidence.append(f"Detected native XGBoost model artifact '{result.model_file}'")

        elif lightgbm_weights and ("lightgbm" in lib or "lightgbm" in tag_list or not pytorch_weights):
            result.framework = "lightgbm"
            result.serialization_format = Path(lightgbm_weights[0]).suffix or ".txt"
            result.model_file = lightgbm_weights[0]
            result.weight_files = lightgbm_weights
            result.is_model_repo = True
            result.domain = "tabular"
            result.task_type = "classification"
            result.architecture = "LightGBM Gradient Boosted Trees"
            result.evidence.append(f"Detected native LightGBM model artifact '{result.model_file}'")

        elif catboost_weights:
            result.framework = "catboost"
            result.serialization_format = ".cbm"
            result.model_file = catboost_weights[0]
            result.weight_files = catboost_weights
            result.is_model_repo = True
            result.domain = "tabular"
            result.task_type = "classification"
            result.architecture = "CatBoost Classifier"
            result.evidence.append(f"Detected native CatBoost model artifact '{result.model_file}'")

        elif sklearn_weights and ("sklearn" in lib or "scikit-learn" in tag_list or not pytorch_weights):
            result.framework = "sklearn"
            result.serialization_format = Path(sklearn_weights[0]).suffix
            result.model_file = sklearn_weights[0]
            result.weight_files = sklearn_weights
            result.is_model_repo = True
            result.domain = "tabular"
            result.task_type = "classification"
            result.architecture = "Scikit-Learn Estimator"
            result.evidence.append(f"Detected Scikit-Learn model artifact '{result.model_file}'")

        elif onnx_weights and ("onnx" in lib or not pytorch_weights):
            result.framework = "onnx"
            result.serialization_format = ".onnx"
            result.model_file = onnx_weights[0]
            result.weight_files = onnx_weights
            result.is_model_repo = True
            result.evidence.append(f"Detected ONNX runtime model artifact '{result.model_file}'")

        elif keras_weights and ("keras" in lib or "tensorflow" in lib or not pytorch_weights):
            result.framework = "keras"
            result.serialization_format = Path(keras_weights[0]).suffix or "SavedModel"
            result.model_file = keras_weights[0]
            result.weight_files = keras_weights
            result.is_model_repo = True
            result.evidence.append(f"Detected TensorFlow/Keras model artifact '{result.model_file}'")

        elif pytorch_weights:
            result.framework = "pytorch"
            result.serialization_format = Path(pytorch_weights[0]).suffix
            result.model_file = pytorch_weights[0]
            result.weight_files = pytorch_weights
            result.is_model_repo = True
            result.evidence.append(f"Detected PyTorch model weights artifact '{result.model_file}'")

        elif deployment_weights:
            dep_f = deployment_weights[0]
            suff = Path(dep_f).suffix.lower()
            if suff == ".xml":
                result.framework = "openvino"
            elif suff in (".engine", ".plan"):
                result.framework = "tensorrt"
            elif suff in (".mlmodel", ".mlpackage"):
                result.framework = "coreml"
            elif suff == ".tflite":
                result.framework = "tflite"
            elif suff in (".pdmodel", ".pdparams"):
                result.framework = "paddle"
            elif suff == ".params":
                result.framework = "mxnet"
            result.serialization_format = suff
            result.model_file = dep_f
            result.weight_files = deployment_weights
            result.is_model_repo = True
            result.evidence.append(f"Detected deployment runtime artifact '{result.model_file}' ({result.framework})")

        elif jax_weights:
            result.framework = "jax"
            result.serialization_format = Path(jax_weights[0]).suffix
            result.model_file = jax_weights[0]
            result.weight_files = jax_weights
            result.is_model_repo = True
            result.evidence.append("Detected JAX/Flax checkpoint")

        else:
            result.is_model_repo = False
            result.repo_type = "incomplete"
            result.error_detail = "No recognized serialized model weights found."

        # 4. Resolve Domain and Task if not yet determined
        if result.domain == "unknown":
            if p_tag:
                if any(k in p_tag for k in ("tabular", "structured", "fraud", "churn")):
                    result.domain = "tabular"
                elif any(k in p_tag for k in ("image", "vision", "cv", "detection", "segmentation")):
                    result.domain = "image"
                elif any(k in p_tag for k in ("text", "nlp", "translation", "summarization", "qa", "ner")):
                    result.domain = "text"
                elif any(k in p_tag for k in ("time-series", "timeseries", "sequence")):
                    result.domain = "time-series"

            if result.domain == "unknown" and repo_id:
                clean_repo = repo_id.lower()
                if any(k in clean_repo for k in ("tabular", "credit", "fraud", "census", "adult", "churn", "xgboost", "catboost")):
                    result.domain = "tabular"
                elif any(k in clean_repo for k in ("patchtst", "timeseries", "time-series", "chronos", "timesfm")):
                    result.domain = "time-series"
                elif any(k in clean_repo for k in ("vit", "resnet", "vision", "cifar", "image", "traffic", "gtsrb")):
                    result.domain = "image"
                elif any(k in clean_repo for k in ("bert", "roberta", "gpt", "t5", "nlp", "text")):
                    result.domain = "text"

        # 5. Extract feature schema & metadata from metadata_dict / config_dict if available
        if metadata_dict:
            if "raw_input_features" in metadata_dict and isinstance(metadata_dict["raw_input_features"], list):
                result.raw_feature_names = [str(col) for col in metadata_dict["raw_input_features"]]
            if "feature_names" in metadata_dict and isinstance(metadata_dict["feature_names"], list):
                result.feature_names = [str(col) for col in metadata_dict["feature_names"]]
                result.num_features = len(result.feature_names)

        result.repo_type = "model" if result.is_model_repo else result.repo_type
        return result

    @staticmethod
    def _inspect_file_magic(file_path: Path) -> Optional[str]:
        """Inspect file magic bytes without executing deserialization."""
        try:
            with open(file_path, "rb") as f:
                head = f.read(64)
            if head.startswith(b"PK\x03\x04"):
                return "zip_keras"
            if head.startswith(b"\x89HDF\r\n\x1a\n"):
                return "hdf5_h5"
            stripped = head.lstrip()
            if stripped.startswith(b"{"):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as tf:
                    snippet = tf.read(4096)
                if '"learner"' in snippet:
                    return "xgboost_json"
        except Exception as exc:
            logger.debug("Magic byte inspection failed for %s: %s", file_path, exc)
        return None

    @staticmethod
    def _scan_pickle_opcodes(file_path: Path, max_bytes: int = 262144) -> List[str]:
        """Scan pickle/joblib opcodes via pickletools WITHOUT executing any code (bounded read)."""
        modules: List[str] = []
        try:
            with open(file_path, "rb") as f:
                raw = f.read(max_bytes)
            data = raw
            if raw.startswith(b"\x1f\x8b"):
                import gzip
                with gzip.GzipFile(fileobj=__import__("io").BytesIO(raw)) as gz:
                    data = gz.read(max_bytes)
            elif raw.startswith(b"BZh"):
                import bz2
                data = bz2.BZ2Decompressor().decompress(raw, max_length=max_bytes)
            elif raw.startswith(b"\xfd7zXZ"):
                import lzma
                data = lzma.LZMADecompressor().decompress(raw, max_length=max_bytes)
            elif len(raw) >= 2 and raw[0] == 0x78 and raw[1] in (0x01, 0x5E, 0x9C, 0xDA):
                import zlib
                data = zlib.decompressobj().decompress(raw, max_bytes)

            str_stack: List[str] = []
            for opcode, arg, _pos in pickletools.genops(data):
                if opcode.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE") and isinstance(arg, str):
                    str_stack.append(arg)
                    if len(str_stack) > 4:
                        str_stack.pop(0)
                elif opcode.name == "GLOBAL" and isinstance(arg, str):
                    mod = arg.split()[0].split(".")[0].lower()
                    if mod and mod not in modules:
                        modules.append(mod)
                elif opcode.name == "STACK_GLOBAL" and len(str_stack) >= 2:
                    mod = str_stack[-2].split(".")[0].lower()
                    if mod and mod not in modules:
                        modules.append(mod)
        except Exception as exc:
            logger.debug("Pickle opcode scan skipped for %s: %s", file_path, exc)
        return modules

    @staticmethod
    def _classify_config_json(cfg_dict: Optional[Dict[str, Any]]) -> str:
        """Classify config.json ecosystem: transformers, skops, autotrain, or unknown."""
        if not isinstance(cfg_dict, dict) or not cfg_dict:
            return "unknown"
        if "sklearn" in cfg_dict:
            return "skops"
        if isinstance(cfg_dict.get("features"), list):
            return "autotrain"
        if cfg_dict.get("model_type") or cfg_dict.get("architectures"):
            return "transformers"
        return "unknown"

    @classmethod
    def rank_candidates(
        cls,
        files: List[str],
        local_dir: Optional[Path] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        hf_info: Optional[Dict[str, Any]] = None,
    ) -> List[RankedCandidate]:
        """
        Build a ranked list of (loader, artifact, confidence, evidence) candidates
        using file manifest, config.json kind, magic bytes, and pickletools opcode scan.
        """
        candidates: List[RankedCandidate] = []
        cfg_kind = cls._classify_config_json(config_dict)
        lib_name = str((hf_info or {}).get("library_name") or "").lower()

        for fname in files:
            low = fname.lower()
            magic = None
            pickle_mods: List[str] = []
            if local_dir is not None and (local_dir / fname).is_file():
                fpath = local_dir / fname
                magic = cls._inspect_file_magic(fpath)
                if low.endswith((".pkl", ".pickle", ".joblib")):
                    pickle_mods = cls._scan_pickle_opcodes(fpath)

            if low.endswith((".json", ".ubj", ".model")) and (
                magic == "xgboost_json" or "xgboost" in low or "xgb" in low
            ):
                candidates.append(RankedCandidate("tree", fname, 0.96, [f"XGBoost artifact {fname}"], "xgboost"))
            elif low.endswith(".cbm"):
                candidates.append(RankedCandidate("tree", fname, 0.96, [f"CatBoost artifact {fname}"], "catboost"))
            elif low.endswith(".onnx"):
                candidates.append(RankedCandidate("onnx", fname, 0.95, [f"ONNX artifact {fname}"], "onnx"))
            elif low.endswith((".keras", ".h5", ".hdf5")) or low == "saved_model.pb" or magic in ("zip_keras", "hdf5_h5"):
                candidates.append(RankedCandidate("keras", fname, 0.93, [f"Keras/TF artifact {fname}"], "keras"))
            elif low.endswith((".joblib", ".pkl", ".pickle")):
                ev = [f"Serialized tabular artifact {fname}"]
                if cfg_kind == "autotrain":
                    ev.append("AutoTrain config.json (features list)")
                if pickle_mods:
                    ev.append(f"pickletools modules: {pickle_mods[:5]}")
                if "xgboost" in pickle_mods or "xgboost" in lib_name or "xgb" in low:
                    candidates.append(RankedCandidate("tree", fname, 0.92, ev, "xgboost"))
                    candidates.append(RankedCandidate("sklearn", fname, 0.85, ev, "sklearn"))
                elif "lightgbm" in pickle_mods:
                    candidates.append(RankedCandidate("tree", fname, 0.92, ev, "lightgbm"))
                    candidates.append(RankedCandidate("sklearn", fname, 0.85, ev, "sklearn"))
                else:
                    conf = 0.94 if (cfg_kind in ("autotrain", "skops") or "sklearn" in pickle_mods) else 0.88
                    candidates.append(RankedCandidate("sklearn", fname, conf, ev, "sklearn"))
                    candidates.append(RankedCandidate("tree", fname, 0.60, ev, "xgboost"))
            elif low.endswith((".safetensors", ".pt", ".pth", ".ckpt", ".bin")):
                if cfg_kind == "transformers":
                    candidates.append(RankedCandidate("huggingface", fname, 0.95, [f"Transformers config + {fname}"], "huggingface"))
                else:
                    candidates.append(RankedCandidate("pytorch", fname, 0.85, [f"PyTorch weights {fname}"], "pytorch"))

        if cfg_kind == "transformers" and not any(c.loader == "huggingface" for c in candidates):
            candidates.append(RankedCandidate("huggingface", "config.json", 0.90, ["Transformers config.json"], "huggingface"))

        # Deduplicate by (loader, artifact) keeping highest confidence
        seen: Dict[Tuple[str, Optional[str]], RankedCandidate] = {}
        for cand in candidates:
            key = (cand.loader, cand.artifact)
            if key not in seen or cand.confidence > seen[key].confidence:
                seen[key] = cand

        return sorted(seen.values(), key=lambda c: c.confidence, reverse=True)

