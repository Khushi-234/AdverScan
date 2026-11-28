"""
Tree-based model loader implementation for XGBoost, LightGBM, and CatBoost models.
Handles native serialization formats (JSON, UBJ, TXT, CBM) and resolves associated
preprocessing, calibration, and metadata artifacts from local disk and Hugging Face repositories.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from app.ingestion.exceptions import (
    ErrorCategory,
    ModelLoadError,
    UnsafeModelError,
    UnsupportedModelError,
    classify_exception,
)
from app.ingestion.loader.base_loader import BaseModelLoader

logger = logging.getLogger(__name__)


class TreeModelLoader(BaseModelLoader):
    """
    Dedicated loader for tree-based models:
    - XGBoost: loads native Booster via load_model() (.json, .ubj, .model)
    - LightGBM: loads native Booster via Booster(model_file=...) (.txt, .model)
    - CatBoost: loads CatBoostClassifier via load_model() (.cbm)
    """

    def __init__(self) -> None:
        pass

    def load(
        self,
        model_path: Union[str, Path],
        framework: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Load a native tree model from local file path or Hugging Face repository ID.
        """
        model_str = str(model_path).strip()
        is_local_file = Path(model_str).is_file()
        is_local_dir = Path(model_str).is_dir()

        local_model_file: Optional[Path] = None
        preprocessing_data: Optional[Dict[str, Any]] = None
        calibrator_data: Optional[Dict[str, Any]] = None
        metadata_data: Optional[Dict[str, Any]] = None

        fw = (framework or "").lower().strip()

        # Case 1: Local file
        if is_local_file:
            local_model_file = Path(model_str)
            parent_dir = local_model_file.parent
            preprocessing_data = self._read_auxiliary_json(parent_dir, ["preprocess.json", "preprocessor.json", "scaler.json"])
            calibrator_data = self._read_auxiliary_json(parent_dir, ["calibrator.json", "calibration.json"])
            metadata_data = self._read_auxiliary_json(parent_dir, ["metadata.json", "metrics.json"])

            if not fw:
                ext = local_model_file.suffix.lower()
                name = local_model_file.name.lower()
                if ext in (".ubj",) or "xgb" in name:
                    fw = "xgboost"
                elif ext in (".txt",) or "lgb" in name or "lightgbm" in name:
                    fw = "lightgbm"
                elif ext in (".cbm",) or "catboost" in name:
                    fw = "catboost"
                elif ext == ".json":
                    # Check JSON contents for XGBoost or CatBoost signature
                    fw = self._inspect_json_signature(local_model_file)

        # Case 2: Hugging Face repository
        elif "/" in model_str and not model_str.startswith((".", "/")):
            from huggingface_hub import hf_hub_download
            from app.ingestion.utils.model_utils import detect_hf_repo_info

            hf_info = detect_hf_repo_info(model_str)
            fw = fw or hf_info.get("framework", "xgboost")
            model_file_name = hf_info.get("model_file")
            available_files = hf_info.get("files", [])

            if not model_file_name:
                # Find candidate weight file
                for cand in ("xgb_model.json", "xgboost.json", "model.json", "model.ubj", "model.txt", "model.cbm"):
                    if cand in available_files:
                        model_file_name = cand
                        break

            if not model_file_name:
                raise ModelLoadError(
                    f"Hugging Face repository '{model_str}' does not contain recognized tree model weights. "
                    f"Available files: {available_files}"
                )

            try:
                downloaded_path = hf_hub_download(model_str, model_file_name)
                local_model_file = Path(downloaded_path)
            except Exception as exc:
                raise ModelLoadError(f"Failed to download '{model_file_name}' from repository '{model_str}': {exc}") from exc

            # Fetch auxiliary metadata, preprocessing, and calibration if present
            for aux_name in ("preprocess.json", "preprocessor.json", "scaler.json"):
                if aux_name in available_files:
                    try:
                        p = hf_hub_download(model_str, aux_name)
                        with open(p, "r", encoding="utf-8", errors="ignore") as f:
                            preprocessing_data = json.load(f)
                        break
                    except Exception:
                        pass

            for aux_name in ("calibrator.json", "calibration.json"):
                if aux_name in available_files:
                    try:
                        p = hf_hub_download(model_str, aux_name)
                        with open(p, "r", encoding="utf-8", errors="ignore") as f:
                            calibrator_data = json.load(f)
                        break
                    except Exception:
                        pass

            for aux_name in ("metadata.json", "metrics.json"):
                if aux_name in available_files:
                    try:
                        p = hf_hub_download(model_str, aux_name)
                        with open(p, "r", encoding="utf-8", errors="ignore") as f:
                            metadata_data = json.load(f)
                        break
                    except Exception:
                        pass

        elif model_str.startswith(("/", ".")) or Path(model_str).suffix:
            raise ModelLoadError(f"Model file '{model_str}' could not be resolved or does not exist.")
        else:
            raise ModelLoadError(f"Invalid model path or repository identifier '{model_str}'.")

        if not local_model_file or not local_model_file.exists():
            raise ModelLoadError(f"Model file '{local_model_file}' could not be resolved or does not exist.")

        # Native tree formats (XGBoost json/ubj/model, LightGBM txt/model, CatBoost cbm) need NO pickle consent.
        # Joblib/pickle estimators DO require explicit consent.
        if local_model_file.suffix.lower() in (".pkl", ".pickle", ".joblib"):
            unsafe_allow_pickle = bool(kwargs.pop("unsafe_allow_pickle", False))
            if not unsafe_allow_pickle:
                raise UnsafeModelError(
                    f"Security risk: Refusing to deserialize tree estimator '{local_model_file}' "
                    f"without explicit user consent (`unsafe_allow_pickle=True`)."
                )
            from app.ingestion.loader.sklearn_loader import SklearnLoader
            return SklearnLoader().load(local_model_file, unsafe_allow_pickle=True, **kwargs)

        # Missing preprocess/calibrator/metadata files = warning only, never failure
        if not preprocessing_data:
            logger.warning("No preprocessing metadata found for tree model '%s'; proceeding with raw features.", local_model_file.name)
        if not calibrator_data:
            logger.debug("No calibrator metadata found for tree model '%s'.", local_model_file.name)
        if not metadata_data:
            logger.warning("No metadata.json found for tree model '%s'; using model-level attributes.", local_model_file.name)

        # Default fallback to xgboost if still undetermined
        fw = fw or "xgboost"

        # Framework-specific loading
        if fw == "xgboost":
            return self._load_xgboost(
                local_model_file,
                preprocessing_data=preprocessing_data,
                calibrator_data=calibrator_data,
                metadata_data=metadata_data,
            )
        elif fw == "lightgbm":
            return self._load_lightgbm(
                local_model_file,
                preprocessing_data=preprocessing_data,
                metadata_data=metadata_data,
            )
        elif fw == "catboost":
            return self._load_catboost(
                local_model_file,
                preprocessing_data=preprocessing_data,
                metadata_data=metadata_data,
            )
        else:
            raise UnsupportedModelError(f"Unsupported tree model framework '{fw}'. Expected 'xgboost', 'lightgbm', or 'catboost'.")

    def _load_xgboost(
        self,
        model_path: Path,
        preprocessing_data: Optional[Dict[str, Any]] = None,
        calibrator_data: Optional[Dict[str, Any]] = None,
        metadata_data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Load native XGBoost Booster."""
        try:
            import xgboost as xgb
        except ImportError as imp_err:
            raise ModelLoadError(
                f"Model at '{model_path}' is an XGBoost model, which requires the 'xgboost' package. "
                f"Please install 'xgboost' via 'pip install xgboost'."
            ) from imp_err

        try:
            booster = xgb.Booster()
            booster.load_model(str(model_path))
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load XGBoost model from '{model_path}': {exc}",
                category=classify_exception(exc),
            ) from exc

        # Attach discovered metadata and features
        feat_names = list(booster.feature_names) if booster.feature_names is not None else None
        if not feat_names and metadata_data and "feature_names" in metadata_data:
            feat_names = list(metadata_data["feature_names"])
            booster.feature_names = feat_names

        booster.preprocessing_info = preprocessing_data
        booster.calibrator_info = calibrator_data
        booster.metadata_info = metadata_data

        if metadata_data:
            if "raw_input_features" in metadata_data:
                booster.raw_input_features = list(metadata_data["raw_input_features"])
            if "threshold" in metadata_data:
                try:
                    booster.threshold = float(metadata_data["threshold"])
                except (ValueError, TypeError):
                    booster.threshold = 0.5
        else:
            booster.threshold = 0.5

        return booster

    def _load_lightgbm(
        self,
        model_path: Path,
        preprocessing_data: Optional[Dict[str, Any]] = None,
        metadata_data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Load native LightGBM Booster."""
        try:
            import lightgbm as lgb
        except ImportError as imp_err:
            raise ModelLoadError(
                f"Model at '{model_path}' is a LightGBM model, which requires 'lightgbm'. "
                f"Please install 'lightgbm' via 'pip install lightgbm'."
            ) from imp_err

        try:
            booster = lgb.Booster(model_file=str(model_path))
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load LightGBM model from '{model_path}': {exc}",
                category=classify_exception(exc),
            ) from exc

        booster.preprocessing_info = preprocessing_data
        booster.metadata_info = metadata_data
        return booster

    def _load_catboost(
        self,
        model_path: Path,
        preprocessing_data: Optional[Dict[str, Any]] = None,
        metadata_data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Load native CatBoost classifier."""
        try:
            import catboost as cb
        except ImportError as imp_err:
            raise ModelLoadError(
                f"Model at '{model_path}' is a CatBoost model, which requires 'catboost'. "
                f"Please install 'catboost' via 'pip install catboost'."
            ) from imp_err

        try:
            model = cb.CatBoostClassifier()
            model.load_model(str(model_path))
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load CatBoost model from '{model_path}': {exc}",
                category=classify_exception(exc),
            ) from exc

        model.preprocessing_info = preprocessing_data
        model.metadata_info = metadata_data
        return model

    def _inspect_json_signature(self, path: Path) -> str:
        """Inspect JSON keys to distinguish XGBoost from other frameworks."""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if "learner" in data or "version" in data or "gradient_booster" in str(data):
                    return "xgboost"
                if "catboost" in str(data).lower():
                    return "catboost"
        except Exception:
            pass
        return "xgboost"

    def _read_auxiliary_json(self, directory: Path, candidate_names: List[str]) -> Optional[Dict[str, Any]]:
        """Attempt to read auxiliary configuration file from directory."""
        for name in candidate_names:
            p = directory / name
            if p.is_file():
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        return json.load(f)
                except Exception:
                    pass
        return None