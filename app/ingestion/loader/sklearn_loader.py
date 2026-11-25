"""
Scikit-learn model loader implementation with secure deserialization boundaries.
"""

from pathlib import Path
from typing import Any, List, Optional, Union
import pickle

from app.ingestion.exceptions import ModelLoadError, UnsafeModelError
from app.ingestion.loader.base_loader import BaseModelLoader, LoadResult
from app.ingestion.utils.logger import log_ingestion_event


class SklearnLoader(BaseModelLoader):
    """
    Loads scikit-learn models from in-memory estimators or serialized files (.joblib, .pkl, .pickle).
    Enforces security boundaries against arbitrary deserialization attacks.
    """

    def load(
        self,
        model_path: Union[str, Path, Any],
        unsafe_allow_pickle: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Load a scikit-learn model.

        Args:
            model_path: Path to serialized estimator file or an instantiated estimator instance.
            unsafe_allow_pickle: Explicit user consent flag required to deserialize pickle/joblib files.
            **kwargs: Extra loading options.

        Returns:
            The loaded scikit-learn estimator instance.

        Raises:
            UnsafeModelError: If model_path is a pickle/joblib file and unsafe_allow_pickle is False.
            ModelLoadError: If file not found or deserialized object is not a valid estimator.
        """
        # Case 1: In-memory model instance
        if not isinstance(model_path, (str, Path)):
            if hasattr(model_path, "predict") and (hasattr(model_path, "fit") or hasattr(model_path, "classes_")):
                return model_path
            raise ModelLoadError(
                f"Provided object of type {type(model_path)} is not a valid scikit-learn estimator."
            )

        path = Path(model_path)
        if not path.is_file():
            # If not a local file, check if it is a Hugging Face repository
            path_str = str(model_path).strip()
            if "/" in path_str and not path_str.startswith((".", "/")):
                try:
                    from huggingface_hub import hf_hub_download
                    from app.ingestion.utils.model_utils import detect_hf_repo_info
                    hf_info = detect_hf_repo_info(path_str)
                    candidates: List[str] = []
                    if hf_info.get("model_file") and str(hf_info["model_file"]).endswith((".joblib", ".pkl", ".pickle")):
                        candidates.append(str(hf_info["model_file"]))
                    for f in hf_info.get("files", []):
                        if str(f).endswith((".joblib", ".pkl", ".pickle")) and f not in candidates:
                            candidates.append(f)
                    for c in ("clf.pkl", "sklearn_model.joblib", "model.joblib", "model.pkl", "model.pickle", "estimator.joblib", "model.bin"):
                        if c not in candidates:
                            candidates.append(c)

                    local_f = None
                    for candidate in candidates:
                        try:
                            local_f = hf_hub_download(path_str, candidate)
                            break
                        except Exception:
                            continue
                    if local_f:
                        path = Path(local_f)
                except Exception:
                    pass

        if not path.is_file():
            raise ModelLoadError(f"Model file not found at path: {path}")

        # Security check: Arbitrary deserialization risk
        if not unsafe_allow_pickle:
            raise UnsafeModelError(
                f"Security risk: Refusing to deserialize model from '{path}' without explicit user consent. "
                f"Loading pickled or joblib files from untrusted sources can lead to arbitrary code execution. "
                f"To load this model, pass `unsafe_allow_pickle=True` after verifying the file source."
            )

        log_ingestion_event(
            {
                "status": "SECURITY_WARNING",
                "message": f"Loading model via unsafe deserialization from {path}.",
                "path": str(path),
                "unsafe_allow_pickle": True,
            }
        )

        try:
            # Try joblib if available
            try:
                import joblib
                loaded_obj = joblib.load(str(path))
            except ImportError:
                with open(path, "rb") as f:
                    loaded_obj = pickle.load(f)

            if isinstance(loaded_obj, dict):
                loaded_obj = self._extract_estimator_from_bundle(loaded_obj, path)

            if not (hasattr(loaded_obj, "predict") and (hasattr(loaded_obj, "fit") or hasattr(loaded_obj, "classes_"))):
                raise ModelLoadError(
                    f"Deserialized object from '{path}' is of type {type(loaded_obj)}, "
                    f"which is not a valid scikit-learn estimator with predict()."
                )

            return loaded_obj

        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(f"Failed to load scikit-learn model from {path}: {str(e)}") from e

    @staticmethod
    def _extract_estimator_from_bundle(bundle: dict, path: Union[str, Path]) -> Any:
        """
        Validate and extract an actual scikit-learn estimator/pipeline from a dictionary bundle,
        preserving preprocessing, feature ordering, thresholds, class labels, and metadata.
        """
        est = None
        est_key = None
        for key in ("pipeline", "model", "estimator", "classifier", "clf", "regressor"):
            cand = bundle.get(key)
            if cand is not None and hasattr(cand, "predict") and (hasattr(cand, "fit") or hasattr(cand, "classes_")):
                est = cand
                est_key = key
                break

        if est is None:
            raise ModelLoadError(
                f"Dictionary bundle from '{path}' (keys: {list(bundle.keys())}) does not contain a valid "
                f"scikit-learn estimator or pipeline with predict()."
            )

        # Attach external scaler/preprocessor if present and estimator is not already a Pipeline
        preproc = bundle.get("preprocessor") or bundle.get("scaler")
        if preproc is not None and hasattr(preproc, "transform") and not hasattr(est, "steps"):
            try:
                from sklearn.pipeline import Pipeline
                est = Pipeline([("preprocessor", preproc), ("estimator", est)])
            except Exception:
                pass

        feat_names = (
            bundle.get("feature_names")
            or bundle.get("features")
            or bundle.get("columns")
            or getattr(est, "feature_names_in_", None)
        )
        if feat_names is not None:
            feat_names = [str(f) for f in feat_names]

        bundle_classes = bundle.get("classes") or bundle.get("class_names") or bundle.get("labels")
        if bundle_classes is None and hasattr(est, "classes_"):
            bundle_classes = list(getattr(est, "classes_"))

        bundle_meta = bundle.get("metadata") if isinstance(bundle.get("metadata"), dict) else {}
        bundle_info = {
            "source_key": est_key,
            "raw_bundle_keys": list(bundle.keys()),
            "feature_names": feat_names,
            "threshold": bundle.get("threshold"),
            "classes": bundle_classes,
            "metadata": bundle_meta,
        }
        try:
            est._bundle_info = bundle_info
            if feat_names and not hasattr(est, "feature_names"):
                est.feature_names = feat_names
        except Exception:
            pass
        return est

    @staticmethod
    def inspect_preprocessing(model: Any) -> dict:
        """
        Inspect whether a scikit-learn estimator has attached preprocessing (e.g. Pipeline/ColumnTransformer)
        or is a raw estimator requiring external feature scaling/encoding.
        """
        steps_info: List[str] = []
        has_preprocessing = False
        requires_config = False
        warning: Optional[str] = None

        if hasattr(model, "steps") and isinstance(getattr(model, "steps"), list):
            steps_info = [f"{name}:{step.__class__.__name__}" for name, step in model.steps]
            # Any step prior to the final estimator is preprocessing
            if len(model.steps) > 1:
                has_preprocessing = True
        elif hasattr(model, "transformers") or hasattr(model, "transformers_"):
            has_preprocessing = True
            steps_info = [model.__class__.__name__]
        else:
            requires_config = True
            warning = (
                f"Estimator '{model.__class__.__name__}' has no attached sklearn Pipeline preprocessing. "
                f"If the model was trained on scaled, encoded, or imputed features, provide explicit "
                f"preprocessing configuration (e.g. TabularProcessor preprocessor/feature_names) before evaluation."
            )

        return {
            "has_preprocessing": has_preprocessing,
            "preprocessing_steps": steps_info,
            "requires_preprocessing_config": requires_config,
            "warning": warning,
        }

    def inspect_and_load(
        self,
        model_path: Union[str, Path, Any],
        sample_input: Optional[Any] = None,
        unsafe_allow_pickle: bool = False,
        **kwargs: Any,
    ) -> LoadResult:
        """
        Load model and generate structured inspection evidence.
        """
        model = self.load(model_path, unsafe_allow_pickle=unsafe_allow_pickle, **kwargs)
        source = "in_memory" if not isinstance(model_path, (str, Path)) else "local"

        evidence: List[str] = [
            f"Detected scikit-learn estimator ({model.__class__.__name__})",
            "Model exposes predict()",
        ]
        if hasattr(model, "predict_proba"):
            evidence.append("Model exposes predict_proba()")
        if hasattr(model, "decision_function"):
            evidence.append("Model exposes decision_function()")
        if hasattr(model, "n_features_in_"):
            evidence.append(f"Model fitted on n_features_in_ = {model.n_features_in_}")
        if hasattr(model, "classes_"):
            evidence.append(f"Classes detected: {len(model.classes_)}")

        prep_info = self.inspect_preprocessing(model)
        evidence.append("Domain inferred as tabular from estimator interface")
        if prep_info["has_preprocessing"]:
            evidence.append(f"Attached sklearn Pipeline preprocessing detected: {prep_info['preprocessing_steps']}")
        else:
            evidence.append(prep_info["warning"] or "Preprocessing metadata unavailable; assuming inputs match model space unless configured")

        return LoadResult(
            model=model,
            source=source,
            framework="sklearn",
            evidence=evidence,
            domain_hint="tabular",
            task_hint="classification",
        )
