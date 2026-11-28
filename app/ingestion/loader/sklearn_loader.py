"""
Scikit-learn model loader implementation with secure deserialization boundaries.

Design Note (Isolated Venv Fallback — NOT IMPLEMENTED):
    For non-tree scikit-learn estimators (e.g. LogisticRegression, SVC, MLPClassifier)
    that fail with VERSION_MISMATCH across major/minor scikit-learn releases, a future
    extension could spawn a short-lived subprocess in an isolated virtualenv pinned to
    the saved scikit-learn version and communicate predictions over JSON/IPC. This is
    intentionally left as a design note only; AdverScan currently reconstructs only
    DecisionTreeClassifier, RandomForestClassifier, and ExtraTreesClassifier in-process.
"""

import bz2
import gzip
import io
import json
import lzma
import pickle
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

from app.ingestion.exceptions import ErrorCategory, ModelLoadError, UnsafeModelError, classify_exception
from app.ingestion.loader.base_loader import BaseModelLoader, LoadResult
from app.ingestion.utils.logger import log_ingestion_event

_ALLOWED_GLOBALS = {
    # Supported tree estimators & internals
    ("sklearn.tree._classes", "DecisionTreeClassifier"),
    ("sklearn.tree._classes", "ExtraTreeClassifier"),
    ("sklearn.tree._tree", "Tree"),
    ("sklearn.ensemble._forest", "RandomForestClassifier"),
    ("sklearn.ensemble._forest", "ExtraTreesClassifier"),
    # Supported pipeline & preprocessing steps
    ("sklearn.pipeline", "Pipeline"),
    ("sklearn.compose._column_transformer", "ColumnTransformer"),
    ("sklearn.compose._column_transformer", "_RemainderColsList"),
    ("sklearn.preprocessing._data", "StandardScaler"),
    ("sklearn.preprocessing._data", "MinMaxScaler"),
    ("sklearn.preprocessing._data", "RobustScaler"),
    ("sklearn.preprocessing._data", "MaxAbsScaler"),
    ("sklearn.preprocessing._encoders", "OneHotEncoder"),
    ("sklearn.preprocessing._encoders", "OrdinalEncoder"),
    ("sklearn.preprocessing._label", "LabelEncoder"),
    ("sklearn.impute._base", "SimpleImputer"),
    ("sklearn.utils._bunch", "Bunch"),
    # NumPy & Joblib array reconstruction primitives
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("joblib.numpy_pickle", "NumpyArrayWrapper"),
    ("joblib.numpy_pickle", "NDArrayWrapper"),
    ("joblib.numpy_pickle", "ZNDArrayWrapper"),
    # Standard safe data structures & pickle reconstruction helpers
    ("collections", "OrderedDict"),
    ("copyreg", "_reconstruct"),
    ("copy_reg", "_reconstruct"),
    ("_codecs", "encode"),
    ("builtins", "dict"),
    ("builtins", "list"),
    ("builtins", "tuple"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "slice"),
    ("builtins", "str"),
    ("builtins", "bytes"),
    ("builtins", "int"),
    ("builtins", "float"),
    ("builtins", "bool"),
    ("builtins", "complex"),
    ("builtins", "object"),
    ("__builtin__", "dict"),
    ("__builtin__", "list"),
    ("__builtin__", "tuple"),
    ("__builtin__", "set"),
    ("__builtin__", "frozenset"),
    ("__builtin__", "slice"),
    ("__builtin__", "str"),
    ("__builtin__", "bytes"),
    ("__builtin__", "int"),
    ("__builtin__", "float"),
    ("__builtin__", "bool"),
    ("__builtin__", "complex"),
    ("__builtin__", "object"),
}
_SUPPORTED_TREE_CLASSIFIERS = (
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
)


def _check_allowlist(module: str, name: str) -> None:
    if (module, name) in _ALLOWED_GLOBALS:
        return
    raise UnsafeModelError(
        f"Restricted unpickler refused non-allowlisted class '{module}.{name}'."
    )


class _TreeStub:
    """Stub for sklearn.tree._tree.Tree capturing state without running Cython __setstate__."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._init_args = args
        self._tree_state: Dict[str, Any] = {}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._tree_state = state


class _RestrictedPickleUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        _check_allowlist(module, name)
        if module == "sklearn.tree._tree" and name == "Tree":
            return _TreeStub
        return super().find_class(module, name)


def _make_restricted_joblib_unpickler():
    import joblib.numpy_pickle as npk

    class _RestrictedNumpyUnpickler(npk.NumpyUnpickler):
        def find_class(self, module: str, name: str) -> Any:
            _check_allowlist(module, name)
            if module == "sklearn.tree._tree" and name == "Tree":
                return _TreeStub
            return super().find_class(module, name)

    return _RestrictedNumpyUnpickler


def _reference_tree_proba(tree_state: Dict[str, Any], X_f32: np.ndarray) -> np.ndarray:
    """Independent scalar reference traversal for verification against pure-numpy predictor."""
    nodes = tree_state["nodes"]
    values = tree_state["values"]
    left = nodes["left_child"]
    right = nodes["right_child"]
    feat = nodes["feature"]
    thresh = nodes["threshold"]
    n_classes = values.shape[-1]
    out = np.empty((X_f32.shape[0], n_classes), dtype=np.float64)
    for i in range(X_f32.shape[0]):
        node_idx = 0
        while left[node_idx] != -1 or right[node_idx] != -1:
            f_idx = int(feat[node_idx])
            if float(X_f32[i, f_idx]) <= float(thresh[node_idx]):
                node_idx = int(left[node_idx])
            else:
                node_idx = int(right[node_idx])
        leaf_val = np.asarray(values[node_idx][0], dtype=np.float64)
        total = float(leaf_val.sum())
        out[i] = leaf_val / total if total > 0.0 else np.full(n_classes, 1.0 / n_classes)
    return out


class _ReconstructedTreePredictor(ClassifierMixin, BaseEstimator):
    """Pure-NumPy predictor reconstructed from a captured sklearn Tree state."""

    def __init__(
        self,
        tree_state: Dict[str, Any],
        classes: np.ndarray,
        n_features_in: int,
        feature_names_in: Optional[np.ndarray] = None,
        saved_version: Optional[str] = None,
        installed_version: Optional[str] = None,
    ) -> None:
        self._tree_state = tree_state
        nodes = tree_state["nodes"]
        self._left = np.asarray(nodes["left_child"], dtype=np.int64)
        self._right = np.asarray(nodes["right_child"], dtype=np.int64)
        self._feature = np.asarray(nodes["feature"], dtype=np.int64)
        self._threshold = np.asarray(nodes["threshold"], dtype=np.float64)
        raw_vals = np.asarray(tree_state["values"], dtype=np.float64)
        if raw_vals.ndim == 3:
            raw_vals = raw_vals[:, 0, :]
        row_sums = raw_vals.sum(axis=1, keepdims=True)
        self._leaf_proba = np.where(row_sums > 0.0, raw_vals / np.where(row_sums > 0.0, row_sums, 1.0), 1.0 / raw_vals.shape[1])
        self.classes_ = np.asarray(classes)
        self.n_features_in_ = int(n_features_in)
        if feature_names_in is not None:
            self.feature_names_in_ = np.asarray(feature_names_in)
        self._reconstructed = True
        self._reconstructed_reason = "RECONSTRUCTED"
        self._saved_sklearn_version = saved_version
        self._installed_sklearn_version = installed_version

    def fit(self, *args: Any, **kwargs: Any) -> "_ReconstructedTreePredictor":
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def _validate_input(self, X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if not np.all(np.isfinite(arr)):
            raise ValueError("Input contains NaN or Inf values, which reconstructed tree predictors reject.")
        return arr

    def fit(self, *args: Any, **kwargs: Any) -> "_ReconstructedForestPredictor":
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def predict_proba(self, X: Any) -> np.ndarray:
        X_f32 = self._validate_input(X)
        n_samples = X_f32.shape[0]
        node_idx = np.zeros(n_samples, dtype=np.int64)
        active = (self._left[node_idx] != -1) | (self._right[node_idx] != -1)
        while np.any(active):
            idx_a = node_idx[active]
            feats = self._feature[idx_a]
            vals = X_f32[active, feats]
            go_left = vals <= self._threshold[idx_a]
            node_idx[active] = np.where(go_left, self._left[idx_a], self._right[idx_a])
            active = (self._left[node_idx] != -1) | (self._right[node_idx] != -1)
        return self._leaf_proba[node_idx]

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class _ReconstructedForestPredictor(ClassifierMixin, BaseEstimator):
    """Pure-NumPy ensemble predictor averaging reconstructed per-tree probabilities."""

    def __init__(
        self,
        trees: List[_ReconstructedTreePredictor],
        classes: np.ndarray,
        n_features_in: int,
        feature_names_in: Optional[np.ndarray] = None,
        saved_version: Optional[str] = None,
        installed_version: Optional[str] = None,
    ) -> None:
        self._trees = trees
        self.classes_ = np.asarray(classes)
        self.n_features_in_ = int(n_features_in)
        if feature_names_in is not None:
            self.feature_names_in_ = np.asarray(feature_names_in)
        self._reconstructed = True
        self._reconstructed_reason = "RECONSTRUCTED"
        self._saved_sklearn_version = saved_version
        self._installed_sklearn_version = installed_version

    def fit(self, *args: Any, **kwargs: Any) -> "_ReconstructedForestPredictor":
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def predict_proba(self, X: Any) -> np.ndarray:
        probas = [t.predict_proba(X) for t in self._trees]
        return np.mean(probas, axis=0)

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def _verify_reconstructed_predictor(
    predictor: Union[_ReconstructedTreePredictor, _ReconstructedForestPredictor],
    tree_states: List[Dict[str, Any]],
) -> None:
    """Verify reconstructed predictor against an independent scalar reference traversal."""
    rng = np.random.default_rng(42)
    X_ref = rng.standard_normal((8, predictor.n_features_in_)).astype(np.float32)
    ref_probas = [_reference_tree_proba(st, X_ref) for st in tree_states]
    expected_proba = ref_probas[0] if len(ref_probas) == 1 else np.mean(ref_probas, axis=0)
    expected_pred = predictor.classes_[np.argmax(expected_proba, axis=1)]
    actual_proba = predictor.predict_proba(X_ref)
    actual_pred = predictor.predict(X_ref)
    if not np.allclose(actual_proba, expected_proba, atol=1e-6) or not np.array_equal(actual_pred, expected_pred):
        raise ValueError("Reconstructed tree predictor failed reference traversal verification.")


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
            if classify_exception(e) == ErrorCategory.VERSION_MISMATCH:
                return self._attempt_tree_reconstruction(path, e)
            raise ModelLoadError(
                f"Failed to load scikit-learn model from {path}: {str(e)}",
                category=classify_exception(e),
            ) from e

    @staticmethod
    def _read_saved_sklearn_version(path: Path, loaded_obj: Any = None) -> Optional[str]:
        ver = getattr(loaded_obj, "_sklearn_version", None)
        if ver:
            return str(ver)
        cfg_path = path.parent / "config.json"
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                sk = cfg.get("sklearn") or {}
                if isinstance(sk, dict) and sk.get("environment"):
                    for pkg in sk["environment"]:
                        if isinstance(pkg, str) and "scikit-learn" in pkg:
                            return pkg
            except Exception:
                pass
        return None

    @staticmethod
    def _restricted_unpickle(path: Path) -> Any:
        raw = path.read_bytes()
        data = raw
        if raw.startswith(b"\x1f\x8b"):
            data = gzip.decompress(raw)
        elif raw.startswith(b"BZh"):
            data = bz2.decompress(raw)
        elif raw.startswith(b"\xfd7zXZ"):
            data = lzma.decompress(raw)
        elif len(raw) >= 2 and raw[0] == 0x78 and raw[1] in (0x01, 0x5E, 0x9C, 0xDA):
            data = zlib.decompress(raw)

        try:
            unpickler_cls = _make_restricted_joblib_unpickler()
            return unpickler_cls(str(path), io.BytesIO(data), ensure_native_byte_order="auto").load()
        except UnsafeModelError:
            raise
        except Exception:
            return _RestrictedPickleUnpickler(io.BytesIO(data)).load()

    def _attempt_tree_reconstruction(self, path: Path, orig_exc: Exception) -> Any:
        import sklearn
        installed_ver = getattr(sklearn, "__version__", "unknown")
        saved_ver = self._read_saved_sklearn_version(path)
        try:
            loaded_obj = self._restricted_unpickle(path)
        except UnsafeModelError:
            raise
        except Exception as rec_exc:
            raise ModelLoadError(
                f"scikit-learn VERSION_MISMATCH (saved={saved_ver or 'unknown'}, installed={installed_ver}): {orig_exc}",
                category=ErrorCategory.VERSION_MISMATCH,
            ) from rec_exc

        if isinstance(loaded_obj, dict):
            loaded_obj = self._extract_estimator_from_bundle(loaded_obj, path)

        saved_ver = self._read_saved_sklearn_version(path, loaded_obj) or saved_ver

        # Support bare tree classifiers or Pipeline whose final step is a supported tree classifier
        is_pipeline = hasattr(loaded_obj, "steps") and isinstance(getattr(loaded_obj, "steps", None), list)
        target_est = loaded_obj.steps[-1][1] if is_pipeline else loaded_obj
        cls_name = type(target_est).__name__
        if cls_name not in _SUPPORTED_TREE_CLASSIFIERS:
            raise ModelLoadError(
                f"scikit-learn VERSION_MISMATCH for unsupported estimator '{cls_name}' "
                f"(saved={saved_ver or 'unknown'}, installed={installed_ver}): {orig_exc}",
                category=ErrorCategory.VERSION_MISMATCH,
            ) from orig_exc

        classes = getattr(target_est, "classes_", None)
        n_features = getattr(target_est, "n_features_in_", None)
        feat_names = getattr(target_est, "feature_names_in_", None)
        if classes is None or n_features is None:
            raise ModelLoadError(
                f"Cannot reconstruct '{cls_name}': missing classes_ or n_features_in_ (saved={saved_ver}, installed={installed_ver}).",
                category=ErrorCategory.VERSION_MISMATCH,
            ) from orig_exc

        tree_states: List[Dict[str, Any]] = []
        if cls_name == "DecisionTreeClassifier":
            st = getattr(getattr(target_est, "tree_", None), "_tree_state", None)
            if st is None and hasattr(target_est, "tree_") and hasattr(target_est.tree_, "__getstate__"):
                st = target_est.tree_.__getstate__()
            if not isinstance(st, dict):
                raise ModelLoadError("Missing Tree state during reconstruction.", category=ErrorCategory.VERSION_MISMATCH) from orig_exc
            tree_states.append(st)
            predictor: Any = _ReconstructedTreePredictor(
                st, classes, n_features, feat_names, saved_ver, installed_ver
            )
        else:
            sub_ests = getattr(target_est, "estimators_", []) or []
            trees: List[_ReconstructedTreePredictor] = []
            for sub in sub_ests:
                st = getattr(getattr(sub, "tree_", None), "_tree_state", None)
                if st is None and hasattr(sub, "tree_") and hasattr(sub.tree_, "__getstate__"):
                    st = sub.tree_.__getstate__()
                if not isinstance(st, dict):
                    raise ModelLoadError("Missing ensemble Tree state during reconstruction.", category=ErrorCategory.VERSION_MISMATCH) from orig_exc
                tree_states.append(st)
                trees.append(_ReconstructedTreePredictor(st, classes, n_features, feat_names, saved_ver, installed_ver))
            if not trees:
                raise ModelLoadError("Empty forest during reconstruction.", category=ErrorCategory.VERSION_MISMATCH) from orig_exc
            predictor = _ReconstructedForestPredictor(
                trees, classes, n_features, feat_names, saved_ver, installed_ver
            )

        try:
            _verify_reconstructed_predictor(predictor, tree_states)
        except Exception as ver_exc:
            raise ModelLoadError(
                f"Reconstructed tree verification failed (saved={saved_ver or 'unknown'}, installed={installed_ver}): {ver_exc}",
                category=ErrorCategory.VERSION_MISMATCH,
            ) from ver_exc

        log_ingestion_event(
            {
                "status": "RECONSTRUCTED",
                "estimator": cls_name,
                "path": str(path),
                "saved_sklearn_version": saved_ver,
                "installed_sklearn_version": installed_ver,
                "n_trees": len(tree_states),
            }
        )

        if is_pipeline:
            loaded_obj.steps[-1] = (loaded_obj.steps[-1][0], predictor)
            loaded_obj._reconstructed = True
            loaded_obj._reconstructed_reason = "RECONSTRUCTED"
            return loaded_obj
        return predictor

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
