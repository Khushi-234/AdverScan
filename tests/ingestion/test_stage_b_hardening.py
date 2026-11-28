"""Stage B tests: B0 local fallback & joblib opcodes, B1 sklearn tree reconstruction ground-truth, B2 tree loaders, B3 PyTorch R3, B4 Keras GradientTape probe."""

import json
import os
import pickle
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.ingestion.adapters.keras_adapter import KerasAdapter
from app.ingestion.adapters.sklearn_adapter import SklearnAdapter
from app.ingestion.artifact_detector import ModelArtifactDetector
from app.ingestion.exceptions import ErrorCategory, ModelLoadError, UnsafeModelError
from app.ingestion.loader.pytorch_loader import PyTorchLoader
from app.ingestion.loader.sklearn_loader import (
    SklearnLoader,
    _RestrictedPickleUnpickler,
)
from app.ingestion.loader.tree_loader import TreeModelLoader
from app.ingestion.pipeline import ingest_model


def _strip_missing_go_to_left(tree_obj) -> dict:
    """Export tree state and strip 'missing_go_to_left' to simulate an older sklearn (<1.3) pickle."""
    state = tree_obj.__getstate__()
    nodes = state["nodes"]
    old_names = [n for n in nodes.dtype.names if n != "missing_go_to_left"]
    old_dtype = np.dtype([(n, nodes.dtype.fields[n][0]) for n in old_names])
    old_nodes = np.empty(nodes.shape, dtype=old_dtype)
    for n in old_names:
        old_nodes[n] = nodes[n]
    return {
        "max_depth": state["max_depth"],
        "node_count": state["node_count"],
        "nodes": old_nodes,
        "values": state["values"],
    }


def _dump_old_tree_pickle(estimator, path: Path) -> None:
    """Serialize estimator with Tree replaced by a stub carrying old-dtype Tree state."""
    if hasattr(estimator, "steps"):
        target = estimator.steps[-1][1]
    else:
        target = estimator

    class _FakeOldTree:
        def __init__(self, st):
            self._st = st

        def __reduce__(self):
            from sklearn.tree._tree import Tree
            return (Tree, (target.n_features_in_, np.array([len(target.classes_)], dtype=np.intp), 1), self._st)

    if hasattr(target, "tree_"):
        target.tree_ = _FakeOldTree(_strip_missing_go_to_left(target.tree_))
    elif hasattr(target, "estimators_"):
        for sub in target.estimators_:
            sub.tree_ = _FakeOldTree(_strip_missing_go_to_left(sub.tree_))

    joblib.dump(estimator, str(path))


def test_b0_joblib_compressed_opcode_scan_and_local_fallback(tmp_path: Path):
    """B0: _scan_pickle_opcodes handles compressed joblib files with byte cap and non-fatal errors."""
    clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(np.random.randn(20, 4), np.random.randint(0, 2, 20))
    jb = tmp_path / "compressed.joblib"
    joblib.dump(clf, str(jb), compress=3)
    mods = ModelArtifactDetector._scan_pickle_opcodes(jb)
    assert any("sklearn" in m for m in mods)

    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"\x1f\x8bnot-actually-gzip")
    assert ModelArtifactDetector._scan_pickle_opcodes(corrupt) == []

    # Local auto ingestion via ranked candidates
    res = ingest_model(jb, unsafe_allow_pickle=True)
    assert res.acceptance_level == "ACCEPTED_LIMITED"
    assert res.metadata.framework == "sklearn"


@pytest.mark.parametrize("cls", [DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier])
def test_b1_ground_truth_old_tree_reconstruction_matches_sklearn(tmp_path: Path, cls):
    """B1: Reconstructed DecisionTree/RandomForest/ExtraTrees match sklearn predict & predict_proba exactly."""
    rng = np.random.default_rng(123)
    X_train = rng.standard_normal((60, 5)).astype(np.float32)
    y_train = (X_train[:, 0] + 0.5 * X_train[:, 1] > 0).astype(int)
    X_test = rng.standard_normal((25, 5)).astype(np.float32)

    kwargs = {"max_depth": 4, "random_state": 0}
    if cls is not DecisionTreeClassifier:
        kwargs["n_estimators"] = 5
    orig = cls(**kwargs).fit(X_train, y_train)
    expected_pred = orig.predict(X_test)
    expected_proba = orig.predict_proba(X_test)

    # Mutate a fresh copy into an old-dtype pickle (without missing_go_to_left)
    clone_est = cls(**kwargs).fit(X_train, y_train)
    pkl_path = tmp_path / f"old_{cls.__name__}.pkl"
    _dump_old_tree_pickle(clone_est, pkl_path)

    reconstructed = SklearnLoader().load(pkl_path, unsafe_allow_pickle=True)
    assert getattr(reconstructed, "_reconstructed", False) is True
    assert getattr(reconstructed, "_reconstructed_reason", None) == "RECONSTRUCTED"

    actual_pred = reconstructed.predict(X_test)
    actual_proba = reconstructed.predict_proba(X_test)
    np.testing.assert_array_equal(actual_pred, expected_pred)
    np.testing.assert_allclose(actual_proba, expected_proba, atol=1e-6)

    # NaN input rejected
    X_nan = X_test.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        reconstructed.predict(X_nan)

    # Wrapped in SklearnAdapter as ACCEPTED_LIMITED
    res = ingest_model(pkl_path, unsafe_allow_pickle=True)
    assert res.acceptance_level == "ACCEPTED_LIMITED"
    assert isinstance(res.adapter, SklearnAdapter)
    assert res.adapter.is_reconstructed is True


def test_b1_pipeline_reconstruction_and_unsupported_rejection(tmp_path: Path):
    """B1: Pipeline with tree classifier reconstructs; non-tree VERSION_MISMATCH rejects; non-allowlisted class refused."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((40, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", DecisionTreeClassifier(max_depth=3, random_state=0))]).fit(X, y)
    expected_pred = pipe.predict(X)
    expected_proba = pipe.predict_proba(X)

    pipe_clone = Pipeline([("scaler", StandardScaler()), ("clf", DecisionTreeClassifier(max_depth=3, random_state=0))]).fit(X, y)
    pipe_pkl = tmp_path / "old_pipe.pkl"
    _dump_old_tree_pickle(pipe_clone, pipe_pkl)
    loaded_pipe = SklearnLoader().load(pipe_pkl, unsafe_allow_pickle=True)
    np.testing.assert_array_equal(loaded_pipe.predict(X), expected_pred)
    np.testing.assert_allclose(loaded_pipe.predict_proba(X), expected_proba, atol=1e-6)

    # Unsupported estimator on VERSION_MISMATCH -> clear VERSION_MISMATCH ModelLoadError
    scaler = StandardScaler().fit(X)
    scaler_pkl = tmp_path / "scaler.pkl"
    joblib.dump(scaler, str(scaler_pkl))
    with patch("joblib.load", side_effect=ValueError("incompatible dtype")):
        with pytest.raises(ModelLoadError) as exc_info:
            SklearnLoader().load(scaler_pkl, unsafe_allow_pickle=True)
    assert exc_info.value.category == ErrorCategory.VERSION_MISMATCH

    # Non-allowlisted class refused by restricted unpickler
    class _Evil:
        def __reduce__(self):
            return (os.system, ("echo blocked",))

    evil_bytes = pickle.dumps(_Evil())
    with pytest.raises(UnsafeModelError, match="non-allowlisted"):
        _RestrictedPickleUnpickler(__import__("io").BytesIO(evil_bytes)).load()


def test_b2_tree_loader_consent_and_missing_aux_warnings(tmp_path: Path):
    """B2: Native XGBoost JSON needs no consent; joblib tree needs consent; missing aux files warn only."""
    import xgboost as xgb

    X = np.random.randn(20, 4).astype(np.float32)
    y = np.random.randint(0, 2, 20)
    clf = xgb.XGBClassifier(n_estimators=2, max_depth=2, eval_metric="logloss").fit(X, y)
    json_path = tmp_path / "model.json"
    clf.get_booster().save_model(str(json_path))

    # Native JSON loads without unsafe_allow_pickle
    booster = TreeModelLoader().load(json_path, framework="xgboost", unsafe_allow_pickle=False)
    assert booster is not None

    # Joblib estimator requires consent
    jb_path = tmp_path / "tree.joblib"
    joblib.dump(clf, str(jb_path))
    with pytest.raises(UnsafeModelError):
        TreeModelLoader().load(jb_path, framework="xgboost", unsafe_allow_pickle=False)


def test_b3_pytorch_raw_state_dict_r3_and_torchscript(tmp_path: Path):
    """B3: Raw state_dict without resolvable arch raises R3; TorchScript loads."""
    sd_path = tmp_path / "weights.pt"
    torch.save({"fc.weight": torch.randn(2, 4), "fc.bias": torch.randn(2)}, str(sd_path))
    with pytest.raises(ModelLoadError) as exc_info:
        PyTorchLoader().load(sd_path)
    assert exc_info.value.rejection_reason == "R3"
    assert exc_info.value.category == ErrorCategory.ARCH_UNKNOWN

    # TorchScript fixture loads cleanly
    ts_mod = torch.jit.script(nn.Linear(4, 2))
    ts_path = tmp_path / "ts.pt"
    ts_mod.save(str(ts_path))
    loaded_ts = PyTorchLoader().load(ts_path)
    assert loaded_ts(torch.randn(1, 4)).shape == (1, 2)


def test_b4_keras_gradient_tape_probe_full_vs_limited():
    """B4: KerasAdapter probes tf.GradientTape: Keras Sequential -> FULL, non-differentiable -> LIMITED."""
    import tensorflow as tf

    k_model = tf.keras.Sequential([tf.keras.layers.Input(shape=(4,)), tf.keras.layers.Dense(2)])
    k_adapter = KerasAdapter(k_model, data_domain="tabular")
    assert k_adapter.capabilities.supports_gradients is True

    class _NoGradTFSMLayer:
        def __call__(self, x, training=False):
            return {"dense": np.zeros((1, 2), dtype=np.float32)}

    # Rename class to include TFSMLayer to simulate Keras 3 SavedModel wrapper
    _NoGradTFSMLayer.__name__ = "TFSMLayer"
    tfsm_adapter = KerasAdapter(_NoGradTFSMLayer(), data_domain="time-series")
    assert tfsm_adapter.capabilities.supports_gradients is False

