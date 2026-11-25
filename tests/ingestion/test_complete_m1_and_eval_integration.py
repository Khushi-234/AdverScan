
"""
End-to-end regression tests for AdverScan M1 + Evaluation integration:
- Direct Scikit-learn estimator & dictionary-based model bundles
- Invalid/missing dictionary pipeline diagnostics
- XGBoost JSON and UBJ loading
- Black-box tabular attack compatibility & explicit unsupported gradient attack reporting
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.attack_engine.attacks.tabular.blackbox_query import TabularBlackBoxAttack
from app.attack_engine.config import AttackConfig
from app.ingestion.adapters.sklearn_adapter import SklearnAdapter
from app.ingestion.adapters.tree_adapter import TreeModelAdapter
from app.ingestion.exceptions import ModelLoadError, UnsafeModelError
from app.ingestion.loader.sklearn_loader import SklearnLoader
from app.ingestion.pipeline import ingest_model
from frontend.services.attack_service import (
    get_compatible_attacks,
    get_unsupported_attacks,
)


def test_direct_sklearn_estimator_and_security_consent(tmp_path: Path):
    """Verify direct sklearn estimator requires explicit unsafe_allow_pickle=True and adapts cleanly."""
    model_path = tmp_path / "direct_lr.joblib"
    X = np.random.randn(30, 4)
    y = np.random.randint(0, 2, size=30)
    clf = LogisticRegression().fit(X, y)
    joblib.dump(clf, model_path)

    with pytest.raises(UnsafeModelError):
        ingest_model(str(model_path), data_domain="tabular", framework="sklearn", unsafe_allow_pickle=False)

    adapter, meta = ingest_model(
        str(model_path),
        data_domain="tabular",
        framework="sklearn",
        unsafe_allow_pickle=True,
    )
    assert isinstance(adapter, SklearnAdapter)
    assert adapter.capabilities.supports_gradients is False
    probs = adapter.predict_proba(torch.randn(3, 4))
    assert probs.shape == (3, 2)


def test_sklearn_dictionary_bundle_with_pipeline_and_metadata(tmp_path: Path):
    """Verify dictionary bundle containing 'pipeline' preserves preprocessing, feature order, threshold, labels, and metadata."""
    bundle_path = tmp_path / "fraud_bundle.joblib"
    feat_names = ["amt", "freq", "age"]
    df_train = pd.DataFrame(
        {
            "amt": [10.0, 200.0, 15.0, 500.0, 20.0, 300.0] * 4,
            "freq": [1.0, 5.0, 2.0, 8.0, 1.0, 6.0] * 4,
            "age": [35.0, 22.0, 45.0, 19.0, 50.0, 28.0] * 4,
        }
    )
    y_train = np.array([0, 1, 0, 1, 0, 1] * 4)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    pipe.fit(df_train.values, y_train)

    bundle = {
        "pipeline": pipe,
        "feature_names": feat_names,
        "threshold": 0.42,
        "classes": ["legit", "fraud"],
        "metadata": {"author": "adverscan-test", "version": "1.0"},
    }
    joblib.dump(bundle, bundle_path)

    adapter, meta = ingest_model(
        str(bundle_path),
        data_domain="tabular",
        framework="sklearn",
        unsafe_allow_pickle=True,
    )
    assert isinstance(adapter, SklearnAdapter)
    assert adapter.feature_names == feat_names
    assert adapter.threshold == 0.42
    assert adapter.class_labels == ["legit", "fraud"]
    assert adapter.bundle_metadata == {"author": "adverscan-test", "version": "1.0"}
    assert meta.extra_info["preprocessing"]["has_preprocessing"] is True

    # Scrambled DataFrame columns should be reordered cleanly by SklearnAdapter
    df_scrambled = pd.DataFrame({"age": [30.0, 25.0], "amt": [12.0, 400.0], "freq": [1.0, 7.0]})
    preds = adapter.predict(df_scrambled)
    assert preds.shape == (2, 2)


def test_invalid_dictionary_bundle_missing_pipeline_raises_clear_error(tmp_path: Path):
    """Verify dictionary bundle without a valid estimator/pipeline raises ModelLoadError."""
    bad_bundle = tmp_path / "invalid_bundle.pkl"
    joblib.dump({"feature_names": ["a", "b"], "threshold": 0.5}, bad_bundle)

    loader = SklearnLoader()
    with pytest.raises(ModelLoadError, match="does not contain a valid scikit-learn estimator"):
        loader.load(bad_bundle, unsafe_allow_pickle=True)


def test_xgboost_json_and_ubj_ingestion(tmp_path: Path):
    """Verify both native XGBoost JSON and UBJ formats ingest and predict properly."""
    import xgboost as xgb

    X = np.random.randn(24, 5).astype(np.float32)
    y = np.random.randint(0, 2, size=24)
    dtrain = xgb.DMatrix(X, label=y, feature_names=[f"f{i}" for i in range(5)])
    booster = xgb.train({"max_depth": 2, "objective": "binary:logistic"}, dtrain, num_boost_round=3)

    for fname in ("model.json", "model.ubj"):
        fpath = tmp_path / fname
        booster.save_model(str(fpath))
        adapter, meta = ingest_model(str(fpath), data_domain="tabular", framework="xgboost")
        assert isinstance(adapter, TreeModelAdapter)
        assert meta.framework == "xgboost"
        out = adapter.predict(torch.randn(3, 5))
        assert out.shape == (3, 2)


def test_tree_and_sklearn_blackbox_attack_compatibility_and_unsupported_reporting():
    """Verify tree/sklearn adapters get black-box attacks and explicitly report unsupported gradient attacks."""
    import xgboost as xgb

    X = np.random.randn(20, 4).astype(np.float32)
    y = np.random.randint(0, 2, size=20)
    dtrain = xgb.DMatrix(X, label=y, feature_names=["a", "b", "c", "d"])
    booster = xgb.train({"max_depth": 2, "objective": "binary:logistic"}, dtrain, num_boost_round=3)
    tree_adapter = TreeModelAdapter(booster, feature_names=["a", "b", "c", "d"])

    compatible = get_compatible_attacks(tree_adapter, "tabular")
    compat_ids = [a["id"] for a in compatible]
    assert "tabular_blackbox" in compat_ids
    assert "cap_pgd" not in compat_ids

    unsupported = get_unsupported_attacks(tree_adapter, "tabular")
    unsup_ids = [u["id"] for u in unsupported]
    assert "cap_pgd" in unsup_ids
    assert any("gradient" in u["reason"].lower() for u in unsupported)

    # Execute black-box attack on TreeModelAdapter
    atk = TabularBlackBoxAttack(
        tree_adapter,
        config=AttackConfig(epsilon=0.5, params={"max_iter": 10}),
    )
    x_in = torch.randn(4, 4)
    y_in = torch.tensor([0, 1, 0, 1])
    x_adv = atk.generate(x_in, y_in)
    assert x_adv.shape == x_in.shape
    assert torch.max(torch.abs(x_adv - x_in)).item() <= 0.5 + 1e-5
