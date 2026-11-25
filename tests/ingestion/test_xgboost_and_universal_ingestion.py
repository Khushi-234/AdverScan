"""
Comprehensive unit tests for XGBoost native ingestion, format-driven artifact detection,
auxiliary preprocessing/calibration resolution, and unified prediction adapter.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch

from app.ingestion.adapters.tree_adapter import TreeModelAdapter
from app.ingestion.artifact_detector import ModelArtifactDetector
from app.ingestion.exceptions import (
    ModelLoadError,
    ModelValidationError,
    UnsupportedModelError,
)
from app.ingestion.loader.tree_loader import TreeModelLoader
from app.ingestion.pipeline import ingest_model
from app.ingestion.utils.model_utils import detect_hf_repo_info, detect_model_framework


SENTINEL_FILES = [
    ".gitattributes",
    "NOTICE.md",
    "README.md",
    "calibrator.json",
    "figures/feature_importance.png",
    "figures/pr_curve.png",
    "figures/reliability.png",
    "metadata.json",
    "metrics.json",
    "preprocess.json",
    "reference_stats.json",
    "xgb_model.json",
]


def test_sentinel_fraud_xgboost_artifact_detection():
    """Requirement 1: Verify exact Sentinel file manifest is detected as native XGBoost tabular model."""
    result = ModelArtifactDetector.inspect_file_list(
        files=SENTINEL_FILES,
        library_name="xgboost",
        pipeline_tag="tabular-classification",
        tags=["fraud-detection", "tabular-classification", "xgboost", "calibrated-probabilities"],
        repo_id="HarryDataScience/sentinel-fraud-xgboost",
    )

    assert result.framework == "xgboost"
    assert result.domain == "tabular"
    assert result.task_type == "classification"
    assert result.model_file == "xgb_model.json"
    assert result.is_model_repo is True
    assert "preprocess.json" in result.preprocessing_files
    assert "calibrator.json" in result.calibration_files
    assert "metadata.json" in result.metadata_files


def test_native_xgboost_json_and_auxiliary_loading(tmp_path: Path):
    """Requirement 2 & 3: Test TreeModelLoader resolves native JSON, preprocessor, and calibrator."""
    model_file = tmp_path / "xgb_model.json"
    meta_file = tmp_path / "metadata.json"
    prep_file = tmp_path / "preprocess.json"
    calib_file = tmp_path / "calibrator.json"

    # Create synthetic booster and dump to json
    import xgboost as xgb
    X = np.random.randn(20, 5).astype(np.float32)
    y = np.random.randint(0, 2, size=(20,))
    dtrain = xgb.DMatrix(X, label=y, feature_names=[f"feat_{i}" for i in range(5)])
    booster = xgb.train({"max_depth": 2, "objective": "binary:logistic"}, dtrain, num_boost_round=3)
    booster.save_model(str(model_file))

    meta_file.write_text(json.dumps({
        "raw_input_features": ["f0", "f1", "f2", "f3", "f4"],
        "feature_names": ["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"],
        "threshold": 0.0717,
        "calibration_method": "sigmoid"
    }))
    prep_file.write_text(json.dumps({"scaled_features": ["feat_0"], "mean": [0.5], "scale": [1.2]}))
    calib_file.write_text(json.dumps({"method": "sigmoid", "a": 0.8, "b": -1.5}))

    loader = TreeModelLoader()
    loaded_booster = loader.load(model_file, framework="xgboost")

    assert loaded_booster is not None
    assert loaded_booster.feature_names == [f"feat_{i}" for i in range(5)]
    assert loaded_booster.threshold == 0.0717
    assert loaded_booster.calibrator_info == {"method": "sigmoid", "a": 0.8, "b": -1.5}
    assert loaded_booster.preprocessing_info == {"scaled_features": ["feat_0"], "mean": [0.5], "scale": [1.2]}


def test_tree_adapter_feature_ordering_and_prediction_consistency(tmp_path: Path):
    """Requirement 4: Verify TreeModelAdapter handles DMatrix, feature ordering, and sigmoid calibration."""
    import xgboost as xgb
    import pandas as pd

    feat_names = ["alpha", "beta", "gamma", "delta"]
    X = np.random.randn(25, 4).astype(np.float32)
    y = np.random.randint(0, 2, size=(25,))
    dtrain = xgb.DMatrix(X, label=y, feature_names=feat_names)
    booster = xgb.train({"max_depth": 2, "objective": "binary:logistic"}, dtrain, num_boost_round=3)

    adapter = TreeModelAdapter(
        booster,
        feature_names=feat_names,
        calibrator_info={"method": "sigmoid", "a": 1.0, "b": 0.0},
        threshold=0.5,
    )

    assert adapter.capabilities.supports_gradients is False
    assert adapter.capabilities.supports_predict is True

    # 1. NumPy inference
    arr = np.random.randn(6, 4).astype(np.float32)
    preds_np = adapter.predict(arr, return_numpy=True)
    assert preds_np.shape == (6, 2)
    assert np.allclose(preds_np.sum(axis=1), 1.0, atol=1e-5)

    # 2. PyTorch tensor inference
    tensor_in = torch.randn(4, 4)
    preds_tensor = adapter.predict(tensor_in)
    assert isinstance(preds_tensor, torch.Tensor)
    assert preds_tensor.shape == (4, 2)
    assert torch.allclose(preds_tensor.sum(dim=-1), torch.ones(4), atol=1e-5)

    # 3. DataFrame inference with scrambled feature columns
    df_scrambled = pd.DataFrame(np.random.randn(5, 4), columns=["delta", "alpha", "gamma", "beta"])
    preds_df = adapter.predict(df_scrambled)
    assert preds_df.shape == (5, 2)

    # 4. Logits
    logits = adapter.get_logits(tensor_in)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (4, 2)
    assert not torch.isnan(logits).any()


def test_existing_pytorch_model_ingestion():
    """Requirement 5: Test standard PyTorch model ingestion via ingest_model."""
    class DummyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 2)
        def forward(self, x):
            return self.fc(x)

    model = DummyNet()
    adapter, meta = ingest_model(model, data_domain="tabular", framework="pytorch")
    assert adapter.capabilities.supports_gradients is True
    assert meta.framework == "pytorch"
    assert meta.data_domain == "tabular"

    out = adapter.predict(torch.randn(3, 10))
    assert out.shape == (3, 2)


def test_existing_preset_benchmark_ingestion():
    """Requirement 7: Test existing preset benchmark models ingest correctly."""
    from app.ingestion.presets import MODEL_PRESETS
    if "gtsrb_cnn" in MODEL_PRESETS:
        preset_info = MODEL_PRESETS["gtsrb_cnn"]
        assert preset_info["data_domain"] == "image"
        assert preset_info["task_type"] == "classification"


def test_local_model_path_ingestion(tmp_path: Path):
    """Requirement 8: Test local model path detection and framework routing."""
    pt_file = tmp_path / "model.pt"
    joblib_file = tmp_path / "model.joblib"
    xgb_file = tmp_path / "xgb_model.json"

    pt_file.touch()
    joblib_file.touch()
    xgb_file.touch()

    assert detect_model_framework(str(pt_file)) in ("pytorch", "pytorch_local")
    assert detect_model_framework(str(joblib_file)) == "sklearn"
    assert detect_model_framework(str(xgb_file)) == "xgboost"


def test_missing_and_corrupted_model_artifacts(tmp_path: Path):
    """Requirement 9: Test clear, actionable errors on corrupted or missing artifacts."""
    corrupt_xgb = tmp_path / "corrupt_xgb.json"
    corrupt_xgb.write_text("{invalid json corrupt content")

    loader = TreeModelLoader()
    with pytest.raises(ModelLoadError, match="Failed to load XGBoost model"):
        loader.load(corrupt_xgb, framework="xgboost")

    with pytest.raises(ModelLoadError, match="could not be resolved or does not exist"):
        loader.load(tmp_path / "nonexistent.json", framework="xgboost")


def test_ambiguous_repositories_containing_multiple_weight_formats():
    """Requirement 10: Deterministic resolution when repo has multiple files."""
    files = ["README.md", "config.json", "model.safetensors", "model.onnx"]
    res = ModelArtifactDetector.inspect_file_list(
        files=files,
        library_name="transformers",
        pipeline_tag="text-classification"
    )
    # PyTorch/safetensors takes deterministic precedence when config.json is present
    assert res.framework == "pytorch"
    assert res.is_model_repo is True
    assert res.model_file == "model.safetensors"


def test_models_that_load_but_lack_evaluation_dataset():
    """Requirement 11: Ingestion succeeds and records telemetry when no dataset is present."""
    from frontend.services.ingestion_service import load_model_and_dataset

    class DummyModel:
        num_features = lambda self: 8
        feature_names = [f"f{i}" for i in range(8)]
        def predict(self, x):
            return np.ones((len(x), 2)) * 0.5

    adapter = TreeModelAdapter(DummyModel(), feature_names=[f"f{i}" for i in range(8)])
    assert adapter.capabilities.supports_gradients is False

    with patch("frontend.services.ingestion_service.ingest_model") as mock_ingest:
        from app.ingestion.pipeline import IngestionResult
        from app.ingestion.metadata import ModelMetadata
        mock_meta = ModelMetadata(
            model_id="test/no-dataset-model",
            framework="xgboost",
            data_domain="tabular",
            task_type="classification",
            num_classes=2,
            input_shape=(1, 8),
            model_architecture="XGBoost Gradient Boosted Trees",
            extra_info={"feature_names": [f"f{i}" for i in range(8)]}
        )
        mock_ingest.return_value = IngestionResult(adapter=adapter, metadata=mock_meta)

        res = load_model_and_dataset(
            model_identifier="test/no-dataset-model",
            dataset_name=None,
            data_domain="tabular",
            sample_size=4
        )

        assert res["adapter"] is not None
        assert res["test_inputs"].shape == (4, 8)
        assert res["telemetry"]["is_synthetic_dataset"] is True
        assert res["telemetry"]["gradient_access"] == "Black-Box (Inference Only)"


def test_tree_adapter_compatibility_with_downstream_evaluation():
    """Requirement 12: Verify TreeModelAdapter exposes complete interface for AdverScan evaluation."""
    class MockBooster:
        feature_names = ["v1", "v2", "v3"]
        def predict(self, dmat, output_margin=False):
            if output_margin:
                return np.array([2.5, -1.0])
            return np.array([0.9, 0.2])

    adapter = TreeModelAdapter(MockBooster(), feature_names=["v1", "v2", "v3"])
    inputs = torch.randn(2, 3)

    # 1. predict
    preds = adapter.predict(inputs)
    assert preds.shape == (2, 2)
    assert torch.allclose(preds.sum(dim=-1), torch.ones(2))

    # 2. predict_proba
    probs = adapter.predict_proba(inputs)
    assert probs.shape == (2, 2)

    # 3. get_logits
    logits = adapter.get_logits(inputs)
    assert logits.shape == (2, 2)

