"""
Unit tests for M1 Hugging Face dynamic ingestion and synthetic fallback improvements.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch

from app.ingestion.exceptions import (
    ConfigurationError,
    ModelLoadError,
    ModelValidationError,
    UnsupportedModelError,
)
from app.ingestion.loader.huggingface_loader import HuggingFaceLoader
from app.ingestion.metadata import ModelMetadata
from app.ingestion.model_registry import ModelPresetRegistry, detect_model_framework
from app.ingestion.pipeline import ingest_model
from app.ingestion.utils.model_utils import detect_hf_repo_info
from app.ingestion.validation.validator import ModelValidator
try:
    from frontend.services.ingestion_service import _generate_domain_sample_batch
except ImportError:
    _generate_domain_sample_batch = None


def test_detect_hf_repo_info_sklearn():
    """Test detect_hf_repo_info identifies Scikit-learn repos with model.pkl."""
    mock_files = ["README.md", "model.pkl", "config.json"]
    mock_model_info = MagicMock()
    mock_model_info.tags = ["tabular", "sklearn"]
    mock_model_info.library_name = "sklearn"
    mock_model_info.pipeline_tag = "tabular-classification"

    with patch("huggingface_hub.HfApi.list_repo_files", return_value=mock_files):
        with patch("huggingface_hub.HfApi.model_info", return_value=mock_model_info):
            info = detect_hf_repo_info("scikit-learn/cancer-prediction-trees")
            assert info["framework"] == "sklearn"
            assert info["domain"] == "tabular"
            assert info["model_file"] == "model.pkl"


def test_detect_hf_repo_info_keras_savedmodel():
    """Test detect_hf_repo_info identifies Keras/TF SavedModel repos."""
    mock_files = ["README.md", "saved_model.pb", "keras_metadata.pb", "variables/variables.data-00000-of-00001"]
    mock_model_info = MagicMock()
    mock_model_info.tags = ["tf-keras", "timeseries"]
    mock_model_info.library_name = "keras"
    mock_model_info.pipeline_tag = "time-series-classification"

    with patch("huggingface_hub.HfApi.list_repo_files", return_value=mock_files):
        with patch("huggingface_hub.HfApi.model_info", return_value=mock_model_info):
            info = detect_hf_repo_info("keras-io/timeseries_transformer_classification")
            assert info["framework"] in ("keras", "tensorflow")
            assert info["domain"] == "time-series"


def test_build_runtime_config_dynamic_routing_sklearn():
    """Test runtime config builds sklearn framework for HF sklearn models."""
    with patch(
        "app.ingestion.utils.model_utils.detect_hf_repo_info",
        return_value={"framework": "sklearn", "domain": "tabular", "model_file": "model.pkl"},
    ):
        config = ModelPresetRegistry.build_runtime_config("scikit-learn/cancer-prediction-trees")
        assert config["framework"] == "sklearn"
        assert config["data_domain"] == "tabular"


def test_hf_loader_keras_missing_tensorflow_raises_descriptive_error():
    """Test HuggingFaceLoader raises informative ModelLoadError when tensorflow is missing."""
    with patch(
        "app.ingestion.utils.model_utils.detect_hf_repo_info",
        return_value={"framework": "keras", "domain": "time-series", "model_file": "saved_model.pb"},
    ):
        loader = HuggingFaceLoader()
        with patch.dict("sys.modules", {"tensorflow": None, "keras": None}):
            with pytest.raises(ModelLoadError, match="requires the 'tensorflow' package"):
                loader.load("keras-io/timeseries_transformer_classification", domain="time-series")


def test_generate_domain_sample_batch_dynamic_tabular_discovery():
    """Test tabular sample generation dynamically discovers dimensions from raw_model."""
    if _generate_domain_sample_batch is None:
        pytest.skip("frontend.services.ingestion_service not available on this branch")
    class DummySklearnModel:
        n_features_in_ = 30

    meta = ModelMetadata(model_id="test-tabular", framework="sklearn", data_domain="tabular", task_type="classification")
    inputs, labels = _generate_domain_sample_batch(meta, sample_size=5, raw_model=DummySklearnModel())
    assert inputs.shape == (5, 30)
    assert labels.shape == (5,)


def test_generate_domain_sample_batch_unknown_tabular_raises():
    """Test tabular sample generation raises ConfigurationError when dimensions are completely unknown."""
    if _generate_domain_sample_batch is None:
        pytest.skip("frontend.services.ingestion_service not available on this branch")
    meta = ModelMetadata(model_id="unknown-tabular", framework="sklearn", data_domain="tabular", task_type="classification")
    with pytest.raises(ConfigurationError, match="Cannot determine number of tabular features"):
        _generate_domain_sample_batch(meta, sample_size=5, raw_model=object())


def test_validator_with_non_empty_forward_pass():
    """Test ModelValidator correctly verifies non-empty forward pass and checks for NaN/empty outputs."""
    from app.ingestion.adapters.pytorch_adapter import PyTorchAdapter
    import torch.nn as nn

    class GoodModel(nn.Module):
        def forward(self, x):
            return torch.ones((x.shape[0], 3))

    class EmptyOutputModel(nn.Module):
        def forward(self, x):
            return torch.empty((0, 3))

    good_adapter = PyTorchAdapter(GoodModel(), device="cpu")
    empty_adapter = PyTorchAdapter(EmptyOutputModel(), device="cpu")

    assert ModelValidator.validate(good_adapter, sample_input=torch.zeros((2, 4))) is True

    with pytest.raises(ModelValidationError, match="empty shape"):
        ModelValidator.validate(empty_adapter, sample_input=torch.zeros((2, 4)))


def test_detect_hf_repo_info_distinguishes_dataset_and_source_only_repos():
    """Test detect_hf_repo_info distinguishes dataset and source-only repos from model repos."""
    from app.ingestion.utils.model_utils import _HF_REPO_INFO_CACHE

    _HF_REPO_INFO_CACHE.pop("org/dataset-only", None)
    _HF_REPO_INFO_CACHE.pop("org/source-only", None)
    _HF_REPO_INFO_CACHE.pop("org/missing-weights", None)

    mock_info = MagicMock()
    mock_info.tags = []
    mock_info.library_name = None
    mock_info.pipeline_tag = None

    with patch("huggingface_hub.HfApi.model_info", return_value=mock_info):
        with patch("huggingface_hub.HfApi.list_repo_files", return_value=["README.md", "dataset_info.json", "train.parquet"]):
            ds_info = detect_hf_repo_info("org/dataset-only")
            assert ds_info["is_model_repo"] is False
            assert ds_info["repo_type"] == "dataset"

        with patch("huggingface_hub.HfApi.list_repo_files", return_value=["README.md", "train.py", "model.py"]):
            src_info = detect_hf_repo_info("org/source-only")
            assert src_info["is_model_repo"] is False
            assert src_info["repo_type"] == "source_only"

        mock_info.tags = ["pytorch"]
        mock_info.library_name = "transformers"
        with patch("huggingface_hub.HfApi.list_repo_files", return_value=["README.md", "config.json"]):
            inc_info = detect_hf_repo_info("org/missing-weights")
            assert inc_info["is_model_repo"] is False
            assert inc_info["repo_type"] == "incomplete"


def test_hf_loader_rejects_dataset_source_only_and_missing_weights():
    """Test HuggingFaceLoader raises clear ModelLoadError for dataset, source-only, and weightless repos."""
    loader = HuggingFaceLoader()

    with patch(
        "app.ingestion.utils.model_utils.detect_hf_repo_info",
        return_value={"framework": "unknown", "domain": "tabular", "is_model_repo": False, "repo_type": "dataset", "files": ["train.parquet"]},
    ):
        with pytest.raises(ModelLoadError, match="dataset repository"):
            loader.load("org/dataset-only", domain="tabular")

    with patch(
        "app.ingestion.utils.model_utils.detect_hf_repo_info",
        return_value={"framework": "unknown", "domain": "tabular", "is_model_repo": False, "repo_type": "source_only", "files": ["train.py"]},
    ):
        with pytest.raises(ModelLoadError, match="only source code/documentation"):
            loader.load("org/source-only", domain="tabular")

    with patch(
        "app.ingestion.utils.model_utils.detect_hf_repo_info",
        return_value={"framework": "pytorch", "domain": "tabular", "is_model_repo": False, "repo_type": "incomplete", "files": ["config.json"]},
    ):
        with pytest.raises(ModelLoadError, match="does not contain recognized serialized model weights"):
            loader.load("org/missing-weights", domain="tabular")


def test_hf_loader_onnx_missing_dependency_and_onnx_adapter():
    """Test ONNX missing dependency error and ONNXAdapter / KerasAdapter unified interfaces."""
    from app.ingestion.adapters import KerasAdapter, ONNXAdapter

    with patch(
        "app.ingestion.utils.model_utils.detect_hf_repo_info",
        return_value={"framework": "onnx", "domain": "tabular", "is_model_repo": True, "repo_type": "model", "model_file": "model.onnx", "files": ["model.onnx"]},
    ):
        loader = HuggingFaceLoader()
        with patch.dict("sys.modules", {"onnxruntime": None}):
            with pytest.raises(ModelLoadError, match="requires 'onnxruntime'"):
                loader.load("org/onnx-model", domain="tabular")

    # Test ONNXAdapter with duck-typed InferenceSession
    mock_session = MagicMock()
    mock_input_meta = MagicMock()
    mock_input_meta.name = "x"
    mock_session.get_inputs.return_value = [mock_input_meta]
    mock_session.run.return_value = [np.array([[0.2, 0.8]], dtype=np.float32)]

    onnx_adapter = ONNXAdapter(mock_session, data_domain="tabular")
    assert onnx_adapter.capabilities.supports_gradients is False
    assert onnx_adapter.capabilities.supports_predict is True
    out = onnx_adapter.predict(torch.zeros((1, 4)))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 2)
    assert ModelValidator.validate(onnx_adapter, sample_input=torch.zeros((1, 4))) is True

    # Test KerasAdapter with duck-typed Keras model
    mock_keras = MagicMock()
    mock_keras.predict.return_value = np.array([[0.1, 0.9]], dtype=np.float32)
    keras_adapter = KerasAdapter(mock_keras, data_domain="time-series")
    assert keras_adapter.capabilities.supports_gradients is False
    k_out = keras_adapter.predict(torch.zeros((1, 10, 2)))
    assert isinstance(k_out, torch.Tensor)
    assert k_out.shape == (1, 2)
    assert ModelValidator.validate(keras_adapter, sample_input=torch.zeros((1, 10, 2))) is True


def test_sklearn_loader_preprocessing_inspection_pipeline_vs_raw():
    """Test SklearnLoader.inspect_preprocessing distinguishes Pipeline from raw estimator."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from app.ingestion.loader.sklearn_loader import SklearnLoader

    X = np.random.randn(20, 4)
    y = (X[:, 0] > 0).astype(int)

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())]).fit(X, y)
    prep_pipe = SklearnLoader.inspect_preprocessing(pipe)
    assert prep_pipe["has_preprocessing"] is True
    assert prep_pipe["requires_preprocessing_config"] is False
    assert any("StandardScaler" in s for s in prep_pipe["preprocessing_steps"])

    raw = LogisticRegression().fit(X, y)
    prep_raw = SklearnLoader.inspect_preprocessing(raw)
    assert prep_raw["has_preprocessing"] is False
    assert prep_raw["requires_preprocessing_config"] is True
    assert "no attached sklearn Pipeline preprocessing" in prep_raw["warning"]


def test_parse_yaml_frontmatter_in_readme():
    """Test _parse_yaml_frontmatter extracts library_name, pipeline_tag, and tags."""
    from app.ingestion.utils.model_utils import _parse_yaml_frontmatter

    readme = """---
library_name: sklearn
pipeline_tag: tabular-classification
tags:
  - tabular
  - sklearn
---
# Model Card
"""
    parsed = _parse_yaml_frontmatter(readme)
    assert parsed["library_name"] == "sklearn"
    assert parsed["pipeline_tag"] == "tabular-classification"
    assert "tabular" in parsed["tags"]
    assert "sklearn" in parsed["tags"]
