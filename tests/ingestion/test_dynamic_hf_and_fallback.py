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
